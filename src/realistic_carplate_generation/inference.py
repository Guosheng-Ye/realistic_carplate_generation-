"""Pinned CPU ONNX inference; all corner/OCR outputs are predictions, never human GT."""
import math
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from .common import require, asset_path, asset_check, read_json


def _runtime():
    """Import the optional runtime only when a model is actually needed."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Model inference requires pip install 'realistic-carplate-generation[detection]'") from exc
    ort.disable_telemetry_events()
    return ort


OCR_DECODE_CONTRACT = 'e3_dual_v2_other_reject'

def cover_box(coords, width, height):
    if len(coords)!=4 or not all(math.isfinite(x) for x in coords):
        raise ValueError('nonfinite or malformed box')
    x1,y1,x2,y2=coords
    if x2<=x1 or y2<=y1:raise ValueError('empty box')
    x=max(0,min(width,math.floor(x1)));y=max(0,min(height,math.floor(y1)))
    r=max(0,min(width,math.ceil(x2)));b=max(0,min(height,math.ceil(y2)))
    if r<=x or b<=y:raise ValueError('out of frame box')
    return [x,y,r-x,b-y]

def valid_quad(points,width,height):
    if len(points)!=4 or any(len(p)!=2 or any(not math.isfinite(v) for v in p) for p in points):return False
    if any(x<0 or y<0 or x>width or y>height for x,y in points):return False
    for i in range(4):
        a,b,c=points[i],points[(i+1)%4],points[(i+2)%4]
        if (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])<=1:return False
    return True

def axis_corners(box):
    x,y,w,h=box
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]

def iou(a,b):
    x,y,w,h=a;u,v,s,t=b
    inter=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    return inter/(w*h+s*t-inter) if inter else 0.0

def merge_proposals(rows, threshold=.45):
    kept=[]
    # Geometry-capable proposals take precedence across teachers, then score.
    # This is an explicit review/recall policy, not cross-model score calibration.
    for row in sorted(rows,key=lambda x:(x['corners_source']=='model_predicted',x['score']),reverse=True):
        if all(iou(row['bbox'],k['bbox'])<threshold for k in kept):kept.append(row)
    return kept

def decode_hyper(raw,width,height,scale,left,top,confidence=.2):
    import numpy as np
    if raw.ndim!=3 or raw.shape[0]!=1 or raw.shape[2]!=15 or not np.isfinite(raw).all():
        raise ValueError('unexpected HyperLPR detection tensor')
    rows=[]
    for d in raw[0]:
        score=float(d[4]*max(d[13:15]))
        if score<confidence:continue
        cx,cy,w,h=map(float,d[:4])
        xyxy=[(cx-w/2-left)/scale,(cy-h/2-top)/scale,(cx+w/2-left)/scale,(cy+h/2-top)/scale]
        points=[[(float(d[5+2*k])-left)/scale,(float(d[6+2*k])-top)/scale] for k in range(4)]
        measured=valid_quad(points,width,height)
        if measured:
            xyxy=[min(xyxy[0],min(p[0] for p in points)),min(xyxy[1],min(p[1] for p in points)),
                  max(xyxy[2],max(p[0] for p in points)),max(xyxy[3],max(p[1] for p in points))]
        try:box=cover_box(xyxy,width,height)
        except ValueError:continue
        if box[2]<2 or box[3]<2:continue
        rows.append(dict(bbox=box,corners=points if measured else axis_corners(box),score=score,
                         corners_source='model_predicted' if measured else 'axis_aligned_proposal_not_measured',
                         teacher='HyperLPR3',layer=int(np.argmax(d[13:15]))))
    return merge_proposals(rows)

class PlateTeacher:
    """Pinned HyperLPR3 ONNX files; no import-time downloads or pickle loading."""
    def __init__(self,models,confidence=.2):
        ort = _runtime()
        self.models=Path(models);self.confidence=confidence
        self.provenance=json.loads((self.models/'provenance.json').read_text())
        self._verify_model('y5fu_640x_sim.onnx')
        self.tokens=None;self.rec=None;self.rec_name=None
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
        self.det=ort.InferenceSession(str(self.models/'y5fu_640x_sim.onnx'),options,providers=['CPUExecutionProvider'])
        self.det_name=self.det.get_inputs()[0].name
        if self.det.get_inputs()[0].shape!=[1,3,640,640]:raise ValueError('teacher input mismatch')

    def _verify_model(self, name):
        import hashlib
        records = [r for r in self.provenance['models'] if r['name'] == name]
        if len(records) != 1 or hashlib.sha256((self.models/name).read_bytes()).hexdigest() != records[0]['sha256']:
            raise ValueError('teacher checksum mismatch')

    def _load_rec(self):
        if self.rec is not None:
            return
        import hashlib
        self._verify_model('rpv3_mdict_160_r3.onnx')
        token_bytes=(self.models/'tokens.json').read_bytes()
        if hashlib.sha256(token_bytes).hexdigest()!=self.provenance['tokens_sha256']:
            raise ValueError('token checksum mismatch')
        tokens=json.loads(token_bytes)
        ort = _runtime()
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
        rec=ort.InferenceSession(str(self.models/'rpv3_mdict_160_r3.onnx'),options,providers=['CPUExecutionProvider'])
        self.rec_name=rec.get_inputs()[0].name
        self.tokens=tokens;self.rec=rec

    def detect(self,image):
        import cv2,numpy as np
        h,w=image.shape[:2];r=min(640/h,640/w);nh,nw=max(1,int(h*r)),max(1,int(w*r))
        top,left=(640-nh)//2,(640-nw)//2
        canvas=np.zeros((640,640,3),np.uint8)
        canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
        data=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255)
        raw=self.det.run(None,{self.det_name:data})[0]
        return decode_hyper(raw,w,h,r,left,top,self.confidence)

    def recognize(self,image):
        import cv2,numpy as np
        self._load_rec()
        h,w=image.shape[:2]
        resized_w=min(160,max(48,math.ceil(48*w/h)))
        scaled=cv2.resize(image,(resized_w,48)).astype(np.float32)
        data=np.zeros((1,3,48,160),np.float32)
        data[0,:,:,:resized_w]=(scaled.transpose(2,0,1)-127.5)/127.5
        prod=self.rec.run(None,{self.rec_name:data})[0]
        if prod.ndim!=3 or prod.shape[0]!=1 or prod.shape[2] not in (len(self.tokens),len(self.tokens)+1) or not np.isfinite(prod).all():
            raise ValueError('teacher recognition output mismatch')
        values=prod[0].astype(np.float64)
        if values.min()<0 or values.max()>1.00001:
            values=np.exp(values-values.max(axis=1,keepdims=True));values/=values.sum(axis=1,keepdims=True)
        ids=values.argmax(axis=1);previous=-1;text='';scores=[];unknown=[]
        for idx,row in zip(ids,values):
            if idx and idx!=previous:
                # Historical upstream graph has 78 classes but token list has 77.
                # Never guess the missing symbol or silently map it to blank.
                text+=self.tokens[int(idx)] if int(idx)<len(self.tokens) else '\ufffd'
                if int(idx)>=len(self.tokens):unknown.append(int(idx))
                scores.append(float(row[idx]))
            previous=idx
        return dict(text=text,score=sum(scores)/len(scores) if scores else 0.0,unsupported_token_ids=unknown)

class Detector:
    def __init__(self,path,confidence=.5):
        ort = _runtime()
        options=ort.SessionOptions();options.intra_op_num_threads=2
        self.session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        require(self.session.get_inputs()[0].shape==[1,3,640,640],'detector must have fixed 640 input')
        self.name=self.session.get_inputs()[0].name;self.confidence=confidence

    def run(self,image):
        h,w=image.shape[:2];ratio=min(640/w,640/h);nw,nh=math.floor(w*ratio+.5),math.floor(h*ratio+.5)
        left,top=(640-nw)//2,(640-nh)//2
        canvas=np.full((640,640,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
        x=np.ascontiguousarray((cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype('float64')*(1/255)).astype('float32').transpose(2,0,1)[None])
        values=self.session.run(None,{self.name:x})
        require(all(np.isfinite(v).all() for v in values),'nonfinite detector output')
        if len(values)==1:
            require(values[0].shape==(1,300,6),'expected one2one end-to-end output [1,300,6]')
            boxes=values[0][0]
        else:
            require(len(values)==6,'expected one2one six raw heads')
            boxes=[]
            for stride,cls,reg in zip((8,16,32),values[::2],values[1::2]):
                size=640//stride
                require(cls.shape==(1,size,size,1) and reg.shape==(1,size,size,4),'raw detector head order/layout mismatch')
                cls=cls.reshape(size,size);reg=reg.reshape(size,size,4)
                scores=1/(1+np.exp(-np.clip(cls,-80,80)))
                for gy,gx in zip(*np.nonzero(scores>=self.confidence)):
                    a,b,c,d=reg[gy,gx];cx,cy=gx+.5,gy+.5
                    boxes.append([(cx-a)*stride,(cy-b)*stride,(cx+c)*stride,(cy+d)*stride,scores[gy,gx],0])
        found=[];filtered=[]
        for x1,y1,x2,y2,score,label in boxes:
            if score<self.confidence:continue
            require(label==0,'unexpected detector class')
            x1,y1=max(0.,(float(x1)-left)/ratio),max(0.,(float(y1)-top)/ratio)
            x2,y2=min(w,(float(x2)-left)/ratio),min(h,(float(y2)-top)/ratio)
            box=[x1,y1,x2-x1,y2-y1]
            if x2<=x1 or y2<=y1:filtered.append(dict(bbox=box,reason='empty_after_clipping'));continue
            # Audit the legacy filter; do not silently remove a valid side-view ROI.
            legacy_pass=box[2]>=30 and box[3]>=15 and 2<=box[2]/box[3]<=6
            found.append(dict(bbox=box,score=float(score),legacy_shape_filter_would_reject=not legacy_pass))
        return found,filtered

def decode(ctc,province,chars,provinces):
    ctc=np.asarray(ctc);province=np.asarray(province)
    if len(chars)!=72 or len(provinces)!=31 or ctc.size!=56*73 or province.size!=32:
        raise ValueError('E3 logits/character contract mismatch')
    if not np.isfinite(ctc).all() or not np.isfinite(province).all():
        raise ValueError('nonfinite E3 logits')
    ids=ctc.reshape(56,73).argmax(axis=1);previous=-1;text=''
    for i in ids:
        if i and i!=previous:text+=chars[int(i)-1]
        previous=i
    k=int(np.asarray(province).argmax())
    return dict(ctc_text=text,text=provinces[k]+text[1:] if k<len(provinces) else '',province_id=k,
                province=provinces[k] if k<len(provinces) else '<other>',
                decode_contract=OCR_DECODE_CONTRACT,decode_status='decoded' if k<len(provinces) else 'unsupported',
                decode_reason='province_prefix_fusion' if k<len(provinces) else 'province_other')

def preprocess(image,mode):
    h,w=image.shape[:2];scale=min(224/w,64/h)
    rounding=round if mode=='pil' else lambda x:int(np.floor(x+.5))
    nw,nh=max(1,min(224,rounding(w*scale))),max(1,min(64,rounding(h*scale)))
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    resized=np.asarray(Image.fromarray(rgb).resize((nw,nh),Image.Resampling.BILINEAR)) if mode=='pil' else cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.zeros((64,224,3),np.uint8);canvas[(64-nh)//2:(64-nh)//2+nh,:nw]=resized
    # The saved board tensor is the authoritative exact input for backend parity.
    return np.ascontiguousarray((canvas.astype(np.float32)/np.float32(127.5)-np.float32(1)).transpose(2,0,1)[None])


class Engine:
    """HyperLPR geometry; YOLO hints and E3 diagnostics are explicit opt-ins."""
    def __init__(self, cfg=None):
        self.cfg = cfg
        models = (cfg or {}).get('models', {})
        self.e3_enabled = bool(models.get('e3_ocr', False))
        required = ['assets/models/hyperlpr3/provenance.json',
                    'assets/models/hyperlpr3/y5fu_640x_sim.onnx']
        if models.get('yolo_hints', False):
            required.append('assets/models/yolo/detector.onnx')
        asset_check(required=required, cfg=cfg)
        self.teacher = PlateTeacher(asset_path('assets/models/hyperlpr3', cfg=cfg))
        self.yolo = (Detector(asset_path('assets/models/yolo/detector.onnx', cfg=cfg), confidence=.22)
                     if models.get('yolo_hints', False) else None)
        self.e3 = None

    def e3_recognize(self, crop):
        require(self.e3_enabled, 'E3 OCR requires cfg.models.e3_ocr=true')
        if self.e3 is None:
            asset_check(required=['assets/models/e3/model.onnx', 'assets/models/e3/verification.json'], cfg=self.cfg)
            ort = _runtime()
            options = ort.SessionOptions(); options.intra_op_num_threads = 1
            meta = read_json(asset_path('assets/models/e3/verification.json', cfg=self.cfg))
            session = ort.InferenceSession(str(asset_path('assets/models/e3/model.onnx', cfg=self.cfg)), options,
                                          providers=['CPUExecutionProvider'])
            self.e3_meta = meta; self.e3 = session
        tensor = preprocess(crop, 'cv')
        values = self.e3.run(None, {self.e3.get_inputs()[0].name: tensor})
        return decode(*values, self.e3_meta['characters'], self.e3_meta['provinces'])

    def detect(self, image, ocr=False):
        h, w = image.shape[:2]
        tw, th = math.ceil(w / 1.8), math.ceil(h / 1.8)
        regions = [(0, 0, w, h)] + [(x, y, tw, th) for y in (0, h - th) for x in (0, w - tw)]
        hints, _ = self.yolo.run(image) if self.yolo is not None else ([], [])
        for row in hints:
            x, y, width, height = row['bbox']; pad = max(4, max(width, height) * .08)
            try:regions.append(tuple(cover_box([x - pad, y - pad, x + width + pad, y + height + pad], w, h)))
            except ValueError:continue
        proposals = []
        for x, y, width, height in dict.fromkeys(regions):
            for row in self.teacher.detect(image[y:y + height, x:x + width]):
                row['bbox'][0] += x; row['bbox'][1] += y
                row['corners'] = [[a + x, b + y] for a, b in row['corners']]
                proposals.append(row)
        for hint in hints:
            x, y, width, height = hint['bbox']
            box = cover_box([x, y, x + width, y + height], w, h)
            proposals.append(dict(bbox=box, corners=axis_corners(box), score=hint['score'],
                corners_source='axis_aligned_proposal_not_measured', teacher='YOLO26n', layer=0))
        output = []
        for row in merge_proposals(proposals):
            measured = row['corners_source'] == 'model_predicted'
            item = dict(quad=[row['corners'][i] for i in (2, 3, 0, 1)] if measured else None,
                        bbox=row['bbox'], confidence=row['score'],
                        quad_origin='model_predicted' if measured else 'box_only_not_measured',
                        teacher=row['teacher'], status='proposal_not_human_GT')
            if ocr and measured:
                from .geometry import rectify
                try:
                    face, _, _ = rectify(image, item['quad'], 'green')
                    item['ocr'] = dict(hyperlpr=self.teacher.recognize(face),
                                       semantics='diagnostic_only_not_ground_truth')
                    if self.e3_enabled:
                        item['ocr']['e3'] = self.e3_recognize(face)
                except (ValueError, cv2.error) as exc:item['ocr_error'] = str(exc)
            output.append(item)
        return output

    def contracts(self):
        """Describe loaded, enabled sessions without triggering optional inference."""
        sessions = [('hyper_detection', self.teacher.det)]
        if self.teacher.rec is not None:
            sessions.append(('hyper_ocr', self.teacher.rec))
        if self.yolo is not None:
            sessions.append(('yolo', self.yolo.session))
        if self.e3 is not None:
            sessions.append(('e3', self.e3))
        return {name: {'inputs': [dict(name=v.name, shape=v.shape) for v in session.get_inputs()],
                       'outputs': [dict(name=v.name, shape=v.shape) for v in session.get_outputs()]}
                for name, session in sessions}
