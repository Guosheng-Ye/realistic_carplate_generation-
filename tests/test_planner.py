import copy
import unittest
from collections import Counter

from realistic_carplate_generation.common import load_config
from realistic_carplate_generation.planner import build_plan


def record(ident, colours):
    return dict(id=ident,sha256=ident,root='test',path=ident+'.jpg',width=1000,height=500,
                plates=[dict(id=str(i),colour=c,native_height=60,confidence=1,
                             quad=[[450,170],[10,170],[10,30],[450,30]]) for i,c in enumerate(colours)])


class PlannerTests(unittest.TestCase):
    def test_multi_count_is_images_and_quotas_count_plates(self):
        index=dict(records=[record('two_blue',['blue','blue']),record('green',['green'])],roots={'test':'/tmp'},heldout_sha256=[])
        cfg=load_config();cfg['reuse_blue']='auto'
        plan=build_plan(index,4,cfg,'all')
        self.assertEqual(plan['image_count'],4);self.assertEqual(plan['plate_count'],5)
        self.assertEqual(plan['target_counts']['plate_type'],{'blue':2,'nev_small':2,'nev_large':1})
        self.assertEqual(sorted(len(j['plates']) for j in plan['jobs']),[1,1,1,2])
        self.assertEqual(plan,build_plan(index,4,cfg,'all'))

    def test_infeasible_composition_and_capacity_stop_before_rendering(self):
        cfg=load_config()
        only_blue=dict(records=[record('blue',['blue'])],roots={},heldout_sha256=[])
        with self.assertRaisesRegex(ValueError,'未找到'):build_plan(only_blue,2,cfg,'all')
        both=dict(records=[record('b',['blue']),record('g',['green'])],roots={},heldout_sha256=[])
        with self.assertRaisesRegex(ValueError,'未找到'):build_plan(both,20,cfg)
        cfg['reuse_green']=1
        with self.assertRaisesRegex(ValueError,'容量不足'):build_plan(both,20,cfg)

    def test_single_selects_largest_and_target_override(self):
        r=record('multi',['blue','green']);r['plates'][1]['native_height']=80
        cfg=load_config();cfg['reuse_blue']='auto'
        index=dict(records=[r,record('other_blue',['blue'])],roots={},heldout_sha256=[])
        p=build_plan(index,20,cfg,'single')
        target=[t for j in p['jobs'] if j['source_id']=='multi' for t in j['plates']]
        self.assertTrue(all(t['target_id']=='1' and t['colour']=='green' for t in target))

    def test_duplicate_source_identity_cannot_expand_capacity(self):
        row=record('same',['blue'])
        index=dict(records=[row,copy.deepcopy(row)],roots={},heldout_sha256=[])
        with self.assertRaisesRegex(ValueError,'重复来源'):build_plan(index,1,load_config())
