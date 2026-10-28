"""Public Python interface. All arrays are OpenCV BGR, all quads BR/BL/TL/TR."""
from .common import load_config as configuration
from .sources import prepare as _prepare
from .planner import build_plan as _plan
from .pipeline import generate as _generate
from .geometry import rectify
from .rendering import render_plate as _render
from .validation import validate
from .reporting import report


def prepare(input_path, output=None, cfg=None, source_type='auto', ocr=False):
    return _prepare(input_path, output, cfg if cfg is not None else configuration(), source_type, ocr)


def plan(index, count, cfg=None, targets='single', target_id=None):
    return _plan(index, count, cfg if cfg is not None else configuration(), targets, target_id)


def generate(index, job_plan, output, cfg=None, workers=4, resume=False):
    return _generate(index, job_plan, output, cfg if cfg is not None else configuration(), workers, resume)


def detect(image, ocr=False, engine=None, cfg=None):
    from .inference import Engine
    return (engine if engine is not None else Engine(cfg)).detect(image, ocr)


def render(text, plate_type, seed=20261023, cfg=None):
    return _render(text, plate_type, seed, cfg=cfg)
