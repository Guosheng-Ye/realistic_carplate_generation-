"""Plate lettering, material-aware lighting and local augmentation primitives."""
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import require

PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

SMALL_THIRD = "ABCDEFGHJK"

DIGITS = "0123456789"

PLATE_TYPES = ("blue", "nev_small", "nev_large")

PATTERN = re.compile(
    r"^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-HJ-NP-Z]([A-HJ-NP-Z0-9]{5}|[A-HJ-K][A-HJ-NP-Z0-9][0-9]{4}|[0-9]{5}[DF])$"
)

TYPE_PATTERNS = {
    "blue": re.compile(r"[A-HJ-NP-Z0-9]{5}"),
    "nev_small": re.compile(r"[A-HJ-K][A-HJ-NP-Z0-9][0-9]{4}"),
    "nev_large": re.compile(r"[0-9]{5}[DF]"),
}

@dataclass(frozen=True)
class Sample:
    text: str
    plate_type: str

@dataclass(frozen=True)
class FontSpec:
    path: str
    index: int = 0
    dedicated: bool = False

def valid_text(text, plate_type):
    if plate_type not in TYPE_PATTERNS or PATTERN.fullmatch(text) is None:
        return False
    if TYPE_PATTERNS[plate_type].fullmatch(text[2:]) is None:
        return False
    # The supplied common regex alone does NOT enforce the blue-plate letter cap.
    return plate_type != "blue" or sum(c in LETTERS for c in text[2:]) <= 2

def validate_font(spec, characters):
    font = ImageFont.truetype(spec.path, 120, index=spec.index)
    missing = font.getmask("\uffff")
    missing_signature = (missing.size, bytes(missing))
    for char in characters:
        mask = font.getmask(char)
        require(mask.getbbox() is not None and (mask.size, bytes(mask)) != missing_signature,
                f"字体 {spec.path} 不含字符 {char}")

class PlateRenderer:
    SCALE = 2

    def __init__(self, han, latin):
        self.fonts = {True: ImageFont.truetype(han.path, 240, index=han.index),
                      False: ImageFont.truetype(latin.path, 240, index=latin.index)}
        boxes = [self.fonts[False].getmask(char).getbbox() for char in LETTERS + DIGITS]
        self.latin_ink_width = max(box[2] - box[0] for box in boxes)

    @lru_cache(maxsize=256)
    def glyph(self, char, width, height):
        font = self.fonts[char in PROVINCES]
        x0, y0, x1, y1 = font.getbbox(char)
        image = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8))
        ImageDraw.Draw(image).text((4 - x0, 4 - y0), char, font=font, fill=255)
        image = image.crop(image.getbbox())
        if char in PROVINCES:
            return image.resize((width, height), Image.Resampling.LANCZOS)
        # One horizontal scale for the whole alphabet preserves narrow strokes,
        # particularly GA36 '1'; stretching every ink box to the slot destroys it.
        ink_width = max(1, min(width, round(width * image.width / self.latin_ink_width)))
        image = image.resize((ink_width, height), Image.Resampling.LANCZOS)
        slot = Image.new("L", (width, height))
        slot.paste(image, ((width - ink_width) // 2, 0))
        return slot

    def render(self, sample, rng):
        require(valid_text(sample.text, sample.plate_type), "禁止渲染非法号码")
        scale = self.SCALE
        width, height = (440 if sample.plate_type == "blue" else 480), 140
        if sample.plate_type == "blue":
            rgb = np.full((height * scale, width * scale, 3), (12, 65, 164), np.uint8)
            ink, border = (248, 248, 244), (235, 239, 240)
            xs, char_width = (18, 76, 157, 210, 263, 316, 369), 45
            dot_x = 140
        else:
            top, bottom = np.array((238, 248, 239)), np.array((36, 178, 83))
            ramp = np.linspace(0, 1, height * scale)[:, None, None]
            rgb = np.broadcast_to(top * (1 - ramp) + bottom * ramp,
                                  (height * scale, width * scale, 3)).copy().astype(np.uint8)
            if sample.plate_type == "nev_large":
                rgb[:] = (65, 185, 91)
                rgb[:, :144 * scale] = (246, 205, 57)
            ink, border = (14, 22, 18), (27, 74, 39)
            xs, char_width, dot_x = (20, 76, 151, 204, 257, 310, 363, 416), 44, 135
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((4 * scale, 4 * scale, (width - 5) * scale, 135 * scale),
                               radius=9 * scale, outline=border, width=2 * scale)
        for x, char in zip(xs, sample.text):
            image.paste(ink, (x * scale, 25 * scale), self.glyph(char, char_width * scale, 90 * scale))
        draw.ellipse(((dot_x - 3) * scale, 68 * scale, (dot_x + 3) * scale, 74 * scale), fill=ink)
        rivet_x = (75, width - 75)
        for y in ((12, 128) if rng.random() < 0.5 else (12,)):
            for x in rivet_x:
                draw.ellipse(((x - 3) * scale, (y - 3) * scale, (x + 3) * scale, (y + 3) * scale),
                             fill=(158, 164, 159), outline=(95, 102, 100), width=scale)
                draw.line(((x - 1) * scale, y * scale, (x + 1) * scale, y * scale),
                          fill=(220, 222, 218), width=scale)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def match_lighting(plate, original, mask, plate_height):
    inside = mask > 0.98
    require(int(inside.sum()) >= 20, "有效牌面过小")
    source_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    plate_lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_mean, plate_mean = source_lab[inside].mean(0), plate_lab[inside].mean(0)
    gain = float(np.clip(source_mean[0] / max(plate_mean[0], 1), 0.68, 1.22))
    # Only low-frequency luminance survives: original characters must not ghost through.
    sigma = max(3.0, plate_height * 0.45)
    smooth = cv2.GaussianBlur(source_lab[:, :, 0], (0, 0), sigma)
    shadow = np.clip((smooth - source_mean[0]) / max(source_mean[0], 25), -0.35, 0.35)
    plate_lab[:, :, 0] *= gain * (1 + shadow * 0.18)
    # Limit chroma adaptation to keep blue/green/yellow plate categories intact.
    plate_lab[:, :, 1:] += np.clip((source_mean[1:] - plate_mean[1:]) * 0.12, -2, 2)
    return cv2.cvtColor(np.clip(plate_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def augment_region(region, alpha, rng, min_height):
    """Choose 0-3 distinct augmentation families, on the already pasted plate."""
    result, jpeg_quality, operations = region.astype(np.float32), 95, []
    noise_rng = np.random.default_rng(rng.getrandbits(64))
    height, width = region.shape[:2]
    for family in rng.sample(("blur", "exposure", "shadow", "noise", "jpeg", "weather"), rng.randrange(4)):
        changed = result.copy()
        if family == "blur":
            if rng.random() < 0.5:
                max_kernel = max(3, min(9, int(min_height / 10) | 1))
                size = rng.choice(list(range(3, max_kernel + 1, 2)))
                kernel = np.zeros((size, size), np.float32)
                angle = rng.uniform(0, math.pi)
                delta = (size - 1) / 2 * np.array((math.cos(angle), math.sin(angle)))
                center = np.full(2, (size - 1) / 2)
                cv2.line(kernel, tuple(np.rint(center - delta).astype(int)),
                         tuple(np.rint(center + delta).astype(int)), 1, 1)
                kernel /= kernel.sum()
                changed = cv2.filter2D(result, -1, kernel)
            else:
                sigma = rng.uniform(0.3, max(0.3, min(1.5, min_height / 40)))
                changed = cv2.GaussianBlur(result, (0, 0), sigma)
        elif family == "exposure":
            mode = rng.choice(("brightness", "contrast", "gamma"))
            if mode == "brightness":
                changed = result * rng.uniform(0.6, 1.3)
            elif mode == "contrast":
                changed = (result - 127.5) * rng.uniform(0.85, 1.15) + 127.5
            else:
                changed = 255 * (np.clip(result, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
        elif family == "shadow":
            field = np.zeros((height, width), np.float32)
            vertices = np.array([(rng.randrange(width), rng.randrange(height)) for _ in range(4)], np.int32)
            cv2.fillConvexPoly(field, cv2.convexHull(vertices), 1)
            field = cv2.GaussianBlur(field, (0, 0), max(1, min_height * 0.08))
            changed *= 1 + field[:, :, None] * rng.uniform(-0.3, 0.3)
        elif family == "noise":
            # Achromatic noise avoids spurious plate hue changes.
            changed += noise_rng.normal(0, rng.uniform(2, 8), (height, width, 1)).astype(np.float32)
        elif family == "jpeg":
            jpeg_quality = rng.randint(40, 90)
        else:
            hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * rng.uniform(0.7, 0.9)).astype(np.uint8)
            changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
            # Very faint dirt; never opaque occluders over the glyphs.
            dirt = np.zeros((height, width), np.float32)
            for _ in range(rng.randint(1, 3)):
                cv2.circle(dirt, (rng.randrange(width), rng.randrange(height)),
                           max(1, int(min_height * rng.uniform(0.02, 0.05))), rng.uniform(0.03, 0.08), -1)
            changed *= 1 - dirt[:, :, None]
        result = np.clip(changed, 0, 255) * alpha[:, :, None] + result * (1 - alpha[:, :, None])
        operations.append(family)
    return np.rint(result).astype(np.uint8), jpeg_quality, operations
