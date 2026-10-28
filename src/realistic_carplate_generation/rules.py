"""User-approved three-category GA36-format subset and balanced integer quotas."""
import random
from collections import Counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ._render_base import PROVINCES, LETTERS, SMALL_THIRD, DIGITS, PLATE_TYPES, valid_text
from .common import require

WEIGHTS = (0.4, 0.45, 0.15)


def apportion(count, weights=WEIGHTS):
    require(count >= 0 and all(w >= 0 for w in weights) and sum(weights) > 0, '配额参数错误')
    ideal = np.asarray(weights, float) / sum(weights) * count
    result = np.floor(ideal + 1e-9).astype(int)
    order = sorted(range(len(weights)), key=lambda i: (-(ideal[i] - result[i]), i))
    for i in order[:count - int(result.sum())]:
        result[i] += 1
    return result.tolist()


def quota_table(count, seed, weights=WEIGHTS):
    require(isinstance(count, int) and count > 0, '数量必须是正整数')
    rng = random.Random(seed)
    order = list(range(31)); rng.shuffle(order)
    province = np.full(31, count // 31, int)
    province[order[:count % 31]] += 1
    target = np.asarray(apportion(count, weights), int)
    ideal = province[:, None] * np.asarray(weights)
    floor = np.floor(ideal + 1e-9).astype(int)
    row_need, col_need = province - floor.sum(1), target - floor.sum(0)
    a = np.zeros((34, 93))
    for p in range(31):
        for t in range(3):
            a[p, p * 3 + t] = 1; a[31 + t, p * 3 + t] = 1
    b = np.concatenate((row_need, col_need))
    cost = -(ideal - floor).ravel() + np.array([rng.random() * 1e-6 for _ in range(93)])
    solution = milp(cost, integrality=np.ones(93), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(a, b, b), options={'time_limit': 10})
    require(solution.success, '省份×牌型整数配额求解失败')
    table = floor + np.rint(solution.x).astype(int).reshape(31, 3)
    require(np.array_equal(table.sum(0), target) and np.array_equal(table.sum(1), province), '整数配额错误')
    return table


def make_samples(count, seed=20261023, plate_type='all'):
    require(plate_type == 'all' or plate_type in PLATE_TYPES, '未知牌型')
    weights = WEIGHTS if plate_type == 'all' else tuple(int(k == plate_type) for k in PLATE_TYPES)
    table = quota_table(count, seed, weights)
    rng = random.Random(seed)
    city_offset = rng.randrange(24); third_offset = rng.randrange(10)
    city_position = third_position = 0
    rows, seen = [], set()
    for p, province in enumerate(PROVINCES):
        cities = [LETTERS[(city_offset + city_position + i) % 24] for i in range(int(table[p].sum()))]
        city_position += len(cities); rng.shuffle(cities)
        thirds = [SMALL_THIRD[(third_offset + third_position + i) % 10] for i in range(table[p, 1])]
        third_position += len(thirds); rng.shuffle(thirds)
        for t, kind in enumerate(PLATE_TYPES):
            for _ in range(table[p, t]):
                city = cities.pop(); third = thirds.pop() if kind == 'nev_small' else None
                for attempt in range(10000):
                    if kind == 'blue':
                        tail = list(rng.choices(DIGITS, k=5))
                        for i in rng.sample(range(5), rng.randrange(3)):tail[i] = rng.choice(LETTERS)
                        tail = ''.join(tail)
                    elif kind == 'nev_small':
                        tail = third + rng.choice(LETTERS + DIGITS) + ''.join(rng.choices(DIGITS, k=4))
                    else:
                        tail = ''.join(rng.choices(DIGITS, k=5)) + rng.choice('DF')
                    text = province + city + tail
                    if text not in seen:break
                else:raise ValueError('号码空间不足，无法满足唯一性')
                require(valid_text(text, kind), '生成非法号码')
                seen.add(text); rows.append(dict(text=text, plate_type=kind))
    rng.shuffle(rows)
    return rows


def summarize(samples) -> dict:
    counts = dict(total=len(samples), province=Counter(), plate_type=Counter(), city_letter=Counter(),
                  nev_small_third=Counter(), province_plate_type={p: Counter() for p in PROVINCES})
    seen = set()
    for row in samples:
        text, kind = row['text'], row['plate_type']
        require(valid_text(text, kind) and text not in seen, '非法或重复号码')
        seen.add(text)
        for field, key in [('province', text[0]), ('plate_type', kind), ('city_letter', text[1])]:counts[field][key] += 1
        counts['province_plate_type'][text[0]][kind] += 1
        if kind == 'nev_small':counts['nev_small_third'][text[2]] += 1
    for name, keys in [('province', PROVINCES), ('city_letter', LETTERS), ('nev_small_third', SMALL_THIRD)]:
        values = [counts[name][key] for key in keys]
        require(max(values) - min(values) <= 1, f'{name} 不均衡')
        counts[name] = {key: counts[name][key] for key in keys}
    counts['plate_type'] = {key: counts['plate_type'][key] for key in PLATE_TYPES}
    counts['province_plate_type'] = {p: dict(counts['province_plate_type'][p]) for p in PROVINCES}
    counts['invalid_text'] = counts['duplicate_text'] = 0
    return counts
