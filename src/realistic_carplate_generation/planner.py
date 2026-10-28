"""Plan image bundles with integer quotas before writing any generated image."""
import math
import random
import heapq
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

from .common import require, dumps
from .rules import PLATE_TYPES, WEIGHTS, apportion, make_samples, summarize


def selected_plates(row, targets, target_id=None):
    plates = row['plates']
    if target_id is not None:
        plates = [p for p in plates if p['id'] == str(target_id)]
        require(bool(plates), f'目标不存在: {row["id"]}:{target_id}')
    if targets == 'single':
        return [max(plates, key=lambda p:(p['native_height'], p.get('confidence', 1), p['id']))]
    return plates


def build_plan(index, count, cfg, targets='single', target_id=None):
    require(isinstance(count, int) and count > 0, 'count 必须为正整数图片数')
    records = index['records']
    require(len({r['id'] for r in records}) == len(records) and
            len({r['sha256'] for r in records}) == len(records), '索引含重复来源身份；请先 prepare 去重')
    kind = cfg.get('plate_type', 'all')
    require(kind == 'all' or kind in PLATE_TYPES, '未知牌型')
    weights = WEIGHTS if kind == 'all' else tuple(int(k == kind) for k in PLATE_TYPES)
    seed = cfg['seed']; rng = random.Random(seed)
    groups = defaultdict(list); rows = {}
    for record in sorted(index['records'], key=lambda r:r['id']):
        candidates = record['plates'] if kind == 'all' else [
            p for p in record['plates'] if p['colour'] == ('blue' if kind == 'blue' else 'green')]
        if not candidates:continue
        plates = selected_plates(dict(record, plates=candidates), targets, target_id)
        blue = sum(p['colour'] == 'blue' for p in plates); green = len(plates) - blue
        cap = cfg['reuse_blue'] if blue else cfg['reuse_green']
        cap = count if cap == 'auto' else int(cap)
        require(cap > 0, '底图复用次数必须为正整数或 auto')
        rows[record['id']] = dict(record=record, plates=plates, cap=cap)
        groups[blue, green].append(record['id'])
    require(bool(groups), '无可用目标')
    keys = sorted(groups); capacities = [sum(rows[i]['cap'] for i in groups[k]) for k in keys]
    total_capacity = sum(capacities)
    require(total_capacity >= count, f'底图容量不足: 需 {count} 张图片，可用上限 {total_capacity}')
    found = None; n = len(keys)
    minimum_p = min(b + g for b, g in keys) * count
    maximum_p = max(b + g for b, g in keys) * count
    # Fractions have denominator 20: P=20*q+r, blue=8*q+apportion(r)[0].
    for remainder in range(20):
        if minimum_p == maximum_p and minimum_p % 20 != remainder:continue
        a = np.zeros((3, n + 1)); a[0, :n] = 1
        a[1, :n] = [b + g for b, g in keys]; a[1, n] = -20
        a[2, :n] = [b for b, g in keys]; a[2, n] = -round(20 * weights[0])
        rhs = np.array([count, remainder, apportion(remainder, weights)[0]], float)
        cost = np.array([b + g + 1e-7 * (i + 1) for i, (b, g) in enumerate(keys)] + [0.0])
        sol = milp(cost, integrality=np.ones(n + 1),
                   bounds=Bounds(np.zeros(n + 1), np.array(capacities + [maximum_p // 20 + 1], float)),
                   constraints=LinearConstraint(a, rhs, rhs), options={'time_limit': cfg.get('plan_time_limit', 30)})
        if sol.success:
            candidate = np.rint(sol.x[:n]).astype(int)
            p = sum((b + g) * int(candidate[i]) for i, (b, g) in enumerate(keys))
            if found is None or (p, tuple(candidate)) < (found[0], tuple(found[1])):found = p, candidate
    quota = '40/45/15 配额' if kind == 'all' else f'{kind} 牌型'
    require(found is not None, f'未找到满足 {quota}和复用限制的计划；蓝/绿组合容量={dict(zip(map(str,keys),capacities))}')
    plate_count, allocation = found
    chosen, uses = [], Counter()
    for key, amount in zip(keys, allocation):
        deck = groups[key].copy(); rng.shuffle(deck)
        heap = [(0, tie, ident) for tie, ident in enumerate(deck)]
        heapq.heapify(heap)
        for _ in range(int(amount)):
            require(bool(heap), '分组底图容量耗尽')
            used, tie, ident = heapq.heappop(heap)
            uses[ident] += 1; chosen.append(ident)
            if uses[ident] < rows[ident]['cap']:heapq.heappush(heap, (used + 1, tie, ident))
    rng.shuffle(chosen)
    samples = make_samples(plate_count, seed, plate_type=kind)
    by_kind = {kind: [s for s in samples if s['plate_type'] == kind] for kind in PLATE_TYPES}
    green_kinds = ['nev_small'] * len(by_kind['nev_small']) + ['nev_large'] * len(by_kind['nev_large'])
    rng.shuffle(green_kinds)
    jobs = []
    width = max(6, len(str(count)))
    plate_index = 0
    for image_index, ident in enumerate(chosen, 1):
        targets_out = []
        for j, plate in enumerate(rows[ident]['plates']):
            plate_index += 1
            kind = 'blue' if plate['colour'] == 'blue' else green_kinds.pop()
            sample = by_kind[kind].pop()
            targets_out.append(dict(**sample, quad=plate['quad'], target_id=plate['id'],
                                    colour=plate['colour'], plate_index=plate_index,
                                    seed=seed + image_index * 1000003 + j * 7919))
        jobs.append(dict(image=f'images/{image_index:0{width}d}.{cfg["format"]}', source_id=ident, plates=targets_out))
    require(sum(len(j['plates']) for j in jobs) == plate_count, '计划目标数错误')
    stats = summarize(samples)
    reuse = {str(k): dict(Counter(uses[i] for i in groups[k])) for k in keys}
    return dict(schema_version=1, image_count=count, plate_count=plate_count, targets=targets,
                render_mode=cfg['render_mode'], seed=seed, jobs=jobs, target_counts=stats,
                background_reuse_histograms=reuse,
                quota_rounding={k: stats['plate_type'][k] - plate_count * w for k, w in zip(PLATE_TYPES, weights)})
