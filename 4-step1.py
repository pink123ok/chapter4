# -*- coding: utf-8 -*-
"""
第四章新增实验 · 模块一：时间划分、真实来源校验与未知OD盲测数据隔离
版本：1.0 (2026-10-29)

使用第三章、第四章已经正式生成的真实数据；不产生模拟数据。
输入：
  1) Step41: 模块一_真实路径与OD/01_第四章有效OD母表.csv
  2) Step6: 第三章_路径经济舒适语义联合特征/01_第三章_路径经济舒适语义联合特征.csv
前提：先运行原 step41(1).py 建立并核验完整OD母表；本脚本不重新做HMM地图匹配。

严格规则：
  - 保持第三章经济性6项、舒适性6项不变；本模块不训练聚类或Logit模型。
  - 当地时间2024年12月1~14日：UDPPM++原型校准期。
  - 12月15~23日：路径选择模型训练期。
  - 12月24~28日：内部验证期。
  - 12月29~31日：最后3天完全独立测试期。
  - 跨越上述边界的行程直接隔离，不能作为任一阶段的完成历史行程。
  - "测试纯OD输入" 不含真实路径、路线长度、真实车次时长和任何标签。
  - "测试真值" 另存为严禁推理读取的文件，事后评估时才可使用。
  - ID保持字符串，禁止猜测缺失字段、自动修正错误时间或虚构记录。

安装：python -m pip install -U pandas matplotlib tzdata
Windows调用：
  python "第四章_模块一_28天训练3天未知测试_数据隔离.py"
  python "第四章_模块一_28天训练3天未知测试_数据隔离.py" --help
  python "第四章_模块一_28天训练3天未知测试_数据隔离.py" --overwrite

重要：默认将不带时区的源时间解释为曼谷当地时间 Asia/Bangkok。
      如果Step3/Step6时间戳实际上已保存为UTC但没有时区，请显式修改--naive-tz UTC。
      源时间口径需要根据真实预处理脚本核对，不能自行猜测。
"""
from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as ex:
    raise SystemExit('缺少依赖：python -m pip install -U pandas matplotlib tzdata\n' + str(ex))

BASE = Path(r'C:\study\轨迹数据预处理')
ROOT = BASE / '曼谷GPS_正式清洗_202412' / 'FULL_全量'
MASTER_DEFAULT = (ROOT / '第四章_无泄漏路径选择实验'
                  / '模块一_真实路径与OD' / '01_第四章有效OD母表.csv')
STEP6_DEFAULT = (ROOT / '第三章_路径经济舒适语义联合特征'
                 / '01_第三章_路径经济舒适语义联合特征.csv')
OUT_DEFAULT = ROOT / '第四章_28天训练_3天独立测试' / '模块一_时间划分与隔离'

# 严格照抄第三章表3-5的 6+6 特征，不新增/删减。
ECON6 = ['path_len_km', 'travel_time_min', 'avg_speed_kmh_geom',
         'ratio_motorway', 'ratio_major', 'ratio_local']
COMF6 = ['slow_ratio', 'turns_per_km', 'ratio_local',
         'ratio_motorway', 'oneway_ratio', 'segment_density']
BEHAVIOR_COLS = list(dict.fromkeys(ECON6 + COMF6))  # 共享列只写一次；两组各6项不变。
MASTER_REQUIRED = ['trip_id', 'VehicleID', 'start_time', 'end_time',
                   'origin_node', 'destination_node', 'od_straight_km',
                   'edge_seq_uvk_json', 'path_len_km', 'duration_min']
STEP6_REQUIRED = ['trip_id', 'VehicleID', 'start_time', 'end_time'] + BEHAVIOR_COLS
# 推理输入白名单：不含VehicleID、真实行驶记录、真实标签。
OD_ONLY_COLS = ['trip_id', 'start_time', 'origin_node', 'destination_node',
                'od_straight_km', 'origin_lon', 'origin_lat',
                'destination_lon', 'destination_lat']
AUDIT_COLS = ['trip_id', 'VehicleID', 'start_time', 'end_time', 'status', 'reason']

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def log(s: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {s}', flush=True)


def get_headers(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        h = next(csv.reader(f), [])
    if not h:
        raise ValueError(f'CSV表头为空：{path}')
    if len(h) != len(set(h)):
        raise ValueError(f'CSV存在重复字段名：{path}')
    return h


def require_headers(path: Path, needed: list[str]) -> list[str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'缺少真实实验文件：{path}\n先运行原始Step41/Step6，不会使用模拟数据。')
    h = get_headers(path)
    missing = sorted(set(needed) - set(h))
    if missing:
        raise ValueError(f'{path.name}缺少字段：{missing}\n实际列名：{h}')
    return h


def rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def timestamp(raw: str, naive_zone: ZoneInfo, local_zone: ZoneInfo) -> pd.Timestamp:
    s = str(raw or '').strip()
    if not s:
        raise ValueError('空时间')
    try:
        t = pd.Timestamp(s)
        if pd.isna(t):
            raise ValueError('NaT')
        if t.tzinfo is None:
            t = t.tz_localize(naive_zone, ambiguous='raise', nonexistent='raise')
        return t.tz_convert(local_zone)
    except Exception as ex:
        raise ValueError(f'非法时间戳 {s!r}: {ex}') from ex


def stage_for_day(d: date) -> str:
    if date(2024, 12, 1) <= d <= date(2024, 12, 14):
        return 'calibration'
    if date(2024, 12, 15) <= d <= date(2024, 12, 23):
        return 'model_train'
    if date(2024, 12, 24) <= d <= date(2024, 12, 28):
        return 'validation'
    if date(2024, 12, 29) <= d <= date(2024, 12, 31):
        return 'future_test'
    return 'outside_month'


def stage_of_trip(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, str]:
    if end <= start:
        return 'excluded', 'end_not_after_start'
    s = stage_for_day(start.date())
    # 完成时刻若恰好等于第二阶段开始，也视为新阶段完成，避免前序阶段偷看。
    e = stage_for_day(end.date())
    if s == 'outside_month' or e == 'outside_month':
        return 'excluded', 'outside_december_2024'
    if s != e:
        return 'excluded', f'cross_stage_boundary_{s}_to_{e}'
    return s, ''


def safe_float(raw: str, *, name: str, trip: str) -> float:
    try:
        x = float(raw)
    except (ValueError, TypeError) as ex:
        raise ValueError(f'{trip} 字段{name}不是有效数字：{raw!r}') from ex
    if not math.isfinite(x):
        raise ValueError(f'{trip} 字段{name}不是有限数字：{raw!r}')
    return x


def file_info(path: Path, sha256: bool) -> dict:
    s = path.stat()
    info = {'path': str(path.resolve()), 'bytes': s.st_size,
            'mtime_local': datetime.fromtimestamp(s.st_mtime).isoformat(timespec='seconds')}
    if sha256:
        dig = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                dig.update(chunk)
        info['sha256'] = dig.hexdigest()
    return info


def dumps_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8-sig')


def utf_csv(path: Path, headers: list[str]):
    """返回打开的 CSV 文件和 writer；外部显式关闭，适于大型路径JSON流式写出。"""
    if path.suffix == '.gz':
        f = gzip.open(path, 'wt', newline='', encoding='utf-8-sig', compresslevel=6)
    else:
        f = path.open('w', newline='', encoding='utf-8-sig')
    w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
    w.writeheader()
    return f, w


def draw_daily_plot(daily: Counter, out: Path) -> None:
    try:
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        days = [date(2024, 12, x) for x in range(1, 32)]
        counts = [daily.get(str(d), 0) for d in days]
        colors = [('#607d98' if d.day <= 14 else
                   '#3393a4' if d.day <= 23 else
                   '#d2a05a' if d.day <= 28 else '#b65c60') for d in days]
        fig, ax = plt.subplots(figsize=(13.5, 4.5))
        ax.bar(range(1, 32), counts, color=colors, width=.76)
        ax.axvline(14.5, color='#607d98', linestyle='--', linewidth=1)
        ax.axvline(23.5, color='#3393a4', linestyle='--', linewidth=1)
        ax.axvline(28.5, color='#b65c60', linestyle='--', linewidth=1.5)
        ax.set_xticks(range(1, 32, 2))
        ax.set_xlabel('2024年12月日期')
        ax.set_ylabel('有效OD数量')
        ax.set_title('第四章数据阶段划分与每日真实OD规模')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(out / '图4-X_训练验证测试时间划分.png', dpi=450, bbox_inches='tight')
        fig.savefig(out / '图4-X_训练验证测试时间划分.svg', bbox_inches='tight')
        plt.close(fig)
    except Exception as ex:
        # 图片生成不是数据拆分必须条件，但明确通知，不允许静默跳过。
        log(f'警告：真实统计图未成功生成：{ex}')


def main() -> None:
    p = argparse.ArgumentParser(description='第四章模块一：前28天模型开发、后3天未知OD盲测的数据来源审计与严格拆分')
    p.add_argument('--master', type=Path, default=MASTER_DEFAULT,
                   help='原Step41输出的有效OD母表（含原始HMM路径）')
    p.add_argument('--step6', type=Path, default=STEP6_DEFAULT,
                   help='第三章真实6+6特征文件（原始Step6）')
    p.add_argument('--out', type=Path, default=OUT_DEFAULT)
    p.add_argument('--naive-tz', default='Asia/Bangkok',
                   help='源时间不带时区时的解释口径；默认Asia/Bangkok，如源时间实际是UTC请改UTC')
    p.add_argument('--local-tz', default='Asia/Bangkok', help='划分日期的当地时区，默认曼谷')
    p.add_argument('--overwrite', action='store_true', help='允许覆盖当前模块一的输出文件')
    p.add_argument('--sha256', action='store_true', help='额外计算两个源文件SHA256以提高溯源严格程度')
    args = p.parse_args()
    try:
        naive_zone = ZoneInfo(args.naive_tz)
        local_zone = ZoneInfo(args.local_tz)
    except ZoneInfoNotFoundError as ex:
        raise SystemExit('找不到时区数据库，请安装tzdata：python -m pip install -U tzdata') from ex

    log('1/6 核对原Step41与第三章Step6真实文件及字段')
    master_cols = require_headers(args.master, MASTER_REQUIRED)
    require_headers(args.step6, STEP6_REQUIRED)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    names = {
      'index': '01_全量OD时间分组索引.csv',
      'counts': '02_四阶段样本规模统计.csv',
      'daily': '03_逐日有效OD分布.csv',
      'dev_od': '04_前28天开发OD与真实路径_仅训练.csv.gz',
      'blind_od': '05_最后3天_纯OD推理输入_绝无真实路线.csv',
      'profile_index': '06_最后3天_历史画像匹配索引_受控.csv',
      'truth': '07_最后3天_真实路径答案_仅评价禁止推理.csv.gz',
      'excluded': '08_跨边界与排除记录.csv',
      'features': '09_前28天_第三章原始6加6特征.csv.gz',
      'cold': '10_最后3天_新旧车辆及OD类型统计.csv',
      'manifest': '11_完整性防泄漏审计.json',
      'readme': 'README_实验数据使用边界.txt',
    }
    for name in list(names.values()) + ['图4-X_训练验证测试时间划分.png', '图4-X_训练验证测试时间划分.svg']:
        target = out / name
        if target.exists() and not args.overwrite:
            raise FileExistsError(f'输出已存在：{target}\n若确定重做，请加 --overwrite；不会删除原Step41或第三章文件。')

    # STEP6真实源表中的每个trip只保留必要字段，不复制全部未用列。
    log('2/6 读取第三章Step6并检查原始6+6特征结构、ID与时间')
    third = {}
    for r in rows(args.step6):
        tid = str(r.get('trip_id', '')).strip()
        if not tid or tid in third:
            raise ValueError(f'第三章Step6存在空/重复trip_id：{tid!r}')
        if not str(r.get('VehicleID', '')).strip():
            raise ValueError(f'第三章Step6存在空VehicleID：trip_id={tid}')
        # 校验6+6字段存在，不自行生成数值、不强制要求每个原值均非空；质量检查留给模块二。
        third[tid] = {c: r[c] for c in STEP6_REQUIRED}
    if not third:
        raise ValueError('第三章Step6表没有任何真实行程记录')
    log(f'第三章Step6总记录：{len(third):,}条')

    log('3/6 读取Step41 OD母表并严格核验来源映射与划分时间')
    all_items = []
    seen_ids = set()
    excluded = []
    reasons = Counter()
    stage_counts = Counter()
    per_stage_vehicles = defaultdict(set)
    daily_counts = Counter()
    for r in rows(args.master):
        tid = str(r.get('trip_id', '')).strip()
        if not tid or tid in seen_ids:
            raise ValueError(f'Step41存在空/重复trip_id：{tid!r}')
        seen_ids.add(tid)
        if tid not in third:
            raise ValueError(f'Step41的trip_id未见于第三章Step6：{tid}')
        srow = third[tid]
        vid = str(r['VehicleID']).strip()
        if vid != str(srow['VehicleID']).strip() or not vid:
            raise ValueError(f'Step41与Step6的VehicleID不一致：{tid}')
        try:
            start = timestamp(r['start_time'], naive_zone, local_zone)
            end = timestamp(r['end_time'], naive_zone, local_zone)
            src_start = timestamp(srow['start_time'], naive_zone, local_zone)
            src_end = timestamp(srow['end_time'], naive_zone, local_zone)
        except ValueError as ex:
            raise ValueError(f'时间错误 trip_id={tid}：{ex}') from ex
        if start != src_start or end != src_end:
            raise ValueError(f'Step41与Step6的时间戳不一致：trip_id={tid}；请核对来源或时区')
        if not str(r['origin_node']).strip() or not str(r['destination_node']).strip():
            raise ValueError(f'OD节点缺失 trip_id={tid}')
        if str(r['origin_node']) == str(r['destination_node']):
            raise ValueError(f'OD起终节点一致 trip_id={tid}；请检查Step41输出')
        od_km = safe_float(r['od_straight_km'], name='od_straight_km', trip=tid)
        if od_km <= 0:
            raise ValueError(f'OD直线距离非正 trip_id={tid}')
        if not str(r['edge_seq_uvk_json']).strip():
            raise ValueError(f'原始HMM边序列为空 trip_id={tid}')
        split, reason = stage_of_trip(start, end)
        item = {'record': r, 'step6': srow, 'trip_id': tid, 'vehicle': vid,
                'split': split, 'reason': reason, 'start': start, 'end': end}
        all_items.append(item)
        if split == 'excluded':
            reasons[reason] += 1
            excluded.append({'trip_id': tid, 'VehicleID': vid,
                             'start_time': r['start_time'], 'end_time': r['end_time'],
                             'status': 'excluded', 'reason': reason})
        else:
            stage_counts[split] += 1
            per_stage_vehicles[split].add(vid)
            daily_counts[str(start.date())] += 1
    if not all_items:
        raise ValueError('OD母表为空')
    if any(stage_counts[s] == 0 for s in ('calibration', 'model_train', 'validation', 'future_test')):
        raise ValueError('至少有一个必须阶段没有可用OD：' + str(dict(stage_counts)) + '\n请先核查真实时间范围，不能虚构缺失数据。')
    dev = [x for x in all_items if x['split'] in ('calibration', 'model_train', 'validation')]
    test = [x for x in all_items if x['split'] == 'future_test']
    train_ids = {x['trip_id'] for x in dev}
    test_ids = {x['trip_id'] for x in test}
    if train_ids & test_ids:
        raise AssertionError('致命错误：开发数据与测试数据有交叉trip_id')
    if max(x['end'] for x in dev) >= min(x['start'] for x in test):
        raise AssertionError('致命错误：开发期完成时间未严格早于测试期开始')
    if len(seen_ids) != len(all_items) or len(all_items) > len(third):
        raise AssertionError('OD记录数量或第三章来源映射不一致')
    log(f'OD母表总记录={len(all_items):,}; 前28天={len(dev):,}; 最后3天={len(test):,}; 跨边界或域外={len(excluded):,}')

    log('4/6 计算历史车辆、历史OD交叉标识（仅使用前28天作历史）')
    history_vehicles = {x['vehicle'] for x in dev}
    history_ods = {(str(x['record']['origin_node']), str(x['record']['destination_node'])) for x in dev}
    cold_counter = Counter()
    test_index = {}
    for x in test:
        old_vehicle = int(x['vehicle'] in history_vehicles)
        old_od = int((str(x['record']['origin_node']), str(x['record']['destination_node'])) in history_ods)
        test_index[x['trip_id']] = (old_vehicle, old_od)
        cold_counter[(old_vehicle, old_od)] += 1

    log('5/6 输出分区文件：开发期真实路径 / 测试期纯OD / 测试真值互相隔离')
    all_col_out = master_cols + ['split_stage']
    # Step6导出只保留原样6+6特征和四个身份/时间列，保留原始字段名和原值。
    step_cols_out = STEP6_REQUIRED + ['split_stage']
    index_cols = ['trip_id','VehicleID','start_time','end_time', 'split_stage',
                  'origin_node','destination_node','od_straight_km']
    profile_cols = ['trip_id', 'VehicleID','start_time', 'end_time',
                    'has_history_vehicle', 'has_seen_od']
    truth_cols = ['trip_id','VehicleID','start_time','end_time', 'origin_node','destination_node',
                  'path_len_km','duration_min','edge_seq_uvk_json']
    handles = []
    try:
        writers = {}
        for key, cols in [('index',index_cols),('dev_od',all_col_out),('blind_od',OD_ONLY_COLS),
                          ('profile_index',profile_cols),('truth',truth_cols),
                          ('excluded',AUDIT_COLS),('features',step_cols_out)]:
            handle, writer = utf_csv(out/names[key], cols)
            handles.append(handle)
            writers[key] = writer
        for x in all_items:
            r = x['record']
            tid = x['trip_id']
            stage = x['split']
            if stage == 'excluded':
                continue
            writers['index'].writerow({**r, 'split_stage': stage})
            if stage != 'future_test':
                writers['dev_od'].writerow({**r, 'split_stage': stage})
                writers['features'].writerow({**x['step6'], 'split_stage': stage})
            else:
                # 仅输出白名单列。没有VehicleID、路径长度、行程时长、真实路段JSON等。
                writers['blind_od'].writerow(r)
                old_v, old_od = test_index[tid]
                writers['profile_index'].writerow({
                    'trip_id': tid, 'VehicleID': x['vehicle'],
                    'start_time': r['start_time'], 'end_time': r['end_time'],
                    'has_history_vehicle': old_v, 'has_seen_od': old_od})
                writers['truth'].writerow(r)
        for r in excluded:
            writers['excluded'].writerow(r)
    finally:
        for h in handles:
            h.close()

    # 统计表：所有数值均从真实源表按时间计算。
    stage_ranges = [
      ('calibration','2024-12-01','2024-12-14','UDPPM++原型拟合'),
      ('model_train','2024-12-15','2024-12-23','路径选择模型训练'),
      ('validation','2024-12-24','2024-12-28','内部验证与调参'),
      ('future_test','2024-12-29','2024-12-31','严格独立测试'),
    ]
    with (out/names['counts']).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['stage','date_start','date_end','n_od','n_vehicle','usage'])
        for s,a,b,use in stage_ranges:
            w.writerow([s,a,b,stage_counts[s],len(per_stage_vehicles[s]),use])
        w.writerow(['excluded','','',len(excluded),len({x['VehicleID'] for x in excluded}),'跨边界及域外记录，禁止入模'])
    with (out/names['daily']).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['date','stage','n_od'])
        for day in range(1,32):
            d = date(2024,12,day)
            w.writerow([str(d),stage_for_day(d),daily_counts[str(d)]])
    with (out/names['cold']).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['has_history_vehicle','has_seen_od','n_test_od'])
        for known_v in (0,1):
            for known_od in (0,1):
                w.writerow([known_v,known_od,cold_counter[(known_v,known_od)]])

    log('6/6 编写完整性审计JSON并基于真实计数绘图')
    manifest = {
        'version':'chapter4_module1_chrono_split_v1.0',
        'created_local':datetime.now().isoformat(timespec='seconds'),
        'source_files':{
            'step41_master':file_info(args.master,args.sha256),
            'chapter3_step6':file_info(args.step6,args.sha256),
        },
        'time_policy':{
            'naive_source_timezone':args.naive_tz,
            'boundary_timezone':args.local_tz,
            'calibration':'2024-12-01 through 2024-12-14',
            'model_train':'2024-12-15 through 2024-12-23',
            'validation':'2024-12-24 through 2024-12-28',
            'future_test':'2024-12-29 through 2024-12-31',
            'cross_phase_trip_policy':'exclude_if_start_or_end_fall_in_different_phases',
        },
        'feature_policy':{
            'econ_exact_6':ECON6,
            'comfort_exact_6':COMF6,
            'unique_columns_in_step6_export':BEHAVIOR_COLS,
            'no_change_to_chapter3':True,
            'test_behavior_features_exported_to_model':False,
        },
        'counts':{
            'step6_all':len(third),'step41_all':len(all_items),
            'stage_od':dict(stage_counts),'stage_vehicle':{s:len(per_stage_vehicles[s]) for s,*_ in stage_ranges},
            'development_28d':len(dev),'future_3d':len(test),
            'excluded':len(excluded),'excluded_reasons':dict(reasons),
            'test_history_vehicle_count':sum(n for (v,_),n in cold_counter.items() if v),
            'test_cold_start_count':sum(n for (v,_),n in cold_counter.items() if not v),
            'test_seen_od_count':sum(n for (_,od),n in cold_counter.items() if od),
            'test_unseen_od_count':sum(n for (_,od),n in cold_counter.items() if not od),
        },
        'checks':{
            'dev_test_trip_id_disjoint':True,
            'dev_end_precedes_test_start':True,
            'master_ids_subset_of_step6':True,
            'master_vehicle_and_timestamps_match_step6':True,
            'blind_od_columns_exact_whitelist':True,
            'test_truth_stored_separately':True,
        },
        'outputs':{k:str(out/v) for k,v in names.items()},
        'warning':[
            '测试纯OD是从轨迹实际起终点恢复的离线OD模拟，不是实时出发前输入。',
            'Step41 OD为HMM道路边首u和尾v，不是原始GPS起终点，也未必是完整门到门出行。',
            '本模块不训练UDPPM++、不生成候选、不计算选择准确率。',
            '最后3天真值CSV只能供最终评价，严禁用于候选生成、特征、评分或调参。',
            '第三章整月原型不可直接用于28天训练/3天未知测试，必须在允许的校准期内重新拟合同一方法。',
            '有历史车辆的判定仅基于本次Step41有效OD，不等于确认同一真实驾驶员。',
        ]
    }
    dumps_json(out/names['manifest'],manifest)
    (out/names['readme']).write_text(
        '【第四章模块一输出使用规范】\n'
        '04_前28天开发OD与真实路径_仅训练.csv.gz：允许进入后续模型学习，不得称为独立测试。\n'
        '09_前28天_第三章原始6加6特征.csv.gz：只在开发期使用，原第三章6+6特征不变。\n'
        '05_最后3天_纯OD推理输入_绝无真实路线.csv：后续自然候选生成器只允许读取此文件。\n'
        '06_最后3天_历史画像匹配索引_受控.csv：用于对接此前冻结的历史车辆画像；不得读入本次真值。\n'
        '07_最后3天_真实路径答案_仅评价禁止推理.csv.gz：只用于全部推荐结束后的独立评价。\n'
        '01_全量OD时间分组索引.csv：仅用于阶段索引和审计，不能替代纯OD输入进入推理。\n'
        '时区重要：本次源时间无时区时按'+args.naive_tz+'解释，按'+args.local_tz+'划分日期。\n'
        '请先核查实际源数据时间语义，否则所有阶段划分都有风险。\n'
        '阶段边界跨越行程隔离，详见08_跨边界与排除记录.csv。\n',
        encoding='utf-8-sig')
    draw_daily_plot(daily_counts,out)
    log('任务完成，全部统计来自真实源文件。')
    log('各阶段有效OD：' + ', '.join(f'{s}={stage_counts[s]:,}' for s,*_ in stage_ranges))
    log(f'冷启动测试OD={manifest["counts"]["test_cold_start_count"]:,}; '
        f'历史未见OD={manifest["counts"]["test_unseen_od_count"]:,}')
    log('输出目录：'+str(out))
    log('重要：后续自然候选生成只能读取05_最后3天_纯OD推理输入。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log('【失败】'+str(exc))
        raise
