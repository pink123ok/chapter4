# -*- coding: utf-8 -*-
"""
第四章模块五 v2｜需求驱动的三场景Top-3推荐（模块四冻结、盲测防泄漏）

【研究边界】
- 读取模块三已经自然生成的测试期候选（道路属性 + 有向边序列）；不重新生成路线。
- 读取模块四已经冻结的六模型在最后3天自然候选上的预测概率；不重新拟合。
- 读取模块三的OD覆盖审计和同一份静态路网用于“长度加权共享边”计算；
  绝不读取后三天真实HMM行驶路线、chosen、VehicleID 或评价答案。
- 经济性代理指标：路线长度（不是实际花费/油耗）。
- 舒适性代理指标：转弯密度、路段密度、地方道路占比、非快速路占比
  （不是实测乘坐舒适度）。
- 模块五设计的是预先固定权重的启发式、多目标偏好重排序；权重没有根据
  最后3天真值调优，不能宣称是经过数据驱动训练的最优权重。
- 产出Top3方案、逐候选目标分数、路线重叠、消融/基线、全OD覆盖与来源审计。
  正式效果比较/统计显著性属于模块六，不能直接将本模块当成效果证明。

【Windows用法】
  # 先确保已经完成 python 4step-4.py --phase predict
  python 5step-5.py --phase check
  python 5step-5.py --phase run
  # 可选：如果模块四已有pilot_od100_seed42的predict结果，则先做对应pilot
  python 5step-5.py --phase run --pilot --pilot-od 100

依赖：pip install numpy pandas networkx
所有正式数值仅来自真实文件，缺失文件直接报错；不会模拟补齐。
特别注意：09~14的需求类型由候选道路代理规则定义，不能称为UDPPM++软聚类概率；
由最后3天实际HMM路线事后推断‘模拟用户需求’的步骤必须在模块六开展。
v2的 --phase extend 专门面向已经正式运行过的v1，不改变其01~08输出。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

BASE = Path(r'C:\study\轨迹数据预处理')
ROOT = BASE / '曼谷GPS_正式清洗_202412' / 'FULL_全量' / '第四章_28天训练_3天独立测试'
M1 = ROOT / '模块一_时间划分与隔离'
M3 = ROOT / '模块三_未知OD自然候选与静态道路属性' / 'full_K10'
M4 = ROOT / '模块四_条件选择六模型与未知OD盲排序'
OUT = ROOT / '模块五_双维偏好_三模式Top3多目标推荐'
GRAPH = BASE / 'luwangdata' / '曼谷_路网_兼容旧版.graphml'

CID = 'candidate_id_TRACE_NOT_FEATURE'
MODEL_NAMES = (
    'MNL_L2', 'PSL_Count_L2', 'PSL_LengthWeighted_L2',
    'UDPPM_Econ_PSL_L2', 'UDPPM_Comf_PSL_L2', 'UDPPM_Dual_PSL_L2'
)
SCENARIOS = ('economic', 'comfortable', 'balanced')
SCENARIO_CHINESE = {'economic':'经济优先', 'comfortable':'舒适优先', 'balanced':'综合均衡'}
VARIANTS = ('proposed', 'no_diversity', 'no_profile', 'utility_only')
PROHIBITED_FIELDS = {'chosen', 'VehicleID', 'edge_seq_real', 'is_real_added',
                     'real_added_TRACE_NOT_FEATURE', 'observed_route', 'observed_edges'}
# 预先固定的评分方法；所有归一化仅在当前OD的自然候选内部计算。
# 各场景每行权重之和 = 1，权重选择不是由未来OD的真实路线拟合。
SCORE_WEIGHTS = {
    'economic': {'specialized':.55, 'dual':.10, 'economy_proxy':.35, 'comfort_proxy':0.},
    'comfortable': {'specialized':.55, 'dual':.10, 'economy_proxy':0., 'comfort_proxy':.35},
    'balanced': {'specialized':0., 'dual':.50, 'economy_proxy':.25, 'comfort_proxy':.25},
}
# 舒适度仅为结构属性代理成本，越小越好。转弯/路段密度较低、local/非motorway占比较低
# 是工程假设，不是路况、颠簸、速度或驾乘者感受的实测数据。
COMFORT_COST_WEIGHTS = {
    'turns_per_km': .40,
    'segment_density_seg_per_km': .30,
    'ratio_local': .20,
    'one_minus_ratio_motorway': .10,
}


def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


def ensure_file(path: Path, label: str):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'缺少{label}: {path}\n请先运行第四步正式predict；禁止模拟补齐真实数据。')


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jread(path: Path):
    ensure_file(path, 'JSON审计文件')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_name(path.name + '.writing')
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                            default=str), encoding='utf-8-sig')
    q.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_name(path.name + '.writing')
    df.to_csv(q, index=False, float_format='%.10g', encoding='utf-8-sig',
              compression='gzip' if path.suffix == '.gz' else None)
    q.replace(path)


def read_csv(path, **kwargs):
    ensure_file(path, '数据文件')
    return pd.read_csv(path, encoding='utf-8-sig', low_memory=False,
                       dtype={'trip_id': 'string', CID: 'string'}, **kwargs)


def ensure_columns(df, required, label):
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f'{label}缺少字段：{absent}')
    blocked = sorted(PROHIBITED_FIELDS & set(df.columns))
    if blocked:
        raise ValueError(f'{label}出现禁止读取的真值/泄漏字段：{blocked}')


def check_files(a):
    work4 = a.m4 / (f'pilot_od{a.pilot_od}_seed{a.seed}' if a.pilot else 'formal_K10')
    files = {
        'model': work4 / '06_正式模型参数与训练来源.json',
        'predict_audit': work4 / '09_最后3天盲排序与时间隔离审计.json',
        'predictions': work4 / '07_最后3天六模型逐路线盲排序.csv.gz',
        'static': a.m3 / '02_未知OD候选静态属性与PathSize_模型输入.csv.gz',
        'raw': a.m3 / '01_真实自然候选路径_含有向边序列.csv.gz',
        'od_audit': a.m3 / '03_全部OD搜索与Top3审计.csv',
        'm3_manifest': a.m3 / '07_模块三完整性_路网及数据隔离审计.json',
        'blind_od': a.m1 / '05_最后3天_纯OD推理输入_绝无真实路线.csv',
        'm1_manifest': a.m1 / '11_完整性防泄漏审计.json',
        'graph': a.graphml,
    }
    for name, p in files.items():
        ensure_file(p, name)
    model = jread(files['model'])
    pred_audit = jread(files['predict_audit'])
    m3 = jread(files['m3_manifest'])
    m1 = jread(files['m1_manifest'])
    if bool(model.get('pilot')) != bool(a.pilot):
        raise ValueError('模块四模型的pilot属性与当前运行模式不一致')
    if pred_audit.get('mode') != ('PILOT_NOT_FOR_PAPER' if a.pilot else 'FULL_REAL'):
        raise ValueError('模块四预测审计不是当前pilot/full模式')
    if len(model.get('models', {})) != 6 or set(model['models']) != set(MODEL_NAMES):
        raise ValueError('模块四正式模型不是要求的六种模型')
    if pred_audit.get('actual_observed_routes_read') != 'NEVER' or \
       pred_audit.get('chosen_test_labels_read') != 'NEVER':
        raise ValueError('模块四盲排序审计未声明真值隔离，拒绝继续')
    if pred_audit.get('source_model_sha256') != digest(files['model']):
        raise ValueError('模块四模型JSON与此前predict时使用的模型哈希不同；请重新运行模块四predict')
    if pred_audit.get('source_test_candidates_sha256') != digest(files['static']):
        raise ValueError('模块三静态特征与模块四predict所用文件哈希不同')
    if not m3.get('is_formal_full') or int(m3.get('selected_od', -1)) != 4469:
        raise ValueError('模块三必须使用正式完整4469条OD自然候选成果')
    if int(m1.get('counts', {}).get('future_3d', -1)) != 4469:
        raise ValueError('模块一最后3天OD数量不符')
    graph_sha = m3.get('source_config', {}).get('input_source', {}).get('graphml', {}).get('sha256')
    if not graph_sha:
        raise ValueError('模块三审计没有GraphML SHA256，无法验证路线重叠计算与搜路使用同一路网')
    if a.phase == 'run' or a.verify_graph_in_check:
        if digest(files['graph']) != graph_sha:
            raise ValueError('GraphML哈希与模块三不同，不能拿另一路网计算道路长度重叠')
    # 不导入任何被留作独立评价答案的文件。
    # 模块三raw仅含“算法生成的”路线，而不是历史实际HMM行驶道路序列。
    with open(files['predictions'], 'rb') as fh:
        pass
    log(f'来源核验成功：模块四={work4.name}，测试OD=4,469，模型=6个，真值未读取')
    return files, model, pred_audit, m3, m1


def one_dimension_good(x):
    """对同一OD的非缺失实数，min-max映射到[0,1]（1=更优）。全部相等记0.5。"""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('必要评分指标包含NaN/Inf；不能用虚构数值补齐')
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.full(len(x), .5, dtype=float)
    return (x - lo) / (hi - lo)


def one_dimension_cost(x):
    return 1. - one_dimension_good(x)


def parse_edge_sequence(text):
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f'自然候选有向道路序列JSON非法：{str(text)[:100]}') from e
    if not isinstance(raw, list) or not raw:
        raise ValueError('自然候选有向边序列为空')
    seq = []
    for e in raw:
        if not isinstance(e, (list, tuple)) or len(e) != 3 or any(t is None for t in e):
            raise ValueError(f'非法路网边三元组 {e!r}')
        seq.append(tuple(map(str, e)))
    if any(e1[1] != e2[0] for e1,e2 in zip(seq, seq[1:])):
        raise ValueError('自然候选有向道路序列存在断裂')
    if len(seq) != len(set(seq)):
        # 有环路在路径推荐中可疑，但不额外删除（保持模块三集合不变）。
        pass
    return seq


def build_used_edge_lengths(graphml, wanted):
    log(f'读取相同GraphML用于长度加权重叠：{graphml}')
    g = nx.read_graphml(str(graphml), force_multigraph=True)
    if not g.is_directed():
        raise ValueError('模块三GraphML不是有向图')
    lens = {}
    for u, v, k, d in g.edges(keys=True, data=True):
        key = (str(u), str(v), str(k))
        if key not in wanted:
            continue
        try:
            length = float(d['length'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'路线使用的边缺少真实长度：{key}')
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f'路线使用的边长度非法：{key}: {length}')
        lens[key] = length
    missing = wanted - lens.keys()
    if missing:
        raise ValueError(f'自然候选路线中有{len(missing)}条边在路网中未找到：示例={next(iter(missing))}')
    log(f'已核验真实路段长度：{len(lens):,}条不同有向边')
    return lens


def weighted_overlap(s1, s2, edge_length):
    """按有向(u,v,key)边长度的加权Jaccard；方向不同的边不视为相同道路。"""
    common = s1 & s2
    union = s1 | s2
    shared = sum(edge_length[e] for e in common)
    denom = sum(edge_length[e] for e in union)
    if denom <= 0:
        raise ValueError('道路重叠计算出现零长度并集')
    return shared / denom, shared


def pareto_levels(cost1, cost2):
    """仅用于解释性指标；按经济代理成本和舒适代理成本进行非支配排序。"""
    c1, c2 = np.asarray(cost1), np.asarray(cost2)
    n = len(c1)
    left = set(range(n))
    ranks = np.zeros(n, dtype=int)
    rank = 1
    while left:
        fronts=[]
        for i in sorted(left):
            dominated = any((c1[j] <= c1[i]+1e-12 and c2[j] <= c2[i]+1e-12)
                            and (c1[j] < c1[i]-1e-12 or c2[j] < c2[i]-1e-12)
                            for j in left if j != i)
            if not dominated:
                fronts.append(i)
        if not fronts:
            raise AssertionError('Pareto非支配排序无法获得前沿')
        for i in fronts:
            ranks[i]=rank
        left.difference_update(fronts)
        rank += 1
    return ranks


def validate_and_load(files, a, m3, pred_audit):
    required_static = ['trip_id', CID, 'total_len_km', 'ratio_to_shortest',
                       'turns_per_km', 'segment_density_seg_per_km',
                       'ratio_local', 'ratio_motorway']
    required_prediction = ['trip_id', CID, 'model', 'utility',
                           'choice_probability', 'rank_in_od']
    required_raw = ['trip_id', CID, 'edge_seq_uvk_json',
                    'route_length_km', 'origin_node', 'destination_node']
    ft = read_csv(files['static'])
    raw = read_csv(files['raw'])
    pred = read_csv(files['predictions'])
    oa = read_csv(files['od_audit'])
    blind = read_csv(files['blind_od'])
    for df, cols, name in [(ft,required_static,'模块三静态属性'),
                           (raw,required_raw,'模块三自然路线'),
                           (pred,required_prediction,'模块四六模型盲评分'),
                           (oa,['trip_id','natural_count','top3_available'],'模块三OD覆盖审计'),
                           (blind,['trip_id'],'模块一纯OD输入')]:
        ensure_columns(df, cols, name)
    if len(ft) != int(m3['all_generated_natural_candidates']):
        raise ValueError('模块三特征总行数与审计不一致')
    if len(raw) != len(ft):
        raise ValueError('模块三自然路径与静态特征行数不同')
    if len(oa)!=4469 or len(blind)!=4469 or oa.trip_id.duplicated().any() or blind.trip_id.duplicated().any():
        raise ValueError('模块一/三最终测试OD未完整覆盖4469个唯一OD')
    if set(oa.trip_id)!=set(blind.trip_id):
        raise ValueError('模块三审计OD集合与模块一最后3天不一致')
    for label, df in [('特征', ft), ('自然路线', raw)]:
        if df[CID].duplicated().any() or df.trip_id.isna().any() or df[CID].isna().any():
            raise ValueError(f'模块三{label}包含重复或空候选ID')
    for label, df in [('模块三特征', ft), ('模块三路线', raw)]:
        if not set(df.trip_id).issubset(set(blind.trip_id)):
            raise ValueError(f'{label}出现测试期外OD')
    if len(pred) != int(pred_audit['n_saved_all_model_prediction_rows']):
        raise ValueError('模块四预测文件总行数与模块四审计不一致')
    if pred[['trip_id',CID,'model']].duplicated().any():
        raise ValueError('模块四同一路线模型得分重复')
    if set(pred.model)!=set(MODEL_NAMES):
        raise ValueError('模型预测文件未包含约定的全部六种模型')
    if len(pred) != 6 * int(pred_audit['n_test_candidate_rows']):
        raise ValueError('模块四预测六模型行数不完整')
    if pred[[CID,'model']].isna().any().any():
        raise ValueError('模块四模型或候选ID存在缺失')
    for c in ('utility','choice_probability','rank_in_od'):
        x = pd.to_numeric(pred[c], errors='coerce')
        if not np.isfinite(x.to_numpy(float)).all():
            raise ValueError(f'模块四预测字段{c}出现非有限数值')
        pred[c] = x
    if ((pred.choice_probability < 0) | (pred.choice_probability > 1)).any():
        raise ValueError('模块四模型选择概率超出[0,1]')
    p_sum = pred.groupby(['trip_id','model'],sort=False).choice_probability.sum()
    if not np.allclose(p_sum.to_numpy(float), 1., atol=1e-6):
        raise ValueError('六模型在单个OD内的预测概率总和不是1')
    if not set(pred[CID]).issubset(set(ft[CID])):
        raise ValueError('模块四预测包含模块三之外的候选')
    if not a.pilot and set(pred[CID]) != set(ft[CID]):
        raise ValueError('正式结果必须给模块三全部自然候选打分，不能缺少部分候选')
    # 逐模型的候选ID必须严格一致，不允许悄悄交叉缺失。
    cnt = pred.groupby(CID).model.nunique()
    if not cnt.eq(6).all():
        raise ValueError('有些自然候选没有完整六模型结果')
    if a.pilot:
        ids=set(pred.trip_id)
        if len(ids)!=int(pred_audit['n_scored_od']):
            raise ValueError('pilot预测OD数与第四步审计不符')
        ft=ft[ft.trip_id.isin(ids)].copy()
        raw=raw[raw.trip_id.isin(ids)].copy()
        oa=oa[oa.trip_id.isin(ids)].copy()
        blind=blind[blind.trip_id.isin(ids)].copy()
        if len(ids)!=a.pilot_od:
            log(f'提醒：该pilot中实际可预测OD为{len(ids)}，与参数{a.pilot_od}不同')
    else:
        if int(pred_audit['n_all_test_od'])!=4469:
            raise ValueError('正式模块四预测审计未包含全部4469个OD分母')
    if set(ft[CID]) != set(raw[CID]) or set(ft[CID]) != set(pred[CID]):
        raise ValueError('模块三特征、真实自然候选序列、模块四六模型得分候选ID集合不相等')
    if not ft.set_index(CID).trip_id.sort_index().equals(raw.set_index(CID).trip_id.sort_index()):
        raise ValueError('模块三静态特征与自然边序列存在候选ID归属OD不一致')
    n_by_od=ft.groupby('trip_id').size()
    expected=oa.set_index('trip_id').natural_count.astype(int)
    if not n_by_od.reindex(expected.index,fill_value=0).equals(expected):
        raise ValueError('逐OD自然候选数量与模块三审计不一致')
    pv=pred.pivot(index=CID, columns='model', values='choice_probability')
    if pv.isna().any().any() or set(pv.columns)!=set(MODEL_NAMES):
        raise ValueError('六模型概率宽表含缺失值')
    pv.columns = ['prob_'+x for x in pv.columns]
    cols = ['trip_id',CID,'total_len_km','ratio_to_shortest',
            'turns_per_km','segment_density_seg_per_km','ratio_local','ratio_motorway']
    ft=ft[cols].copy()
    joined=ft.join(pv,on=CID,how='left',validate='one_to_one')
    if joined.filter(like='prob_').isna().any().any():
        raise ValueError('特征与六模型预测未能完全关联')
    joined=joined.sort_values(['trip_id',CID],kind='mergesort').reset_index(drop=True)
    raw=raw.sort_values(['trip_id',CID],kind='mergesort').reset_index(drop=True)
    for col in ('total_len_km','ratio_to_shortest','turns_per_km',
                'segment_density_seg_per_km','ratio_local','ratio_motorway'):
        values=pd.to_numeric(joined[col],errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'自然候选关键属性{col}缺失/非有限；不能模拟插值')
        joined[col]=values
    if (joined.total_len_km<=0).any() or (joined.ratio_to_shortest<1-1e-4).any():
        raise ValueError('模块三候选长度/最短路径比例异常')
    for ratio in ('ratio_local','ratio_motorway'):
        if ((joined[ratio]<-1e-8)|(joined[ratio]>1+1e-8)).any():
            raise ValueError(f'{ratio}道路占比不在[0,1]')
    if (joined.ratio_local+joined.ratio_motorway>1+1e-6).any():
        raise ValueError('地方道路和高速道路占比之和超过1')
    log(f'正式关联核验：{joined.trip_id.nunique():,}个可达OD，'
        f'{len(joined):,}条候选，六模型概率均完整；总OD分母={len(oa):,}')
    return joined, raw, oa, blind


def load_routes_and_lens(raw, graphml):
    seq_by_id = {}
    wanted = set()
    for cid, txt in zip(raw[CID].astype(str),raw.edge_seq_uvk_json):
        seq = parse_edge_sequence(txt)
        seq_by_id[cid] = seq
        wanted.update(seq)
    lengths = build_used_edge_lengths(graphml, wanted)
    reported = pd.to_numeric(raw.route_length_km,errors='coerce').to_numpy(float)
    if not np.isfinite(reported).all() or (reported<=0).any():
        raise ValueError('模块三真实自然候选报告长度无效')
    bad=[]
    for cid, rl in zip(raw[CID].astype(str),reported):
        calc=sum(lengths[e] for e in seq_by_id[cid])/1000.
        if abs(calc-rl)>max(.0001,.00005*rl):
            bad.append((cid,rl,calc))
    if bad:
        raise ValueError(f'模块三自然候选路段长度与GraphML不一致：{len(bad)}条，如{bad[0]}')
    return seq_by_id,lengths


def mode_scores(group):
    """固定两目标代理 + 三种已冻结模型分数，分组内按真实当前OD自然候选归一化。"""
    n=len(group)
    out={}
    route_len=group.total_len_km.to_numpy(float)
    economy=one_dimension_cost(route_len)
    costs={
        'turns_per_km':group.turns_per_km.to_numpy(float),
        'segment_density_seg_per_km':group.segment_density_seg_per_km.to_numpy(float),
        'ratio_local':group.ratio_local.to_numpy(float),
        'one_minus_ratio_motorway':1.-group.ratio_motorway.to_numpy(float),
    }
    comfort_cost=sum(w*one_dimension_good(costs[col]) for col,w in COMFORT_COST_WEIGHTS.items())
    comfort=1.-comfort_cost
    p={m:one_dimension_good(group['prob_'+m].to_numpy(float)) for m in MODEL_NAMES}
    pareto=pareto_levels(1.-economy,comfort_cost)
    for scenario in SCENARIOS:
        w=SCORE_WEIGHTS[scenario]
        model = ('UDPPM_Econ_PSL_L2' if scenario=='economic'
                 else 'UDPPM_Comf_PSL_L2' if scenario=='comfortable'
                 else 'UDPPM_Dual_PSL_L2')
        # 方案一：历史偏好增强；完整模型用于引导当前场景的预估偏好相关性
        target = (w['specialized']*p[model]+w['dual']*p['UDPPM_Dual_PSL_L2']
                  +w['economy_proxy']*economy+w['comfort_proxy']*comfort)
        # 消融：剔除历史画像的模型项，改由PSL_Length的同一分数替代，目标权重不变。
        no_profile = ((w['specialized']+w['dual'])*p['PSL_LengthWeighted_L2']
                      +w['economy_proxy']*economy+w['comfort_proxy']*comfort)
        # 完全只使用场景对应模型自然候选概率排名：不加显式经济/舒适目标、不去重。
        raw_model = p[model]
        for v,z in [('proposed',target),('no_diversity',target),
                    ('no_profile',no_profile),('utility_only',raw_model)]:
            if len(z)!=n or not np.isfinite(z).all() or z.min() < -1e-9 or z.max()>1+1e-9:
                raise ValueError(f'多目标评分出错：{scenario}/{v}')
        out[scenario]={
            'economic_proxy_score':economy,
            'comfort_proxy_score':comfort,
            'comfort_proxy_cost':comfort_cost,
            'pareto_rank':pareto,
            'proposed':target,
            'no_diversity':target,
            'no_profile':no_profile,
            'utility_only':raw_model,
            'relevance_model':model,
        }
    return out


def rank_topk(ids, scores, overlap, k, lam, maximum_overlap, diversity=True):
    """固定、确定性贪心重排。严格阈值无法选够K时记录放宽次数，不虚构候选。"""
    chosen=[]
    relaxed=[]
    effective=[]
    max_available=min(k,len(ids))
    while len(chosen)<max_available:
        remaining=[i for i in range(len(ids)) if i not in chosen]
        opts=[]
        for i in remaining:
            max_sim=max((float(overlap[i,j]) for j in chosen),default=0.)
            diversity_penalty = lam*max_sim if diversity and chosen else 0.
            adjusted=float(scores[i])-diversity_penalty
            feasible=(not diversity or not chosen or max_sim<=maximum_overlap+1e-12)
            opts.append((i,adjusted,max_sim,feasible))
        has_feasible=any(q[3] for q in opts)
        pool=([q for q in opts if q[3]] if has_feasible else opts)
        # 按得分降序、相似度升序、候选ID字典序确定性破同分。
        best=min(pool,key=lambda q:(-round(q[1],12),round(q[2],12),str(ids[q[0]])))
        chosen.append(best[0])
        relaxed.append(bool(diversity and len(chosen)>1 and not has_feasible))
        effective.append((best[1],best[2]))
    return chosen,relaxed,effective


def process_one_od(group, seq_by_id, lengths, a):
    group=group.reset_index(drop=True)
    ids=group[CID].astype(str).tolist()
    n=len(ids)
    sets=[set(seq_by_id[c]) for c in ids]
    overlap=np.eye(n,dtype=float)
    pair_rows=[]
    tid=str(group.trip_id.iloc[0])
    for i in range(n):
        for j in range(i+1,n):
            ov,shared=weighted_overlap(sets[i],sets[j],lengths)
            if ov < -1e-10 or ov > 1+1e-10:
                raise ValueError(f'异常长度加权路线Jaccard：{tid}: {ov}')
            overlap[i,j]=overlap[j,i]=ov
            pair_rows.append({'trip_id':tid,'candidate_id_a':ids[i], 'candidate_id_b':ids[j],
                              'pairwise_length_jaccard':ov,'shared_directed_edge_length_km':shared/1000.})
    mode = mode_scores(group)
    all_rows=[];top3_rows=[];baseline_rows=[];per_od=[]
    for scenario,z in mode.items():
        for i,cid in enumerate(ids):
            all_rows.append({
                'trip_id':tid,'scenario':scenario,'scenario_zh':SCENARIO_CHINESE[scenario],
                CID:cid,'candidate_count':n,
                'total_len_km':float(group.total_len_km.iloc[i]),
                'ratio_to_shortest':float(group.ratio_to_shortest.iloc[i]),
                'turns_per_km':float(group.turns_per_km.iloc[i]),
                'segment_density_seg_per_km':float(group.segment_density_seg_per_km.iloc[i]),
                'ratio_local':float(group.ratio_local.iloc[i]),
                'ratio_motorway':float(group.ratio_motorway.iloc[i]),
                'economic_proxy_score':float(z['economic_proxy_score'][i]),
                'comfort_proxy_score':float(z['comfort_proxy_score'][i]),
                'comfort_proxy_cost':float(z['comfort_proxy_cost'][i]),
                'pareto_rank':int(z['pareto_rank'][i]),
                'scenario_relevance_model':z['relevance_model'],
                'prediversity_score':float(z['proposed'][i]),
                'model_UDPPM_Econ_probability':float(group['prob_UDPPM_Econ_PSL_L2'].iloc[i]),
                'model_UDPPM_Comf_probability':float(group['prob_UDPPM_Comf_PSL_L2'].iloc[i]),
                'model_UDPPM_Dual_probability':float(group['prob_UDPPM_Dual_PSL_L2'].iloc[i]),
                'model_PSL_Length_probability':float(group['prob_PSL_LengthWeighted_L2'].iloc[i]),
            })
        for variant in VARIANTS:
            diverse = variant in ('proposed','no_profile')
            order,relaxed,eff=rank_topk(ids,z[variant],overlap, a.topk,
                                        a.diversity_lambda,a.maximum_overlap,diversity=diverse)
            pairs=[]
            for ix,j in enumerate(order):
                for jj in order[ix+1:]:
                    pairs.append(float(overlap[j,jj]))
            worst=max(pairs,default=0.) if len(order)>=2 else None
            record = {
                'trip_id':tid,'scenario':scenario,'variant':variant,
                'n_natural_candidates':n,'n_recommended':len(order),
                'top3_available':int(len(order)>=3),
                'relaxed_n':int(sum(relaxed)),
                'max_selected_pairwise_jaccard':worst,
                'mean_selected_pairwise_jaccard':float(np.mean(pairs)) if pairs else None,
                'recommended_mean_len_km':float(np.mean(group.total_len_km.to_numpy(float)[order])),
                'recommended_mean_ratio_to_shortest':float(np.mean(group.ratio_to_shortest.to_numpy(float)[order])),
                'recommended_mean_economy_score':float(np.mean(z['economic_proxy_score'][order])),
                'recommended_mean_comfort_score':float(np.mean(z['comfort_proxy_score'][order])),
                'n_recommended_pareto_first':int(np.sum(z['pareto_rank'][order]==1)),
            }
            per_od.append(record)
            for j,(i,was_relaxed,(adjusted,max_sim)) in enumerate(zip(order,relaxed,eff),start=1):
                row={
                    'trip_id':tid,'scenario':scenario,'scenario_zh':SCENARIO_CHINESE[scenario],
                    'variant':variant,'rank_in_scenario':j,CID:ids[i],
                    'candidate_count':n,'total_len_km':float(group.total_len_km.iloc[i]),
                    'ratio_to_shortest':float(group.ratio_to_shortest.iloc[i]),
                    'turns_per_km':float(group.turns_per_km.iloc[i]),
                    'segment_density_seg_per_km':float(group.segment_density_seg_per_km.iloc[i]),
                    'ratio_local':float(group.ratio_local.iloc[i]),
                    'ratio_motorway':float(group.ratio_motorway.iloc[i]),
                    'economic_proxy_score':float(z['economic_proxy_score'][i]),
                    'comfort_proxy_score':float(z['comfort_proxy_score'][i]),
                    'pareto_rank':int(z['pareto_rank'][i]),
                    'original_score':float(z[variant][i]),
                    'diversity_adjusted_score_at_selection':float(adjusted),
                    'max_overlap_with_earlier_selected':float(max_sim),
                    'overlap_threshold_relaxed_at_selection':int(was_relaxed),
                    'relevance_model':z['relevance_model'],
                    'model_UDPPM_Dual_probability':float(group['prob_UDPPM_Dual_PSL_L2'].iloc[i]),
                }
                if variant=='proposed':top3_rows.append(row)
                else:baseline_rows.append(row)
    return all_rows,top3_rows,baseline_rows,per_od,pair_rows


def output_summary(per_od, all_od_audit, a):
    df = pd.DataFrame(per_od)
    all_od = len(all_od_audit)
    rows=[]
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            d=df[(df.scenario==scenario) & (df.variant==variant)]
            if len(d)!=int((all_od_audit.natural_count>0).sum()):
                raise AssertionError('统计中可用OD数量与OD审计不一致')
            rows.append({
                'scenario':scenario,'scenario_zh':SCENARIO_CHINESE[scenario],
                'variant':variant,'all_requested_od':all_od,'scored_od':len(d),
                'at_least_one_route_od':int((d.n_recommended>=1).sum()),
                'top3_available_od':int(d.top3_available.sum()),
                'top3_available_rate_all_od':float(d.top3_available.sum()/all_od),
                'mean_recommended_len_km':float(d.recommended_mean_len_km.mean()),
                'mean_recommended_ratio_to_shortest':float(d.recommended_mean_ratio_to_shortest.mean()),
                'mean_economic_proxy_score':float(d.recommended_mean_economy_score.mean()),
                'mean_comfort_proxy_score':float(d.recommended_mean_comfort_score.mean()),
                'mean_max_pairwise_jaccard':float(d.max_selected_pairwise_jaccard.mean()),
                'mean_selected_pairwise_jaccard':float(d.mean_selected_pairwise_jaccard.mean()),
                'od_with_overlap_relaxation':int((d.relaxed_n>0).sum()),
                'count_pareto_first_in_recommendations':int(d.n_recommended_pareto_first.sum()),
                'warning':'仅候选集内部代理指标/路线重叠；尚非独立推荐有效性检验',
            })
    return pd.DataFrame(rows)


def run(a, files, model, pred_audit, m3):
    start=time.perf_counter()
    feature,raw,oa,blind = validate_and_load(files,a,m3,pred_audit)
    seq_by_id,lengths=load_routes_and_lens(raw,a.graphml)
    by_od={tid:g for tid,g in feature.groupby('trip_id',sort=True)}
    all_out=[];recommended=[];baselines=[];audit_rows=[];pairs=[]
    log(f'开始三模式多目标重排序：{len(by_od):,}个OD，{len(feature):,}条自然候选')
    for i,(tid,group) in enumerate(by_od.items(),1):
        a1,b1,c1,d1,e1=process_one_od(group,seq_by_id,lengths,a)
        all_out.extend(a1);recommended.extend(b1);baselines.extend(c1)
        audit_rows.extend(d1);pairs.extend(e1)
        if i%500==0 or i==len(by_od):
            log(f'三模式盲推荐 {i:,}/{len(by_od):,}个OD，主方法推荐记录={len(recommended):,}')
    result=pd.DataFrame(recommended)
    all_scores=pd.DataFrame(all_out)
    ablations=pd.DataFrame(baselines)
    od_stats=pd.DataFrame(audit_rows)
    pair_df=pd.DataFrame(pairs)
    if not a.pilot:
        if len(oa)!=4469 or len(by_od)!=int(m3['blind_test_1plus_od']):
            raise AssertionError('正式重排序的OD覆盖不等于模块三正式审计')
        if int((oa.natural_count.astype(int)>=3).sum())!=int(m3['blind_test_3plus_od']):
            raise AssertionError('模块三Top3来源统计不符')
    if len(result)!=3*sum(min(a.topk,int(n)) for n in oa.natural_count):
        raise AssertionError('三种模式推荐总行数与自然候选数量不符')
    if result[[CID,'scenario','trip_id','rank_in_scenario']].duplicated().any():
        raise AssertionError('推荐结果重复')
    if not set(result[CID]).issubset(set(feature[CID])):
        raise AssertionError('推荐结果出现非自然候选')
    if result.rank_in_scenario.max()>a.topk:
        raise AssertionError('TopK结果输出过量')
    # 保存包含失败OD的完整审计，避免只以可搜路OD为分母。
    coverage=oa[['trip_id','natural_count','top3_available']].copy()
    coverage['test_od_count_in_denominator']=True
    coverage['recommendation_status']=np.where(coverage.natural_count>0,'scored','no_natural_route')
    if not a.pilot and len(coverage)!=4469:
        raise AssertionError('总体OD审计必须包含最后3天全部OD')
    summary=output_summary(audit_rows,oa,a)
    audit_path=a.out / ('pilot_od'+str(a.pilot_od)+'_seed'+str(a.seed) if a.pilot else 'formal_K10')
    audit_path.mkdir(parents=True, exist_ok=True)
    atomic_csv(all_scores,audit_path/'01_全部候选_三模式多目标代理分数.csv.gz')
    atomic_csv(result,audit_path/'02_三模式Top3推荐_主方法.csv.gz')
    atomic_csv(ablations,audit_path/'03_三模式Top3推荐_对照与消融.csv.gz')
    atomic_csv(od_stats,audit_path/'04_每OD每模式_推荐质量和多样性审计.csv')
    atomic_csv(coverage,audit_path/'05_全部测试OD_候选与推荐覆盖审计.csv')
    atomic_csv(summary,audit_path/'06_三模式_推荐属性与可用率汇总.csv')
    atomic_csv(pair_df,audit_path/'07_逐OD_自然候选两两有向边长度重叠.csv.gz')
    elapsed=time.perf_counter()-start
    audit={
        'version':'chapter4_module5_blind_three_scenario_top3_v1.0',
        'mode':'PILOT_NOT_FOR_PAPER' if a.pilot else 'FULL_REAL',
        'created_at':datetime.now().isoformat(),
        'input_sources':{key:{'path':str(files[key]),'sha256':digest(files[key])}
                         for key in ('model','predict_audit','predictions','static','raw','od_audit','m3_manifest','blind_od','graph')},
        'module4_frozen_model_sha256':digest(files['model']),
        'model_retrained':'NEVER',
        'future_actual_hmm_route_read':'NEVER',
        'future_chosen_labels_read':'NEVER',
        'future_vehicle_id_read':'NEVER',
        'candidate_route_generated_or_augmented':'NEVER',
        'source_scores':'module4 six model blind choice probabilities',
        'all_test_od':len(oa),'scored_od':len(by_od),
        'top3_available_od':int((oa.natural_count>=3).sum()),
        'candidate_rows':len(feature),
        'main_top3_rows':len(result),'baseline_top3_rows':len(ablations),
        'per_od_audit_rows':len(od_stats),
        'length_jaccard_pair_rows':len(pair_df),
        'scenarios':SCENARIO_CHINESE,
        'variants':VARIANTS,
        'score_weights':SCORE_WEIGHTS,
        'comfort_cost_proxy_weights':COMFORT_COST_WEIGHTS,
        'economic_cost_proxy':'total_len_km, within-OD minmax normalization (smaller better)',
        'comfort_cost_proxy':'turns_per_km / segment_density / ratio_local / (1-ratio_motorway), within-OD minmax',
        'probability_normalization':'per model per OD minmax [0,1]; constants -> 0.5',
        'overlap':'directed (u,v,key) edge length weighted Jaccard, graph SHA matched module3',
        'topk':a.topk,'diversity_lambda':a.diversity_lambda,
        'maximum_overlap':a.maximum_overlap,
        'fallback_policy':'when threshold cannot yield k routes, relax it deterministically; write explicit flag',
        'elapsed_seconds':round(elapsed,4),
        'warnings':[
            'Economic cost is route length proxy only; no fuel/fees/toll dataset was supplied.',
            'Comfort is a road structure proxy, not independently validated passenger comfort.',
            'Scenario weights are preset heuristic weights, NOT optimized on the last 3 days.',
            'No future true selected route read. Module6 must evaluate generalization separately.',
            'Minmax scores are within-OD and should not be compared as absolute values between different ODs.',
            'When fewer than 3 natural routes exist, output only available routes; denominator includes all OD.',
            'Route diversity threshold may be relaxed to preserve available Top3; relaxation is audited.',
            'Module4 supervised training used observed-route augmentation; inference candidate sets differ.',
        ],
    }
    atomic_json(audit,audit_path/'08_模块五_完整性盲测与来源审计.json')
    log(f'模块五完成：请求OD={len(oa):,}，可评分OD={len(by_od):,}，'
        f'有Top3的OD={audit["top3_available_od"]:,}，三模式主推荐输出={len(result):,}行')
    log(f'输出目录：{audit_path}')
    return audit




# ============================================================================
# v2 场景需求扩展。严格只读取模块五已经盲推理完成的输出，永远不读取真实路线。
# 这里的类型标签是“同OD内道路结构目标的规则性判定”，不是UDPPM++原始软聚类概率，
# 也不是乘客出发前的真实需求。用户事后利用真实路线构造模拟需求，必须放到模块六。
# ============================================================================
DEMAND_VARIANTS = VARIANTS + ('objective_only',)
DEMAND_SCENARIOS = SCENARIOS
DEMAND_FILES = {
    'candidate': '09_候选路径_需求类型代理标签.csv.gz',
    'perod': '10_全部OD_需求满足与可达上限_逐方法.csv',
    'switch': '11_同OD_三场景推荐切换审计.csv',
    'summary': '12_需求类型匹配与场景切换汇总.csv',
    'direct': '13_直接道路属性规则_三场景Top3.csv.gz',
    'audit': '14_模块五_需求扩展与来源审计.json',
}


def demand_mask_and_eligibility(df: pd.DataFrame, args):
    """只用事先固定的道路代理规则判定“是否满足某个场景”。不构造真实标签。

    对不同OD分别计算相对经济/舒适代理分数；只在该OD至少存在2条可辨
    候选时才判断。双目标均衡使用双代理达到阈值且Pareto第一前沿。
    与原推荐打分有共同道路特征，因此只能用于需求规则符合性，不是独立效能证据。
    """
    econ = df['economic_proxy_score'].to_numpy(dtype=float)
    comf = df['comfort_proxy_score'].to_numpy(dtype=float)
    pr = df['pareto_rank'].to_numpy(dtype=int)
    finite = np.isfinite(econ).all() and np.isfinite(comf).all()
    if not finite or ((econ < -1e-8) | (econ > 1+1e-8)).any() or \
       ((comf < -1e-8) | (comf > 1+1e-8)).any():
        raise ValueError('候选需求标签不能计算：代理分数非法')
    if not np.isin(pr, np.arange(1,len(df)+1)).all():
        raise ValueError('候选Pareto等级异常')
    e_disc = len(df)>=2 and float(np.max(econ)-np.min(econ)) > 1e-9
    c_disc = len(df)>=2 and float(np.max(comf)-np.min(comf)) > 1e-9
    allowed = {
        'economic': bool(e_disc),
        'comfortable': bool(c_disc),
        'balanced': bool(e_disc and c_disc),
    }
    mask = {
        'economic': (econ >= args.economic_match_threshold-1e-12) & bool(e_disc),
        'comfortable': (comf >= args.comfort_match_threshold-1e-12) & bool(c_disc),
        'balanced': ((econ >= args.balanced_match_threshold-1e-12) &
                     (comf >= args.balanced_match_threshold-1e-12) &
                     (pr==1) & bool(e_disc and c_disc)),
    }
    return mask, allowed


def _load_and_validate_existing_m5(args, files):
    """核对旧模块五结果的冻结性，不触碰模块一留出的真实答案。"""
    p=args.work
    old_files = {
        'score':p/'01_全部候选_三模式多目标代理分数.csv.gz',
        'main':p/'02_三模式Top3推荐_主方法.csv.gz',
        'ablations':p/'03_三模式Top3推荐_对照与消融.csv.gz',
        'coverage':p/'05_全部测试OD_候选与推荐覆盖审计.csv',
        'pairs':p/'07_逐OD_自然候选两两有向边长度重叠.csv.gz',
        'manifest':p/'08_模块五_完整性盲测与来源审计.json',
    }
    for name,path in old_files.items():
        ensure_file(path,'原模块五'+name)
    manifest=jread(old_files['manifest'])
    expected_mode='PILOT_NOT_FOR_PAPER' if args.pilot else 'FULL_REAL'
    if manifest.get('mode')!=expected_mode:
        raise ValueError('原模块五结果的PILOT/FULL_REAL属性不匹配')
    if manifest.get('future_actual_hmm_route_read')!='NEVER' or \
       manifest.get('future_chosen_labels_read')!='NEVER':
        raise ValueError('原模块五审计未声明真实路线隔离')
    for key in ('model','predict_audit','predictions','static','raw','od_audit','m3_manifest','blind_od','graph'):
        old=manifest.get('input_sources',{}).get(key,{}).get('sha256')
        if not old:
            raise ValueError('原模块五来源审计缺少'+key+'SHA256')
        # 重新读取实际GraphML并非扩展实验必要条件；模块三图哈希已经在旧审计中冻结。
        if key == 'graph':
            continue
        if old!=digest(files[key]):
            raise ValueError(f'原模块五与当前{key}的源文件哈希不一致，不能混合两个批次')
    return old_files,manifest


def _validate_old_m5_tables(old_files, manifest, args):
    score=read_csv(old_files['score'])
    main=read_csv(old_files['main'])
    ablation=read_csv(old_files['ablations'])
    cov=read_csv(old_files['coverage'])
    pairs=read_csv(old_files['pairs'])
    for df,req,name in [
        (score,['trip_id',CID,'scenario','economic_proxy_score',
                'comfort_proxy_score','pareto_rank','total_len_km','ratio_to_shortest'], '原01候选分数'),
        (main,['trip_id',CID,'scenario','rank_in_scenario'],'原02主推荐'),
        (ablation,['trip_id',CID,'scenario','variant','rank_in_scenario'],'原03消融'),
        (cov,['trip_id','natural_count','top3_available'],'原05覆盖'),
        (pairs,['trip_id','candidate_id_a','candidate_id_b','pairwise_length_jaccard'],'原07道路共享')]:
        ensure_columns(df,req,name)
    if score[['trip_id',CID,'scenario']].duplicated().any():
        raise ValueError('原01有重复候选场景组合')
    if set(score.scenario.astype(str))!=set(SCENARIOS):
        raise ValueError('原01不是完整的三种场景')
    if len(score)!=int(manifest['candidate_rows'])*3:
        raise ValueError('原01行数与完整审计不一致')
    if len(main)!=int(manifest['main_top3_rows']) or \
       len(ablation)!=int(manifest['baseline_top3_rows']) or \
       len(pairs)!=int(manifest['length_jaccard_pair_rows']):
        raise ValueError('原模块五输出行数与审计不一致')
    if len(cov)!=int(manifest['all_test_od']) or cov.trip_id.duplicated().any():
        raise ValueError('原模块五OD覆盖审计缺失或重复')
    if set(main.scenario.astype(str))!=set(SCENARIOS) or set(ablation.scenario.astype(str))!=set(SCENARIOS):
        raise ValueError('主方法/消融缺少场景')
    if set(ablation.variant.astype(str))!=set(VARIANTS)-{'proposed'}:
        raise ValueError('消融类型与原模块五不匹配')
    if main[['trip_id','scenario','rank_in_scenario']].duplicated().any() or \
       ablation[['trip_id','scenario','variant','rank_in_scenario']].duplicated().any():
        raise ValueError('原推荐名次重复')
    if len(main)+len(ablation)!=int(manifest['main_top3_rows']+manifest['baseline_top3_rows']):
        raise ValueError('主方法与消融总行数异常')
    return score,main,ablation,cov,pairs


def extend_demand_scenarios(args, files, from_run=False):
    """根据已有正式盲推理结果增加场景规则、直接属性基线、场景切换分析。

    --phase extend: 不重新处理GraphML和模型，也不重写旧01~08。
    --phase run: 已正常生成01~08之后再调用本扩展。
    """
    start=time.perf_counter()
    old_files,manifest=_load_and_validate_existing_m5(args,files)
    audit_path=args.work/DEMAND_FILES['audit']
    if audit_path.exists() and not args.overwrite_extension and not from_run:
        raise FileExistsError('场景需求扩展已存在，避免误覆盖：'+str(audit_path)+
                              '；明确重做时添加 --overwrite-extension')
    score,main,ablation,coverage,pairs=_validate_old_m5_tables(old_files,manifest,args)
    # 所有候选的经济/舒适代理和Pareto与需求场景无关。
    base=score[score.scenario=='economic'].copy()
    bcheck=score[['trip_id',CID,'scenario','economic_proxy_score','comfort_proxy_score','pareto_rank']]
    consistency=bcheck.groupby(['trip_id',CID],sort=False).agg(
        ns=('scenario','nunique'),
        e_min=('economic_proxy_score','min'),e_max=('economic_proxy_score','max'),
        c_min=('comfort_proxy_score','min'),c_max=('comfort_proxy_score','max'),
        p_min=('pareto_rank','min'),p_max=('pareto_rank','max'))
    if not (consistency.ns.eq(3).all() and
            np.allclose(consistency.e_min,consistency.e_max,atol=1e-10) and
            np.allclose(consistency.c_min,consistency.c_max,atol=1e-10) and
            consistency.p_min.eq(consistency.p_max).all()):
        raise ValueError('三场景之间道路客观属性或Pareto标签不一致')
    if base[[CID]].duplicated().any() or len(base)!=int(manifest['candidate_rows']):
        raise ValueError('候选ID数量不符合原模块五审计')
    if set(base.trip_id)!=set(coverage.loc[coverage.natural_count>0,'trip_id']):
        raise ValueError('可达OD集合与原模块五审计不一致')
    ncount=base.groupby('trip_id').size()
    if not ncount.reindex(coverage.trip_id,fill_value=0).equals(
        coverage.set_index('trip_id').natural_count.astype(int).reindex(coverage.trip_id)):
        raise ValueError('新扩展候选计数与全OD覆盖不一致')
    # 将原模块五推荐结果改成同一个统一的variant索引。
    sel_main=main[['trip_id','scenario','rank_in_scenario',CID]].copy()
    sel_main['variant']='proposed'
    sel_old=ablation[['trip_id','scenario','rank_in_scenario',CID,'variant']].copy()
    select=pd.concat([sel_main,sel_old],ignore_index=True)
    if select[['trip_id','scenario','variant','rank_in_scenario']].duplicated().any():
        raise ValueError('原Top3排序索引重复')
    have_by_od={}
    for (tid,sc,vari),g in select.groupby(['trip_id','scenario','variant'],sort=False):
        g=g.sort_values('rank_in_scenario',kind='mergesort')
        if list(g.rank_in_scenario.astype(int))!=list(range(1,len(g)+1)):
            raise ValueError(f'原Top3名次不连续：{tid}/{sc}/{vari}')
        have_by_od[(str(tid),str(sc),str(vari))]=g[CID].astype(str).tolist()
    pairs_by_od={str(tid):g for tid,g in pairs.groupby('trip_id',sort=False)}
    demand_candidate=[]
    demand_perod=[]
    direct_rows=[]
    switch_rows=[]
    summaries=[]
    base_by_od={str(tid):g.sort_values(CID,kind='mergesort').reset_index(drop=True)
                for tid,g in base.groupby('trip_id',sort=False)}
    cov_count=coverage.set_index('trip_id').natural_count.astype(int).to_dict()
    if len(base_by_od)!=int(manifest['scored_od']):
        raise ValueError('扩展可达OD数量与原审计不一致')
    for i,(tid,natural_n) in enumerate(cov_count.items(),1):
        tid=str(tid)
        group=base_by_od.get(tid)
        if group is None:
            if int(natural_n)!=0:
                raise AssertionError('缺少自然候选但源审计标记可达')
            for sc in SCENARIOS:
                for vari in DEMAND_VARIANTS:
                    demand_perod.append({
                        'trip_id':tid,'scenario':sc,'variant':vari,
                        'candidate_count':0,'n_recommended':0,
                        'label_decidable':0,'has_matching_candidate':0,
                        'top1_matches_demand':0,'top3_any_matches_demand':0,
                        'top3_match_count':0,'top3_match_fraction':np.nan,
                        'mean_recommended_length_km':np.nan,
                        'recommendation_status':'no_natural_route',
                    })
            switch_rows.append({'trip_id':tid,'natural_count':0,
                                'three_scenario_top1_all_equal':np.nan,
                                'economic_vs_comfortable_top1_different':np.nan,
                                'economic_vs_balanced_top1_different':np.nan,
                                'comfortable_vs_balanced_top1_different':np.nan,
                                'economic_vs_comfortable_top3_jaccard':np.nan,
                                'economic_vs_balanced_top3_jaccard':np.nan,
                                'comfortable_vs_balanced_top3_jaccard':np.nan,
                                'status':'no_natural_route'})
            continue
        if len(group)!=int(natural_n):
            raise AssertionError('原05自然候选与原01行数不一致')
        ids=group[CID].astype(str).tolist()
        idx={c:j for j,c in enumerate(ids)}
        mask,eligible=demand_mask_and_eligibility(group,args)
        for j,cid in enumerate(ids):
            demand_candidate.append({
                'trip_id':tid,CID:cid,'candidate_count':len(ids),
                'economic_proxy_score':float(group.economic_proxy_score.iloc[j]),
                'comfort_proxy_score':float(group.comfort_proxy_score.iloc[j]),
                'pareto_rank':int(group.pareto_rank.iloc[j]),
                'economic_type_rule_match':int(mask['economic'][j]),
                'comfortable_type_rule_match':int(mask['comfortable'][j]),
                'balanced_type_rule_match':int(mask['balanced'][j]),
                'economic_type_decidable':int(eligible['economic']),
                'comfortable_type_decidable':int(eligible['comfortable']),
                'balanced_type_decidable':int(eligible['balanced']),
                'total_len_km':float(group.total_len_km.iloc[j]),
                'ratio_to_shortest':float(group.ratio_to_shortest.iloc[j]),
                'label_basis':'within_OD_static_road_proxy_NOT_UDPPM_membership_NOT_ground_truth',
            })
        # 已冻结的道路边对重叠：直接使用原07，不重新加载GraphML。
        ov=np.eye(len(ids),dtype=float)
        p=pairs_by_od.get(tid)
        if len(ids)>1 and (p is None or len(p)!=len(ids)*(len(ids)-1)//2):
            raise ValueError(f'道路两两重叠对数量不符，OD={tid}')
        if p is not None:
            for q in p.itertuples(index=False):
                ca=str(q.candidate_id_a);cb=str(q.candidate_id_b)
                if ca not in idx or cb not in idx or ca==cb:
                    raise ValueError(f'道路重叠中存在非法候选ID：{tid}')
                v=float(q.pairwise_length_jaccard)
                if not np.isfinite(v) or not -1e-9<=v<=1+1e-9:
                    raise ValueError(f'道路重叠率无效：{tid}')
                ov[idx[ca],idx[cb]]=ov[idx[cb],idx[ca]]=v
        objective={
            'economic':group.economic_proxy_score.to_numpy(float),
            'comfortable':group.comfort_proxy_score.to_numpy(float),
            'balanced':np.sqrt(group.economic_proxy_score.to_numpy(float)*
                               group.comfort_proxy_score.to_numpy(float)),
        }
        for sc in SCENARIOS:
            chosen,relaxed,effect=rank_topk(
                ids,objective[sc],ov,args.topk,args.diversity_lambda,
                args.maximum_overlap,diversity=True)
            for rank,(j,rel,(adjusted,max_overlap)) in enumerate(zip(chosen,relaxed,effect),start=1):
                direct_rows.append({
                    'trip_id':tid,'scenario':sc,'scenario_zh':SCENARIO_CHINESE[sc],
                    'variant':'objective_only','rank_in_scenario':rank,
                    CID:ids[j],'candidate_count':len(ids),
                    'scenario_objective_score':float(objective[sc][j]),
                    'diversity_adjusted_score_at_selection':float(adjusted),
                    'max_overlap_with_earlier_selected':float(max_overlap),
                    'overlap_threshold_relaxed_at_selection':int(rel),
                    'economic_proxy_score':float(group.economic_proxy_score.iloc[j]),
                    'comfort_proxy_score':float(group.comfort_proxy_score.iloc[j]),
                    'total_len_km':float(group.total_len_km.iloc[j]),
                    'ratio_to_shortest':float(group.ratio_to_shortest.iloc[j]),
                    'is_scenario_type_rule_match':int(mask[sc][j]),
                    'method_note':'static_objective_proxy_plus_same_diversity_no_UDPPM_no_PSL',
                })
            have_by_od[(tid,sc,'objective_only')]=[ids[j] for j in chosen]
            for vari in DEMAND_VARIANTS:
                selected=have_by_od.get((tid,sc,vari))
                if selected is None:
                    raise ValueError(f'原推荐结果缺少OD/场景/方法：{tid}/{sc}/{vari}')
                if len(selected)!=min(args.topk,len(ids)) or len(set(selected))!=len(selected):
                    raise ValueError(f'推荐记录数量错误或重复：{tid}/{sc}/{vari}')
                if any(cid not in idx for cid in selected):
                    raise ValueError(f'发现非自然候选路线：{tid}/{sc}/{vari}')
                vals=[int(mask[sc][idx[c]]) for c in selected]
                lengths=[float(group.total_len_km.iloc[idx[c]]) for c in selected]
                demand_perod.append({
                    'trip_id':tid,'scenario':sc,'variant':vari,
                    'candidate_count':len(ids),'n_recommended':len(selected),
                    'label_decidable':int(eligible[sc]),
                    'has_matching_candidate':int(np.any(mask[sc])),
                    'top1_matches_demand':vals[0],
                    'top3_any_matches_demand':int(any(vals)),
                    'top3_match_count':int(sum(vals)),
                    'top3_match_fraction':float(np.mean(vals)),
                    'mean_recommended_length_km':float(np.mean(lengths)),
                    'recommendation_status':'scored' if eligible[sc] else 'type_not_decidable',
                })
        first={sc:have_by_od[(tid,sc,'proposed')][0] for sc in SCENARIOS}
        sets={sc:set(have_by_od[(tid,sc,'proposed')]) for sc in SCENARIOS}
        def jj(a,b):
            return float(len(sets[a]&sets[b])/len(sets[a]|sets[b]))
        switch_rows.append({
            'trip_id':tid,'natural_count':len(ids),
            'three_scenario_top1_all_equal':int(len(set(first.values()))==1),
            'economic_vs_comfortable_top1_different':int(first['economic']!=first['comfortable']),
            'economic_vs_balanced_top1_different':int(first['economic']!=first['balanced']),
            'comfortable_vs_balanced_top1_different':int(first['comfortable']!=first['balanced']),
            'economic_vs_comfortable_top3_jaccard':jj('economic','comfortable'),
            'economic_vs_balanced_top3_jaccard':jj('economic','balanced'),
            'comfortable_vs_balanced_top3_jaccard':jj('comfortable','balanced'),
            'status':'scored',
        })
        if i%1000==0:
            log(f'需求扩展：完成{i:,}/{len(cov_count):,}条OD（只读取现有候选/推荐结果）')
    cand_df=pd.DataFrame(demand_candidate)
    perod_df=pd.DataFrame(demand_perod)
    switch_df=pd.DataFrame(switch_rows)
    direct_df=pd.DataFrame(direct_rows)
    expected_rows=len(cov_count)*len(SCENARIOS)*len(DEMAND_VARIANTS)
    if len(perod_df)!=expected_rows or len(switch_df)!=len(cov_count) or \
       len(cand_df)!=int(manifest['candidate_rows']) or \
       len(direct_df)!=int(manifest['main_top3_rows']):
        raise AssertionError('需求扩展数据数量未满足完整OD审计要求')
    if cand_df[['trip_id',CID]].duplicated().any() or \
       perod_df[['trip_id','scenario','variant']].duplicated().any() or \
       direct_df[['trip_id','scenario','rank_in_scenario']].duplicated().any():
        raise AssertionError('需求扩展结果包含重复键')
    for (sc,vari),d in perod_df.groupby(['scenario','variant'],sort=False):
        applicable=d[d.label_decidable==1]
        possible=applicable[applicable.has_matching_candidate==1]
        summaries.append({
            'scenario':sc,'scenario_zh':SCENARIO_CHINESE[sc],
            'variant':vari,'all_requested_od':len(cov_count),
            'scored_od':int((d.candidate_count>0).sum()),
            'type_decidable_od':len(applicable),
            'type_undecidable_od':len(d)-len(applicable),
            'oracle_type_available_od':len(possible),
            'oracle_type_available_rate_all_od':len(possible)/len(cov_count),
            'top1_rule_match_od':int(applicable.top1_matches_demand.sum()),
            'top3_any_rule_match_od':int(applicable.top3_any_matches_demand.sum()),
            'top1_rule_match_rate_decidable':(float(applicable.top1_matches_demand.mean()) if len(applicable) else np.nan),
            'top3_rule_match_rate_decidable':(float(applicable.top3_any_matches_demand.mean()) if len(applicable) else np.nan),
            'top3_rule_match_rate_oracle_possible':(float(possible.top3_any_matches_demand.mean()) if len(possible) else np.nan),
            'top1_rule_match_rate_all_od':float(d.top1_matches_demand.sum()/len(cov_count)),
            'top3_rule_match_rate_all_od':float(d.top3_any_matches_demand.sum()/len(cov_count)),
            'mean_fraction_top3_rule_matching_decidable':(float(applicable.top3_match_fraction.mean()) if len(applicable) else np.nan),
            'metric_type':'heuristic_within_OD_road_proxy_compliance_NOT_actual_preference_accuracy',
        })
    sums=pd.DataFrame(summaries)
    x=switch_df[switch_df.status=='scored']
    switch_summary={
        'all_od':len(switch_df),'scored_od':len(x),
        'top1_changes_economic_vs_comfortable':int(x.economic_vs_comfortable_top1_different.sum()),
        'top1_change_rate_scored_economic_vs_comfortable':float(x.economic_vs_comfortable_top1_different.mean()),
        'top1_changes_economic_vs_balanced':int(x.economic_vs_balanced_top1_different.sum()),
        'top1_change_rate_scored_economic_vs_balanced':float(x.economic_vs_balanced_top1_different.mean()),
        'top1_changes_comfortable_vs_balanced':int(x.comfortable_vs_balanced_top1_different.sum()),
        'top1_change_rate_scored_comfortable_vs_balanced':float(x.comfortable_vs_balanced_top1_different.mean()),
        'three_scenarios_top1_all_equal_od':int(x.three_scenario_top1_all_equal.sum()),
        'mean_top3_jaccard_economic_vs_comfortable':float(x.economic_vs_comfortable_top3_jaccard.mean()),
    }
    # 重要：新扩展从来不改变旧01~08，保留既有模块六的来源哈希。
    args.work.mkdir(parents=True,exist_ok=True)
    for key,df in [('candidate',cand_df),('perod',perod_df),('switch',switch_df),
                   ('summary',sums),('direct',direct_df)]:
        atomic_csv(df,args.work/DEMAND_FILES[key])
    audit={
        'version':'module5_demand_scenario_extension_v2.0',
        'created_at':datetime.now().isoformat(),
        'mode':'PILOT_NOT_FOR_PAPER' if args.pilot else 'FULL_REAL',
        'extension_only_unchanged_original_outputs_01_to_08':True,
        'module4_model_retrained':'NEVER',
        'module4_predictions_modified':'NEVER',
        'new_candidates_generated':'NEVER',
        'future_real_hmm_route_read':'NEVER',
        'future_chosen_or_vehicle_id_read':'NEVER',
        'scenario_request_source':'USER_SPECIFIED_BEFORE_RECOMMENDATION',
        'posthoc_simulated_demand_from_real_route':'NOT_IN_MODULE5_RESERVED_FOR_MODULE6',
        'matching_labels':'HEURISTIC_WITHIN_OD_ROAD_PROXY_NOT_UDPPM_SOFT_MEMBERSHIP',
        'type_thresholds':{
            'economic_score_min':args.economic_match_threshold,
            'comfortable_score_min':args.comfort_match_threshold,
            'balanced_each_score_min':args.balanced_match_threshold,
            'balanced_requires_pareto_front_1':True,
            'requires_objective_range_for_label_decidable':True,
        },
        'original_v1_manifest_sha256':digest(old_files['manifest']),
        'old_outputs_sha256':{key:digest(path) for key,path in old_files.items()},
        'new_outputs_sha256':{key:digest(args.work/DEMAND_FILES[key])
                              for key in ('candidate','perod','switch','summary','direct')},
        'counts':{
            'all_od':len(cov_count),'scored_od':int(manifest['scored_od']),
            'candidate_rows':len(cand_df),'perod_records':len(perod_df),
            'direct_rule_top3_rows':len(direct_df),
            'scenario_switch_od':len(switch_df),
        },
        'scenario_switch_descriptive':switch_summary,
        'direct_rule_scoring':{
            'economic':'within_OD_economy_score',
            'comfortable':'within_OD_comfort_score',
            'balanced':'geometric_mean_of_two_objective_benefits',
            'diversity':'same_graph_edge_Jaccard_as_original_module5',
        },
        'elapsed_seconds':round(time.perf_counter()-start,4),
        'warnings':[
            '规则一致率只说明符合预定义目标，不是与真实乘客需求的独立验证。',
            '同OD分数属于相对值，不能拿不同OD阈值比较绝对经济/舒适品质。',
            '规则标签与推荐评分共享道路属性，需以直接规则基线评估模型增量。',
            '原有UDPPM模型使用车辆历史画像，未因增加需求场景而自动变成无车辆身份模型。',
            '真实轨迹事后推断需求、与推荐匹配只允许在第六模块评价中做。',
            '原始第六模块不会自动读取新增09至14结果，需要单独升级第六模块。',
        ]
    }
    atomic_json(audit,audit_path)
    log('需求扩展完成：候选标签=%s，逐OD逐方法=%s，直接规则推荐=%s，场景切换OD=%s' %
        (f'{len(cand_df):,}',f'{len(perod_df):,}',f'{len(direct_df):,}',f'{len(switch_df):,}'))
    log('原模块五01~08保持不变。新增09~14输出：'+str(args.work))
    return audit


def recommend_one_trip(args, files):
    if not args.trip_id or not args.scenario:
        raise ValueError('--phase recommend必须指定 --trip-id 和 --scenario')
    old_files, manifest=_load_and_validate_existing_m5(args,files)
    top=read_csv(old_files['main'])
    rows=top.loc[(top.trip_id==str(args.trip_id)) &
                 (top.scenario==args.scenario)].sort_values('rank_in_scenario')
    if rows.empty:
        cov=read_csv(old_files['coverage'])
        if str(args.trip_id) not in set(cov.trip_id.astype(str)):
            raise ValueError('指定trip_id不是正式最后3天OD：'+str(args.trip_id))
        log('该OD没有自然候选路线，无法给出推荐（不会虚构路线）')
        return
    label_file=args.work/DEMAND_FILES['candidate']
    labels=read_csv(label_file) if label_file.exists() else None
    labelmap=labels.set_index(CID) if labels is not None else None
    col=args.scenario+'_type_rule_match'
    log(f'指定需求：{SCENARIO_CHINESE[args.scenario]}；OD={args.trip_id}；历史真值未读取')
    for r in rows.itertuples(index=False):
        cid=str(getattr(r,CID))
        match='未生成需求代理标签' if labelmap is None else str(int(labelmap.loc[cid,col]))
        log(f'Top{r.rank_in_scenario}  候选ID={cid}  长度={r.total_len_km:.3f}km  '
            f'经济代理={r.economic_proxy_score:.4f}  舒适代理={r.comfort_proxy_score:.4f}  '
            f'相对需求规则满足={match}')
    log('候选类型是同OD道路结构规则判定，不是实际驾驶员偏好准确率。')


# ==========================================================================
# v3 需求条件优先的完整批量推荐（直接复用模块四冻结模型经原模块五校验后的评分）
# 仅增加15-22新输出；旧01-14不覆盖。每个OD批量处理全部三场景。
# 本方法是查看原试验结果后增加的探索性方案，不把同批测试数据宣称新的盲测。
# ==========================================================================
DEMAND_V3_METHODS = ('proposed', 'no_diversity', 'no_profile', 'objective_only', 'utility_only')
DEMAND_V3_FILE_NAMES = {
    'scores': '15_需求优先_候选分数及类型标记.csv.gz',
    'main': '16_需求优先_三场景Top3_主方法.csv.gz',
    'ablations': '17_需求优先_三场景Top3_消融与规则对照.csv.gz',
    'perod': '18_需求优先_逐OD逐场景逐方法审计.csv',
    'coverage': '19_需求优先_全部OD场景需求可达上限.csv',
    'summary': '20_需求优先_五方法三场景汇总.csv',
    'switch': '21_需求优先_场景切换与目标冲突审计.csv',
    'audit': '22_需求优先_模块四冻结来源与研究边界审计.json',
}


def _v3_determine_demand(econ, comfort, pareto, args):
    """返回场景类型标签以及在该OD能否区分类型；指标是道路代理，不是乘客真偏好。"""
    n=len(econ)
    e_vary=n>1 and float(np.ptp(econ))>1e-9
    c_vary=n>1 and float(np.ptp(comfort))>1e-9
    eligible={
        'economic': np.asarray(econ>=args.economic_match_threshold-1e-12,dtype=bool) if e_vary else np.zeros(n,bool),
        'comfortable': np.asarray(comfort>=args.comfort_match_threshold-1e-12,dtype=bool) if c_vary else np.zeros(n,bool),
        'balanced': np.asarray((econ>=args.balanced_match_threshold-1e-12) &
                               (comfort>=args.balanced_match_threshold-1e-12) &
                               (pareto==1),dtype=bool) if (e_vary and c_vary) else np.zeros(n,bool),
    }
    decidable={'economic':bool(e_vary),'comfortable':bool(c_vary),
               'balanced':bool(e_vary and c_vary)}
    return eligible,decidable


def _v3_priorities(scenario, eligible, decidable, pareto):
    """硬需求仅在道路代理可判定时应用。balanced缺少同时满足者时退到第一Pareto前沿。"""
    n=len(pareto)
    if not decidable:
        return np.zeros(n,dtype=int)
    if scenario == 'balanced':
        return np.where(eligible,0,np.where(np.asarray(pareto)==1,1,2)).astype(int)
    return np.where(eligible,0,1).astype(int)


def _v3_rank(ids, scores, overlap, tier, args, diversity=True):
    """固定目标类型优先，类型内按冻结模型效用分与路径重叠贪心排序。

    与旧v1最大的差别：更低的需求层级优先级始终高于更高的模型概率；
    不能满足时明确fallback。多样性阈值在相同需求层内放宽，不造路线。
    """
    n=len(ids)
    if n==0:return []
    scores=np.asarray(scores,dtype=float)
    tier=np.asarray(tier,dtype=int)
    if len(scores)!=n or len(tier)!=n or np.isfinite(scores).sum()!=n:
        raise ValueError('需求优先推荐分数/类型优先级含非有限值或数量不一致')
    remaining=set(range(n))
    result=[]
    for _ in range(min(args.topk,n)):
        besttier=min(tier[j] for j in remaining)
        candidates=sorted(j for j in remaining if tier[j]==besttier)
        def sim_to_selected(j):
            return max((float(overlap[j,k]) for k in result),default=0.)
        within=[j for j in candidates if (not diversity or not result or
                                           sim_to_selected(j)<=args.maximum_overlap+1e-12)]
        relax=int(diversity and len(within)==0 and bool(result))
        opts=within if within else candidates
        def key(j):
            maxsim=sim_to_selected(j) if diversity else 0.
            adjusted=float(scores[j])-args.diversity_lambda*maxsim if diversity else float(scores[j])
            # 先最大调整得分、再最小重合、最后稳定按字符串ID
            return (-adjusted, maxsim, str(ids[j]))
        chosen=min(opts,key=key)
        maxsim=sim_to_selected(chosen)
        adjusted=float(scores[chosen])-args.diversity_lambda*maxsim if diversity else float(scores[chosen])
        result.append(chosen)
        remaining.remove(chosen)
        # 返回：索引、代理需求层级、放宽重叠阈值、该次最大重叠、最终得分
        if _==0:out=[]
        out.append((chosen,int(tier[chosen]),relax,maxsim,adjusted))
    return out


def _v3_score_arrays(g, scenario):
    """分数源自原模块五已审计候选分数和模块四冻结六模型概率。"""
    econ=g.economic_proxy_score.to_numpy(float)
    comfort=g.comfort_proxy_score.to_numpy(float)
    pp=g.model_PSL_Length_probability.to_numpy(float)
    psl=one_dimension_good(pp)
    primary={
        'economic':'model_UDPPM_Econ_probability',
        'comfortable':'model_UDPPM_Comf_probability',
        'balanced':'model_UDPPM_Dual_probability',
    }[scenario]
    modelnorm=one_dimension_good(g[primary].to_numpy(float))
    # 旧主模型的权重严格不改：来自第四步已训练概率 + 第五步既定目标权重。
    proposed=g.prediversity_score.to_numpy(float)
    w=SCORE_WEIGHTS[scenario]
    no_profile=(w['specialized']+w['dual'])*psl + w['economy_proxy']*econ + w['comfort_proxy']*comfort
    if scenario=='economic': direct=econ
    elif scenario=='comfortable': direct=comfort
    else: direct=np.sqrt(np.maximum(0.,econ)*np.maximum(0.,comfort))
    return {
        'proposed':proposed,
        'no_diversity':proposed,
        'no_profile':no_profile,
        'objective_only':direct,
        # utility_only恢复模块四原概率直接排序，不施加需求类型硬约束也不去重。
        'utility_only':modelnorm,
    }


def _v3_get_overlap(tid, ids, pair_lookup):
    n=len(ids)
    mat=np.zeros((n,n),dtype=float)
    cmap={str(cid):i for i,cid in enumerate(ids)}
    pair_rows=pair_lookup.get(str(tid),[])
    if len(pair_rows)!=n*(n-1)//2:
        raise ValueError(f'{tid}: 长度加权道路共享记录数量错误：{len(pair_rows)}，预期{n*(n-1)//2}')
    seen=set()
    for a,b,v in pair_rows:
        if a not in cmap or b not in cmap or a==b:
            raise ValueError(f'{tid}: 有向边Jaccard包含外部候选或同候选对')
        ia,ib=sorted((cmap[a],cmap[b]))
        if (ia,ib) in seen or not math.isfinite(v) or not -1e-9<=v<=1+1e-9:
            raise ValueError(f'{tid}: Jaccard记录重复/非法')
        seen.add((ia,ib))
        mat[ia,ib]=mat[ib,ia]=float(v)
    return mat


def _v3_prepare_source(a, files):
    old_files,manifest=_load_and_validate_existing_m5(a,files)
    if manifest.get('module4_frozen_model_sha256')!=digest(files['model']):
        raise ValueError('原模块五记录的模块四冻结模型与当前文件不同')
    score,old_main,old_ablations,cov,pairs=_validate_old_m5_tables(old_files,manifest,a)
    # 除了抽取已冻结的模块四评分外，不读取任何最后3天真实路线、司机/车辆身份。
    pair_lookup={}
    for r in pairs[['trip_id','candidate_id_a','candidate_id_b','pairwise_length_jaccard']].itertuples(index=False,name=None):
        t,ca,cb,v=r
        pair_lookup.setdefault(str(t),[]).append((str(ca),str(cb),float(v)))
    base=score[score.scenario=='economic'].copy()
    if len(base)!=int(manifest['candidate_rows']) or base[CID].duplicated().any():
        raise ValueError('原候选评分原始表重复或缺少候选')
    # 三个场景客观代理特征必须相同，仅相应模型评分与场景分数不同。
    keep=['trip_id',CID,'economic_proxy_score','comfort_proxy_score','pareto_rank',
          'total_len_km','ratio_to_shortest','prediversity_score',
          'model_UDPPM_Econ_probability','model_UDPPM_Comf_probability',
          'model_UDPPM_Dual_probability','model_PSL_Length_probability']
    ensure_columns(base,keep,'原模块五冻结评分')
    for sc in SCENARIOS:
        chunk=score[score.scenario==sc][['trip_id',CID,'prediversity_score']].copy()
        if len(chunk)!=len(base) or chunk[CID].duplicated().any() or set(chunk[CID])!=set(base[CID]):
            raise ValueError(f'{sc}: 原候选场景评分集合不完整')
        base=base.merge(chunk[[CID,'prediversity_score']].rename(
            columns={'prediversity_score':f'score_{sc}'}),on=CID,how='left',validate='one_to_one')
    if base.filter(like='score_').isna().any().any():
        raise ValueError('原三场景分数关联失败')
    for col in ['economic_proxy_score','comfort_proxy_score','total_len_km',
                'ratio_to_shortest','model_UDPPM_Econ_probability','model_UDPPM_Comf_probability',
                'model_UDPPM_Dual_probability','model_PSL_Length_probability']:
        x=pd.to_numeric(base[col],errors='coerce').to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f'{col}存在缺失/Inf，不得填充')
    base=base.sort_values(['trip_id',CID],kind='mergesort').reset_index(drop=True)
    return base,cov,pair_lookup,old_files,manifest


def v3_batch_recommend(a,files):
    t0=time.perf_counter()
    base,cov,pair_lookup,old_files,manifest=_v3_prepare_source(a,files)
    outputs={k:a.work/v for k,v in DEMAND_V3_FILE_NAMES.items()}
    existed=[str(p) for p in outputs.values() if p.exists()]
    if existed and not a.overwrite_demand_v3:
        raise FileExistsError('15-22需求优先实验已存在，禁止覆盖。明确重做时加 --overwrite-demand-v3。示例：'+existed[0])
    # 在全体4469 OD上按三种需求、五组方法输出；没有可选路线的OD也计入分母。
    score_rows=[];main_rows=[];ablation_rows=[];perod_rows=[];reach_rows=[];change_rows=[]
    ngroups=0
    for ngroups,(tid,g) in enumerate(base.groupby('trip_id',sort=True),1):
        tid=str(tid)
        ids=g[CID].astype(str).tolist()
        econ=g.economic_proxy_score.to_numpy(float)
        comfort=g.comfort_proxy_score.to_numpy(float)
        pareto=g.pareto_rank.to_numpy(int)
        eligible,decidable=_v3_determine_demand(econ,comfort,pareto,a)
        overlap=_v3_get_overlap(tid,ids,pair_lookup)
        # 经济/舒适真正发生权衡的OD，单独报告，不能人为强制路线切换。
        eb=int(np.argmax(econ));cb=int(np.argmax(comfort))
        different_best=ids[eb]!=ids[cb]
        conflict=bool(different_best and
                      econ[eb]-econ[cb]>=a.conflict_gap and
                      comfort[cb]-comfort[eb]>=a.conflict_gap)
        top1={};top3={}
        for sc in SCENARIOS:
            # 原模块五模型对sc的冻结评分（不是当前车辆真实偏好标签）。
            g2=g.copy()
            g2['prediversity_score']=g2[f'score_{sc}']
            scores=_v3_score_arrays(g2,sc)
            tier=_v3_priorities(sc,eligible[sc],decidable[sc],pareto)
            possible=int(eligible[sc].sum())
            for j,cid in enumerate(ids):
                score_rows.append({'trip_id':tid,CID:cid,'scenario':sc,
                                   'economic_proxy_score':econ[j],'comfort_proxy_score':comfort[j],
                                   'pareto_rank':int(pareto[j]),
                                   'rule_matches_demand':int(eligible[sc][j]) if decidable[sc] else None,
                                   'demand_label_decidable':int(decidable[sc]),
                                   'demand_priority_tier':int(tier[j]),
                                   'frozen_model_demand_score':float(scores['proposed'][j]),
                                   'no_profile_score':float(scores['no_profile'][j]),
                                   'objective_only_score':float(scores['objective_only'][j]),
                                   'model_utility_only_score':float(scores['utility_only'][j])})
            for variant in DEMAND_V3_METHODS:
                # utility_only忠实复原模块四模型概率Top-3：不加强需求约束或多样性。
                raw_only=variant=='utility_only'
                used_tier=np.zeros(len(g),dtype=int) if raw_only else tier
                ranked=_v3_rank(ids,scores[variant],overlap,used_tier,a,
                                diversity=(variant in ('proposed','no_profile','objective_only')))
                indices=[r[0] for r in ranked]
                overlaps=[float(overlap[i,j]) for pos,i in enumerate(indices)
                          for j in indices[pos+1:]]
                eligible_count=sum(int(eligible[sc][i]) for i in indices) if decidable[sc] else None
                row={'trip_id':tid,'scenario':sc,'variant':variant,
                     'candidate_count':len(g),'n_recommended':len(indices),
                     'demand_label_decidable':int(decidable[sc]),
                     'eligible_n_in_candidates':possible,
                     'has_eligible_candidate':int(possible>0),
                     'top1_rule_match':int(eligible[sc][indices[0]]) if indices and decidable[sc] else None,
                     'top3_any_rule_match':int(eligible_count>0) if decidable[sc] else None,
                     'top3_rule_match_fraction':eligible_count/len(indices) if indices and decidable[sc] else None,
                     'n_demand_fallback':sum(int(tier[i]>0) for i in indices) if decidable[sc] else None,
                     'n_overlap_relaxed':sum(r[2] for r in ranked),
                     'mean_length_km':float(np.mean(g.total_len_km.to_numpy(float)[indices])),
                     'mean_ratio_to_shortest':float(np.mean(g.ratio_to_shortest.to_numpy(float)[indices])),
                     'mean_economic_proxy':float(np.mean(econ[indices])),
                     'mean_comfort_proxy':float(np.mean(comfort[indices])),
                     'mean_pairwise_overlap':float(np.mean(overlaps)) if overlaps else None,
                     'economic_comfort_best_conflict':int(conflict),
                     'different_objective_optimum':int(different_best),
                     }
                perod_rows.append(row)
                for rank,entry in enumerate(ranked,1):
                    j,level,relaxed,maxsim,adjusted=entry
                    rec={'trip_id':tid,'scenario':sc,'variant':variant,'rank_in_scenario':rank,
                         CID:ids[j],'candidate_count':len(g),
                         'total_len_km':float(g.total_len_km.iloc[j]),
                         'ratio_to_shortest':float(g.ratio_to_shortest.iloc[j]),
                         'economic_proxy_score':float(econ[j]),
                         'comfort_proxy_score':float(comfort[j]),
                         'pareto_rank':int(pareto[j]),
                         'demand_label_decidable':int(decidable[sc]),
                         'rule_matches_demand':int(eligible[sc][j]) if decidable[sc] else None,
                         'demand_priority_tier':level,
                         'original_score':float(scores[variant][j]),
                         'diversity_adjusted_score_at_selection':adjusted,
                         'max_overlap_with_earlier_selected':maxsim,
                         'overlap_threshold_relaxed_at_selection':int(relaxed)}
                    if variant=='proposed':main_rows.append(rec)
                    else:ablation_rows.append(rec)
                if variant=='proposed':
                    top1[sc]=ids[indices[0]] if indices else None
                    top3[sc]=set(ids[i] for i in indices)
            reach_rows.append({'trip_id':tid,'scenario':sc,'natural_count':len(g),
                               'type_decidable':int(decidable[sc]),
                               'eligible_n':possible,
                               'oracle_type_available':int(possible>0),
                               'oracle_top3_max_matching':min(3,possible),
                               'economic_comfort_best_conflict':int(conflict)})
        change_rows.append({'trip_id':tid,'natural_count':len(g),
                            'economic_top1_id':top1['economic'],
                            'comfortable_top1_id':top1['comfortable'],
                            'balanced_top1_id':top1['balanced'],
                            'top1_economic_vs_comfortable_changed':int(top1['economic']!=top1['comfortable']),
                            'top1_economic_vs_balanced_changed':int(top1['economic']!=top1['balanced']),
                            'top1_comfortable_vs_balanced_changed':int(top1['comfortable']!=top1['balanced']),
                            'top1_three_scenarios_all_same':int(len(set(top1.values()))==1),
                            'top3_economic_vs_comfortable_jaccard':(len(top3['economic']&top3['comfortable']) /
                                                                      len(top3['economic']|top3['comfortable']))
                                                                      if top3['economic']|top3['comfortable'] else None,
                            'different_objective_optimum':int(different_best),
                            'economic_comfort_best_conflict':int(conflict)})
        if ngroups%500==0:log(f'需求优先全量推荐：{ngroups:,}个可用OD / {len(base.trip_id.unique()):,}')
    coverage=cov[['trip_id','natural_count']].copy()
    coverage['trip_id']=coverage.trip_id.astype(str)
    if coverage.trip_id.duplicated().any():
        raise ValueError('总OD覆盖表存在重复OD')
    known=set(base.trip_id.astype(str))
    for rr in coverage.itertuples(index=False):
        if str(rr.trip_id) not in known:
            if int(rr.natural_count)!=0:
                raise ValueError('覆盖表中有未处理且自然候选数非零的OD')
            for sc in SCENARIOS:
                reach_rows.append({'trip_id':str(rr.trip_id),'scenario':sc,'natural_count':0,
                                   'type_decidable':0,'eligible_n':0,'oracle_type_available':0,
                                   'oracle_top3_max_matching':0,'economic_comfort_best_conflict':0})
            change_rows.append({'trip_id':str(rr.trip_id),'natural_count':0,
                                'economic_top1_id':None,'comfortable_top1_id':None,
                                'balanced_top1_id':None,
                                'top1_economic_vs_comfortable_changed':0,
                                'top1_economic_vs_balanced_changed':0,
                                'top1_comfortable_vs_balanced_changed':0,
                                'top1_three_scenarios_all_same':0,
                                'top3_economic_vs_comfortable_jaccard':None,
                                'different_objective_optimum':0,
                                'economic_comfort_best_conflict':0})
    scores_df=pd.DataFrame(score_rows); main_df=pd.DataFrame(main_rows)
    ablations_df=pd.DataFrame(ablation_rows);perod_df=pd.DataFrame(perod_rows)
    reach_df=pd.DataFrame(reach_rows);switch_df=pd.DataFrame(change_rows)
    if len(reach_df)!=len(coverage)*3 or len(switch_df)!=len(coverage):
        raise AssertionError('需求可达审计/切换审计未完整覆盖全部OD')
    if len(perod_df)!=len(known)*3*len(DEMAND_V3_METHODS):
        raise AssertionError('需求优先逐OD消融数不一致')
    if len(main_df)!=3*sum(min(a.topk,int(x)) for x in cov.natural_count):
        raise AssertionError('三场景Top3数量与真实自然候选数不一致')
    if main_df[['trip_id','scenario','rank_in_scenario']].duplicated().any():
        raise AssertionError('新主方法Top3排名重复')
    # 独立汇总：评价自身规则符合率时，需明确其非独立标签性质。
    summary_rows=[]
    for (sc,method),g in perod_df.groupby(['scenario','variant'],sort=True):
        d=g[g.demand_label_decidable.eq(1)]
        p=d[d.has_eligible_candidate.eq(1)]
        summary_rows.append({
            'scenario':sc,'variant':method,'all_test_od':len(coverage),
            'scored_od':len(g),'top3_available_od':int((g.n_recommended>=3).sum()),
            'type_decidable_od':len(d),'oracle_type_available_od':len(p),
            'top1_rule_match_rate_decidable':float(d.top1_rule_match.mean()) if len(d) else None,
            'top3_any_rule_match_rate_decidable':float(d.top3_any_rule_match.mean()) if len(d) else None,
            'top1_rule_match_rate_oracle_possible':float(p.top1_rule_match.mean()) if len(p) else None,
            'top3_rule_match_rate_oracle_possible':float(p.top3_any_rule_match.mean()) if len(p) else None,
            'top3_rule_fraction_decidable':float(d.top3_rule_match_fraction.mean()) if len(d) else None,
            'mean_length_km':float(g.mean_length_km.mean()),
            'mean_detour_ratio':float(g.mean_ratio_to_shortest.mean()),
            'mean_economy_proxy':float(g.mean_economic_proxy.mean()),
            'mean_comfort_proxy':float(g.mean_comfort_proxy.mean()),
            'mean_pairwise_road_jaccard':float(g.mean_pairwise_overlap.mean()),
            'od_with_overlap_relaxation':int((g.n_overlap_relaxed>0).sum()),
            'n_conflict_od':int(g.economic_comfort_best_conflict.sum()),
        })
    summary_df=pd.DataFrame(summary_rows)
    # 输出目录只能新增15~22；覆盖需用户明确参数；无分母丢失。
    a.work.mkdir(parents=True,exist_ok=True)
    for key,frame in [('scores',scores_df),('main',main_df),('ablations',ablations_df),
                      ('perod',perod_df),('coverage',reach_df),('summary',summary_df),
                      ('switch',switch_df)]:
        atomic_csv(frame,outputs[key])
    audit={
        'version':'module5_demand_first_exploratory_v3.0',
        'created_at':datetime.now().isoformat(),
        'full_real_source_mode':manifest.get('mode'),
        'posthoc_exploratory_method':True,
        'model_4_retrained':'NEVER',
        'historical_vehicle_8d_in_frozen_m4':'YES_IN_UDPPM_MODELS_NOT_STABLE_DRIVER_IDENTITY',
        'user_demand_input':'ECONOMIC_COMFORTABLE_BALANCED_PRESELECTED',
        'true_last3day_route_read':'NEVER',
        'true_last3day_demand_inferred':'NEVER',
        'natural_candidates_generated':'NEVER',
        'original_outputs_01_to_14_modified':'NEVER',
        'n_all_test_od':len(coverage),'n_scored_od':len(known),'n_natural_candidates':len(base),
        'n_main_recommendation_rows':len(main_df),
        'n_method_od_rows':len(perod_df),
        'n_scenario_change_rows':len(switch_df),
        'n_economic_comfort_strong_conflict_od':int(switch_df.economic_comfort_best_conflict.sum()),
        'scenario_type_proxy_rule':{
            'economic_econ_score_min':a.economic_match_threshold,
            'comfortable_comfort_score_min':a.comfort_match_threshold,
            'balanced_each_score_min':a.balanced_match_threshold,
            'balanced_requires_pareto_front':True,
            'requires_within_OD_objective_range':True,
            'scenarios_are_mutually_exclusive':'NO',
            'type_is_soft_cluster_posterior':'NO',
        },
        'demand_first_ranking':'candidate class tier before module4 frozen model score; within tier penalize overlap',
        'overlap_lambda':a.diversity_lambda,
        'max_overlap':a.maximum_overlap,
        'strong_conflict_gap':a.conflict_gap,
        'five_methods':list(DEMAND_V3_METHODS),
        'source_hashes':{'m4_model_sha256':digest(files['model']),
                         'm4_predictions_sha256':digest(files['predictions']),
                         'm5_old_manifest_sha256':digest(old_files['manifest']),
                         'm5_frozen_scores_sha256':digest(old_files['score']),
                         'm5_frozen_pairs_sha256':digest(old_files['pairs'])},
        'output_sha256':{k:digest(p) for k,p in outputs.items() if k!='audit'},
        'elapsed_seconds':round(time.perf_counter()-t0,3),
        'methodological_warnings':[
            'This is a posthoc exploratory method designed after inspecting earlier test results. Do not call these same test ODs untouched independent validation.',
            'Demand labels are derived from the same road structure proxy objectives used in ranking; matching rates are algorithm rule compliance, not independently measured preference accuracy.',
            'UDPPM scores originate from frozen vehicle historical 8D features; do not claim fully driver-independent model.',
            'A route may satisfy both economic and comfortable demands. Do not force scene switching.',
            'One-OD normalized objective scores are relative, not absolute comfort or cost.',
            'No new real-world medical emergency/hazardous cargo road safety data are provided.',
        ],
    }
    atomic_json(audit,outputs['audit'])
    log('需求优先批量推荐完成：全部OD=%s，可用OD=%s，主方法Top3记录=%s，方法OD记录=%s' %
        (len(coverage),len(known),len(main_df),len(perod_df)))
    log('需求场景Top-1是否切换和规则匹配结果，请直接查看：'+str(outputs['summary']))
    log('旧结果01-14不变；所有新结果15-22位于：'+str(a.work))


def parse_args():
    p=argparse.ArgumentParser(description='模块五v3：继承模块四冻结模型，4,469OD批量需求优先Top3与消融')
    p.add_argument('--phase',choices=['check','run','extend','recommend','demand-batch'],default='check')
    p.add_argument('--m1',type=Path,default=M1)
    p.add_argument('--m3',type=Path,default=M3)
    p.add_argument('--m4',type=Path,default=M4)
    p.add_argument('--graphml',type=Path,default=GRAPH)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--pilot-od',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--topk',type=int,default=3)
    p.add_argument('--diversity-lambda',type=float,default=.18)
    p.add_argument('--maximum-overlap',type=float,default=.80)
    p.add_argument('--verify-graph-in-check',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--overwrite-extension',action='store_true')
    p.add_argument('--overwrite-demand-v3',action='store_true',
                   help='显式覆盖新15-22号需求优先结果；不修改旧01-14')
    p.add_argument('--economic-match-threshold',type=float,default=.80)
    p.add_argument('--comfort-match-threshold',type=float,default=.80)
    p.add_argument('--balanced-match-threshold',type=float,default=.50)
    p.add_argument('--conflict-gap',type=float,default=.15,
                   help='两目标代表路线互相损失达到该差值，才称较强冲突（仅诊断不改变评分）')
    p.add_argument('--trip-id',type=str)
    p.add_argument('--scenario',choices=SCENARIOS)
    return p.parse_args()


def main():
    a=parse_args()
    if a.topk!=3:raise ValueError('本文Top3实验只支持 --topk 3')
    if a.pilot_od<1:raise ValueError('pilot-od必须>=1')
    for field in ('economic_match_threshold','comfort_match_threshold',
                  'balanced_match_threshold','conflict_gap','maximum_overlap','diversity_lambda'):
        val=getattr(a,field)
        if not 0<=val<=1:raise ValueError(field+'必须在[0,1]')
    if any(abs(sum(w.values())-1.)>1e-10 for w in SCORE_WEIGHTS.values()):
        raise ValueError('三场景权重不等于1')
    if abs(sum(COMFORT_COST_WEIGHTS.values())-1)>1e-10:
        raise ValueError('舒适代理权重不等于1')
    a.work=a.out/(f'pilot_od{a.pilot_od}_seed{a.seed}' if a.pilot else 'formal_K10')
    files,model,pred_audit,m3,m1=check_files(a)
    if a.phase=='check':
        log('模块五v3来源检查完成；不读取真实路线、不训练模型、不改旧文件')
    elif a.phase=='demand-batch':
        v3_batch_recommend(a,files)
    elif a.phase=='extend':
        extend_demand_scenarios(a,files)
    elif a.phase=='recommend':
        recommend_one_trip(a,files)
    elif a.phase=='run':
        output=a.work/'08_模块五_完整性盲测与来源审计.json'
        if output.exists() and not a.overwrite:
            raise FileExistsError(f'旧模块五已完成：{output}；建议 --phase demand-batch 复用已有成果')
        run(a,files,model,pred_audit,m3)
        extend_demand_scenarios(a,files,from_run=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        log('用户中断；未修改模块一至四或已冻结旧实验文件')
        raise SystemExit(130)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(f'模块五停止：{type(e).__name__}: {e}')
