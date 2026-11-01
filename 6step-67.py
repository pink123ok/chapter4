# -*- coding: utf-8 -*-
"""
第四章模块六补充：6step-67.py｜冻结 UDPPM++ 效用模型的历史画像输入敏感性
================================================================================
目的：在完全冻结模型参数、候选OD、道路特征、标准化器、70/30评分和Top-3多样性
规则的情况下，只把每个OD的历史8维软画像替换成前28天冻结的群体参考画像。

数据完全来源于真实模块二、三、四、五的保存文件；不重训、不生成新路线、不读后三天真值。
1) 按模块四实际 BASE10+PathSize+画像×道路属性公式重建模型预测；
2) 对照模块四已冻结的逐候选预测概率，未复现成功就停止；
3) 对照模块五v4已冻结的原始Top3候选ID，未复现成功就停止；
4) 相同模型系数与相同评分规则下，仅替换8维画像，再算概率/Top-3；
5) 统计Top-1/第3名/Top-3集合与排序变化、场景质量、历史记录分层及强冲突OD。

重要：本程序只做**同一冻结模型对画像输入的敏感性分析**，不是严格有无画像训练消融，
也不代表真实乘客需求准确率或独立泛化验证。测试数据已多次查看，结果属于探索性。

Windows用法：
    python.exe .\\6step-67.py --phase check
    python.exe .\\6step-67.py --phase analyze
    python.exe .\\6step-67.py --phase self-test
    python.exe .\\6step-67.py --phase analyze --no-plot

依赖：pip install numpy pandas matplotlib
编码：utf-8-sig，CSV输出Excel/WPS兼容。
所有输出写入单独目录，绝不覆盖原来模块一至六文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = Path(r'C:\study\轨迹数据预处理')
ROOT = BASE / '曼谷GPS_正式清洗_202412' / 'FULL_全量' / '第四章_28天训练_3天独立测试'
CID = 'candidate_id_TRACE_NOT_FEATURE'
SCENARIOS = ('economic', 'comfortable', 'balanced')
SC_MODEL = {
    'economic': 'UDPPM_Econ_PSL_L2',
    'comfortable': 'UDPPM_Comf_PSL_L2',
    'balanced': 'UDPPM_Dual_PSL_L2',
}
MODEL_BLOCKS = {
    'UDPPM_Econ_PSL_L2': ['base', 'ps_length', 'econ'],
    'UDPPM_Comf_PSL_L2': ['base', 'ps_length', 'comf'],
    'UDPPM_Dual_PSL_L2': ['base', 'ps_length', 'econ', 'comf'],
}
BASE10 = [
    'total_len_km', 'ratio_to_shortest', 'ratio_motorway', 'ratio_major', 'ratio_local',
    'turns_per_km', 'segment_density_seg_per_km', 'oneway_ratio',
    'lanes_mean_w', 'maxspeed_mean_known_kmh',
]
SCALE_COLS = BASE10 + ['log_ps_count', 'log_ps_length']
ECON_ROUTE = ['total_len_km', 'ratio_to_shortest', 'ratio_motorway', 'ratio_major', 'ratio_local']
COMF_ROUTE = ['turns_per_km', 'segment_density_seg_per_km', 'ratio_local', 'ratio_motorway', 'oneway_ratio']
PROFILE = [f'history_{d}_{j}' for d in ('econ', 'comf') for j in range(4)]
GOAL_COL = {'economic': 'economic_proxy_score', 'comfortable': 'comfort_proxy_score'}
FORBIDDEN = {
    'chosen', 'VehicleID', 'vehicle_id', 'is_real_added', 'real_added_TRACE_NOT_FEATURE',
    'observed_route', 'real_edge_seq', 'observed_edges', 'actual_path', 'label',
}


def log(msg: str):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def file_ok(p: Path, what: str):
    if not p.is_file() or p.stat().st_size == 0:
        raise FileNotFoundError(f'找不到{what}：{p}\n请确认第四模块fit/predict、第五模块demand-soft已完整运行。')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def readj(p: Path):
    file_ok(p, 'JSON')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def readc(p: Path, **kwargs):
    file_ok(p, 'CSV')
    return pd.read_csv(p, encoding='utf-8-sig', low_memory=False,
                       dtype={'trip_id': 'string', CID: 'string'}, **kwargs)


def need(df, cols, tag):
    missing = [c for c in cols if c not in df]
    if missing:
        raise ValueError(f'{tag}缺少字段 {missing}，不能猜测或使用模拟值。')


def nparr(df, name, *, finite=True):
    v = pd.to_numeric(df[name], errors='coerce').to_numpy(float)
    if finite and not np.isfinite(v).all():
        raise ValueError(f'字段{name}包含缺失/非数值/Inf，禁止用虚拟值补齐')
    return v


def numeric_score(x):
    """模块五 one_dimension_good 的严格复制：候选集合内 min-max，无变化时均置0.5。"""
    x = np.asarray(x, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('minmax输入为空或非有限数')
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.full(len(x), .5, dtype=float)
    return (x - lo) / (hi - lo)


def scale_static(df, params):
    """与第四模块 CandidateScaler.transform 完全相同。"""
    if [p.get('name') for p in params] != SCALE_COLS:
        raise ValueError('模块四冻结标准化字段顺序与当前源代码不一致')
    xx = []
    for p in params:
        col = p['name']
        need(df, [col], '模块三静态特征')
        v = pd.to_numeric(df[col], errors='coerce').to_numpy(float)
        v = np.where(np.isfinite(v), v, float(p['median']))
        std = float(p['std'])
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f'{col}标准差不合法')
        xx.append(np.clip((v - float(p['mean'])) / std, -8, 8))
    return np.stack(xx, axis=1)


def check_profiles(q: np.ndarray, label: str):
    if q.ndim != 2 or q.shape[1] != 8:
        raise ValueError(f'{label}不是8维历史软画像')
    if not np.isfinite(q).all() or (q < -1e-8).any() or (q > 1 + 1e-8).any():
        raise ValueError(f'{label}存在非法画像概率')
    if not np.allclose(q[:, :4].sum(axis=1), 1, atol=1e-5) or not np.allclose(q[:, 4:].sum(axis=1), 1, atol=1e-5):
        raise ValueError(f'{label}经济/舒适各4维概率之和必须为1')


def x_for_model(z: np.ndarray, q: np.ndarray, blocks: list[str]):
    """严格复刻第四模块 features_for_models: (q_k - .25)*z_attr，原始块顺序不改。"""
    if len(q) != len(z):
        raise ValueError('画像/道路候选行数不一致')
    check_profiles(q, '模型输入')
    index = {key: i for i, key in enumerate(SCALE_COLS)}
    pieces = {'base': z[:, :10], 'ps_length': z[:, [index['log_ps_length']]]}
    for d, attrs, start in [('econ', ECON_ROUTE, 0), ('comf', COMF_ROUTE, 4)]:
        pieces[d] = np.column_stack([(q[:, start+k] - .25) * z[:, index[c]]
                                     for k in range(4) for c in attrs])
    if any(b not in pieces for b in blocks):
        raise ValueError(f'模型特征块非法：{blocks}')
    return np.hstack([pieces[b] for b in blocks])


def group_softmax(utilities, sizes):
    utilities = np.asarray(utilities, dtype=float)
    sizes = np.asarray(sizes, dtype=int)
    if sizes.sum() != len(utilities) or (sizes <= 0).any():
        raise ValueError('条件softmax分组不连续或数量不符')
    begin = np.r_[0, np.cumsum(sizes)[:-1]]
    group = np.repeat(np.arange(len(sizes)), sizes)
    ex = np.exp(utilities - np.maximum.reduceat(utilities, begin)[group])
    res = ex / np.add.reduceat(ex, begin)[group]
    if not np.isfinite(res).all() or not np.allclose(np.add.reduceat(res, begin), 1, atol=1e-8):
        raise ValueError('按OD归一化模型概率失败')
    return res


def run_frozen_model(z, q, model, sizes, model_name):
    meta = model['models'][model_name]
    blocks = MODEL_BLOCKS[model_name]
    if meta.get('blocks') != blocks:
        raise ValueError(f'{model_name}特征块不匹配，拒绝假定模型结构')
    x = x_for_model(z, q, blocks)
    beta = np.asarray(meta['beta'], dtype=float)
    if len(beta) != x.shape[1] or not np.isfinite(beta).all():
        raise ValueError(f'{model_name}冻结模型系数数量或数值无效')
    utility = x @ beta
    return utility, group_softmax(utility, sizes)


def overlap_matrix(trip, ids, pairs):
    """依赖模块五已保存的真实有向边长度加权Jaccard；不读取GraphML。"""
    n = len(ids)
    mat = np.zeros((n, n), dtype=float)
    idmap = {str(cid): i for i, cid in enumerate(ids)}
    if len(idmap) != n:
        raise ValueError(f'{trip}: 候选ID重复')
    records = pairs.get(str(trip), [])
    if len(records) != n*(n-1)//2:
        raise ValueError(f'{trip}: 原模块五道路重合记录不足，应为{n*(n-1)//2}，实为{len(records)}')
    touched = set()
    for ca, cb, value in records:
        if ca not in idmap or cb not in idmap or ca == cb:
            raise ValueError(f'{trip}: 两两道路重叠含外部/重复候选')
        i, j = sorted((idmap[ca], idmap[cb]))
        if (i, j) in touched or not np.isfinite(value) or not -1e-9 <= value <= 1+1e-9:
            raise ValueError(f'{trip}: 重叠项重复或数值不合法')
        touched.add((i,j))
        mat[i,j] = mat[j,i] = float(value)
    return mat


def rank_topk(ids, scores, overlap, k=3, lam=.18, maximum_overlap=.8):
    """与第五模块 rank_topk 相同，按照调整分、最大重叠和候选ID破同分。"""
    ids = list(map(str, ids))
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('排序得分非有限值')
    if len(set(ids)) != len(ids) or len(scores) != len(ids):
        raise ValueError('排序候选ID不唯一或分数缺失')
    chosen=[]
    while len(chosen) < min(k, len(ids)):
        opts=[]
        for i in range(len(ids)):
            if i in chosen: continue
            mx = max((float(overlap[i,j]) for j in chosen), default=0.)
            adj = float(scores[i]) - (lam * mx if chosen else 0.)
            feasible = (not chosen or mx <= maximum_overlap+1e-12)
            opts.append((i, adj, mx, feasible))
        use = [t for t in opts if t[3]]
        if not use: use=opts
        best = min(use, key=lambda t: (-round(t[1],12), round(t[2],12), ids[t[0]]))
        chosen.append(best[0])
    return chosen


def paths(a):
    root = a.root
    m2 = root/'模块二_UDPPMpp双维历史画像'
    m3 = root/'模块三_未知OD自然候选与静态道路属性'/'full_K10'
    m4 = root/'模块四_条件选择六模型与未知OD盲排序'/'formal_K10'
    m5 = root/'模块五_双维偏好_三模式Top3多目标推荐'/'formal_K10'
    return {
        'model': a.model or m4/'06_正式模型参数与训练来源.json',
        'm4_audit': a.m4_audit or m4/'09_最后3天盲排序与时间隔离审计.json',
        'm4_pred': a.m4_pred or m4/'07_最后3天六模型逐路线盲排序.csv.gz',
        'static': a.static or m3/'02_未知OD候选静态属性与PathSize_模型输入.csv.gz',
        'profile': a.profile or m2/'05_最后3天盲推理8维画像_只按trip_id.csv',
        'reference': a.reference or m2/'06_前28天冻结群体画像.json',
        'm5_raw_score': a.m5_raw_score or m5/'01_全部候选_三模式多目标代理分数.csv.gz',
        'm5_pair': a.m5_pair or m5/'07_逐OD_自然候选两两有向边长度重叠.csv.gz',
        'm5_audit': a.m5_audit or m5/'31_连续需求_来源防泄漏与统计审计.json',
        'm5_top3': a.m5_top3 or m5/'24_连续需求_三场景Top3主方法.csv.gz',
        'm5_original_audit': m5/'08_模块五_完整性盲测与来源审计.json',
    }


def validate_inputs(p, a):
    for label, pp in p.items():
        file_ok(pp, label)
    model, m4, m5 = readj(p['model']), readj(p['m4_audit']), readj(p['m5_audit'])
    if model.get('pilot') is not False or m4.get('mode') != 'FULL_REAL' or m5.get('mode') != 'FULL_REAL':
        raise ValueError('本程序必须读取formal_K10真实全量模型与结果，禁止混用pilot')
    if m4.get('actual_observed_routes_read') != 'NEVER' or m4.get('chosen_test_labels_read') != 'NEVER':
        raise ValueError('模块四输入隔离审计不通过')
    if m5.get('future_true_HMM_trajectory_read') != 'NEVER':
        raise ValueError('模块五连续需求实验有真值读入痕迹')
    if m5.get('module4_retrained') != 'NEVER':
        raise ValueError('模块五来源说明模块四模型曾重新训练')
    for key in MODEL_BLOCKS:
        if key not in model.get('models', {}):
            raise ValueError(f'缺失已冻结模型{key}')
    if model.get('model_features') != SCALE_COLS:
        raise ValueError('冻结模型的特征顺序不是当前预期顺序')
    if m4.get('source_model_sha256') != sha(p['model']):
        raise ValueError('模块四保存的模型SHA256与当前模型不同')
    if m4.get('source_test_candidates_sha256') != sha(p['static']):
        raise ValueError('模块四使用的静态候选与当前文件不同')
    if m5.get('frozen_module4_model_sha256') != sha(p['model']):
        raise ValueError('模块五所依赖的冻结模型与本次模型不一致')
    if m5.get('frozen_module4_predictions_sha256') != sha(p['m4_pred']):
        raise ValueError('模块五所依赖的冻结预测文件与本次文件不一致')
    old = readj(p['m5_original_audit'])
    # 第五模块v4在生成之前会核对01号、07号冻结来源。
    for key, file_key in [('frozen_module5_all_candidates_sha256','m5_raw_score'),
                          ('frozen_module5_pairwise_jaccard_sha256','m5_pair')]:
        if m5.get(key) and m5[key] != sha(p[file_key]):
            raise ValueError(f'{key}哈希与当前输入不匹配')
    if m5.get('new_output_sha256', {}).get('main') != sha(p['m5_top3']):
        raise ValueError('模块五24号Top-3与其31号审计SHA256不匹配')
    if int(m5.get('all_OD',0)) != 4469 or int(m5.get('natural_candidates',0)) != 37383:
        raise ValueError('不是预期正式4469 OD与37383自然候选版本')
    if not math.isclose(float(m5.get('continuous_demand_objective_weight', -1)), .70, abs_tol=1e-12):
        raise ValueError('模块五冻结的需求目标权重不是0.7，不能复用本代码公式')
    if not math.isclose(float(m5.get('diversity_lambda', -1)), .18, abs_tol=1e-12) or \
       not math.isclose(float(m5.get('maximum_overlap', -1)), .8, abs_tol=1e-12):
        raise ValueError('模块五冻结的重叠规则不同，拒绝假设')
    reference = readj(p['reference'])
    if '前28天' not in str(reference.get('source','')):
        raise ValueError('群体参考画像不是从前28天历史数据冻结得到的')
    if str(reference.get('as_of',''))[:10] != '2024-12-29':
        raise ValueError('群体参考画像的冻结时间不在测试期开始之前')
    econ = np.asarray(reference.get('economic_probabilities'), dtype=float)
    comf = np.asarray(reference.get('comfort_probabilities'), dtype=float)
    if econ.shape != (4,) or comf.shape != (4,):
        raise ValueError('群体参考画像应为经济4维+舒适4维')
    ref = np.concatenate([econ,comf])[None,:]
    check_profiles(ref, '前28天群体参考画像')
    log('来源核验通过：模块四formal_K10模型、模块五连续需求v4、前28天群体参考画像全部冻结')
    return model, m4, m5, ref[0]


def prep_data(p, model, ref):
    static = readc(p['static'])
    need(static, ['trip_id', CID] + SCALE_COLS, '模块三真实候选')
    if static[CID].duplicated().any() or static[['trip_id',CID]].duplicated().any():
        raise ValueError('模块三候选ID重复')
    if FORBIDDEN.intersection(static.columns):
        raise ValueError(f'模块三静态候选混入未来标签/身份列：{FORBIDDEN.intersection(static.columns)}')
    profiles = readc(p['profile'])
    need(profiles, ['trip_id','history_n_prior']+PROFILE, '模块二严格历史8维画像')
    if profiles.trip_id.duplicated().any():
        raise ValueError('模块二同一trip_id出现多条画像')
    profiles['history_n_prior'] = pd.to_numeric(profiles['history_n_prior'], errors='raise')
    if (profiles.history_n_prior<0).any():
        raise ValueError('历史轨迹次数不能为负')
    qod = profiles[PROFILE].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    check_profiles(qod,'模块二逐OD真实画像')
    raw = static.merge(profiles[['trip_id','history_n_prior']+PROFILE], on='trip_id',
                       how='left', validate='many_to_one')
    if raw[PROFILE+['history_n_prior']].isna().any().any():
        raise ValueError('自然候选关联画像时有缺失，不能插入假画像')
    raw = raw.sort_values(['trip_id',CID], kind='mergesort').reset_index(drop=True)
    if len(raw) != 37383 or raw.trip_id.nunique() != 4466:
        raise ValueError(f'真实候选数量不匹配：{len(raw)}条，OD={raw.trip_id.nunique()}')
    # 只读取原第五模块经济场景一次，以获得前一阶段冻结的道路需求代理分数。
    oldscore = readc(p['m5_raw_score'])
    need(oldscore,['trip_id',CID,'scenario','economic_proxy_score','comfort_proxy_score',
                   'model_UDPPM_Econ_probability','model_UDPPM_Comf_probability',
                   'model_UDPPM_Dual_probability'], '模块五01号冻结候选评分')
    oldscore = oldscore.loc[oldscore.scenario.astype(str)=='economic'].copy()
    if len(oldscore)!=len(raw) or oldscore[[CID]].duplicated().any():
        raise ValueError('原第五模块经济场景候选数量或ID不唯一')
    keep = [CID,'economic_proxy_score','comfort_proxy_score',
            'model_UDPPM_Econ_probability','model_UDPPM_Comf_probability','model_UDPPM_Dual_probability']
    raw = raw.merge(oldscore[keep],on=CID,how='left',validate='one_to_one')
    if raw[keep[1:]].isna().any().any():
        raise ValueError('模型候选与原第五模块冻结需求评分没有完整对应')
    pairs = readc(p['m5_pair'])
    need(pairs,['trip_id','candidate_id_a','candidate_id_b','pairwise_length_jaccard'],'原模块五道路重叠记录')
    if len(pairs) != 152661:
        raise ValueError(f'当前道路两两重叠记录数量{len(pairs)}，预期152661')
    lookup=defaultdict(list)
    for t,ca,cb,v in pairs[['trip_id','candidate_id_a','candidate_id_b','pairwise_length_jaccard']].itertuples(index=False,name=None):
        lookup[str(t)].append((str(ca),str(cb),float(v)))
    scores = readc(p['m4_pred'])
    need(scores,['trip_id',CID,'model','choice_probability'], '模块四冻结盲排序')
    scores = scores.loc[scores.model.isin(MODEL_BLOCKS)].copy()
    if len(scores)!=3*len(raw):
        raise ValueError('模块四三种画像PSL候选预测数不全')
    if scores[['trip_id',CID,'model']].duplicated().any():
        raise ValueError('模块四候选预测ID/model重复')
    src = scores.pivot(index=CID,columns='model',values='choice_probability')
    src = src.reindex(raw[CID].astype(str))
    if src.isna().any().any():
        raise ValueError('模块四冻结预测与模块三候选有缺口')
    old = readc(p['m5_top3'])
    need(old, ['trip_id','scenario','rank_in_scenario',CID], '模块五24号真实Top3推荐')
    if old[['trip_id','scenario','rank_in_scenario']].duplicated().any():
        raise ValueError('模块五真实推荐存在重复OD+场景+名次')
    if len(old) != 38931:
        raise ValueError(f'模块五Top-3实际行数{len(old)}不等于38931')
    return raw, lookup, src, old


def replay_and_compare(raw, lookup, frozen_prob, old_top3, model, reference, a):
    z=scale_static(raw,model['scaler_params'])
    q=raw[PROFILE].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    check_profiles(q,'逐候选历史画像')
    qr=np.broadcast_to(reference[None,:],q.shape).copy()
    ids=raw[CID].astype(str).to_numpy()
    sizes=raw.groupby('trip_id',sort=False).size().to_numpy(dtype=int)
    oid=raw.groupby('trip_id',sort=False).size().index.astype(str).tolist()
    original={};neutral={};pred_err={}
    for name in MODEL_BLOCKS:
        _,pa=run_frozen_model(z,q,model,sizes,name)
        _,pb=run_frozen_model(z,qr,model,sizes,name)
        original[name]=pa;neutral[name]=pb
        expected=frozen_prob[name].to_numpy(dtype=float)
        err=np.abs(pa-expected)
        pred_err[name]={'maximum_probability_absolute_error':float(err.max()),
                        'mean_absolute_error':float(err.mean())}
        if err.max()>a.prediction_tolerance:
            idx=int(err.argmax())
            raise ValueError(f'冻结模型无法复现模块四概率：{name} 候选{ids[idx]} '
                             f'重新计算={pa[idx]:.12g}, 已保存={expected[idx]:.12g}, '
                             f'maxdiff={err.max():.3g}，停止敏感性研究。'
                             f'检查模型JSON/标准化/画像/候选来源。')
        log(f'{name}重建成功：最大概率误差={err.max():.3g}')
    n = len(raw)
    assert sum(sizes)==n
    counts=np.r_[0,np.cumsum(sizes)]
    oldlookup={}
    for (t,s),g in old_top3.groupby(['trip_id','scenario'],sort=False):
        gg=g.sort_values('rank_in_scenario',kind='mergesort')
        oldlookup[(str(t),str(s))]=gg[CID].astype(str).tolist()
    rows=[];score_rows=[];summary_sample=[];rank_replay_mismatches=[]
    for k,t in enumerate(oid):
        lo,hi=int(counts[k]),int(counts[k+1]);g=raw.iloc[lo:hi]
        curids=ids[lo:hi].tolist();sim=overlap_matrix(t,curids,lookup)
        econ=nparr(g,'economic_proxy_score');comf=nparr(g,'comfort_proxy_score')
        local_real=g[PROFILE].iloc[0].to_numpy(dtype=float)
        n_prior=int(g.history_n_prior.iloc[0])
        # 独立验证模型在当前OD之间是否产生了直接效用分数变化。
        for sc in SCENARIOS:
            name=SC_MODEL[sc]
            actual_model=original[name][lo:hi]
            neutral_model=neutral[name][lo:hi]
            qd = econ if sc=='economic' else (comf if sc=='comfortable' else .5*(econ+comf))
            score_a = a.objective_weight*qd + (1-a.objective_weight)*numeric_score(actual_model)
            score_b = a.objective_weight*qd + (1-a.objective_weight)*numeric_score(neutral_model)
            ia=rank_topk(curids,score_a,sim,3,a.diversity_lambda,a.maximum_overlap)
            ib=rank_topk(curids,score_b,sim,3,a.diversity_lambda,a.maximum_overlap)
            aa=[curids[i] for i in ia];bb=[curids[i] for i in ib]
            oldkey=(t,sc)
            if oldkey not in oldlookup:
                raise ValueError(f'模块五原Top3缺少{oldkey}')
            if aa!=oldlookup[oldkey]:
                rank_replay_mismatches.append({'trip_id':t,'scenario':sc,
                                               'recomputed':aa,'original':oldlookup[oldkey]})
            actual_sel = g.iloc[ia]
            neut_sel = g.iloc[ib]
            # 越低越好的长度/转弯指标，越高越好的代理指标，不把代理值叫真实舒适度。
            goal_a=qd[ia[0]];goal_b=qd[ib[0]]
            pair_a=[sim[ia[i],ia[j]] for i in range(len(ia)) for j in range(i+1,len(ia))]
            pair_b=[sim[ib[i],ib[j]] for i in range(len(ib)) for j in range(i+1,len(ib))]
            same_set = set(aa)==set(bb)
            same_order=aa==bb
            row={
                'trip_id':t,'scenario':sc,'candidate_count':len(curids),
                'history_n_prior':n_prior,
                'history_group': ('0' if n_prior==0 else '1-5' if n_prior<=5 else '6-20' if n_prior<=20 else '>20'),
                'has_3_candidates':int(len(curids)>=3),
                'top1_same':int(aa[0]==bb[0]),
                'rank2_same':(int(aa[1]==bb[1]) if len(aa)>=2 else np.nan),
                'rank3_same':(int(aa[2]==bb[2]) if len(aa)>=3 else np.nan),
                'top3_set_same':int(same_set),
                'top3_order_same':int(same_order),
                'set_same_but_order_changed':int(same_set and not same_order),
                'num_replaced_top3':len(set(aa)-set(bb)),
                'actual_top1_id':aa[0],'reference_top1_id':bb[0],
                'actual_top2_id':aa[1] if len(aa)>1 else '',
                'reference_top2_id':bb[1] if len(bb)>1 else '',
                'actual_top3_id':aa[2] if len(aa)>2 else '',
                'reference_top3_id':bb[2] if len(bb)>2 else '',
                'actual_top3_set':json.dumps(sorted(aa),ensure_ascii=False),
                'reference_top3_set':json.dumps(sorted(bb),ensure_ascii=False),
                'actual_top3_order':json.dumps(aa,ensure_ascii=False),
                'reference_top3_order':json.dumps(bb,ensure_ascii=False),
                'profile_l1_distance_from_reference':float(np.abs(local_real-reference).sum()),
                'model_choice_prob_max_abs_diff':float(np.abs(actual_model-neutral_model).max()),
                'model_choice_prob_mean_abs_diff':float(np.abs(actual_model-neutral_model).mean()),
                'model_rank_same':int(curids[int(np.argmax(actual_model))]==curids[int(np.argmax(neutral_model))]),
                'top1_goal_actual':float(goal_a),'top1_goal_reference':float(goal_b),
                'top1_goal_delta_actual_minus_reference':float(goal_a-goal_b),
                'top1_len_actual_km':float(actual_sel.total_len_km.iloc[0]),
                'top1_len_reference_km':float(neut_sel.total_len_km.iloc[0]),
                'top1_length_delta_km':float(actual_sel.total_len_km.iloc[0]-neut_sel.total_len_km.iloc[0]),
                'top1_turns_per_km_delta':float(actual_sel.turns_per_km.iloc[0]-neut_sel.turns_per_km.iloc[0]),
                'top3_mean_len_delta_km':float(actual_sel.total_len_km.mean()-neut_sel.total_len_km.mean()),
                'top3_mean_economic_proxy_delta':float(actual_sel.economic_proxy_score.mean()-neut_sel.economic_proxy_score.mean()),
                'top3_mean_comfort_proxy_delta':float(actual_sel.comfort_proxy_score.mean()-neut_sel.comfort_proxy_score.mean()),
                'top3_mean_overlap_delta':(float(np.mean(pair_a)-np.mean(pair_b)) if pair_a else np.nan),
            }
            rows.append(row)
            # 保留路线级预测差值，必要时可还原任意OD全部自然候选的两种画像效用。
            for j,cid in enumerate(curids):
                score_rows.append({'trip_id':t,'scenario':sc,CID:cid,
                    'history_n_prior':n_prior,'model':name,
                    'candidate_probability_actual':float(actual_model[j]),
                    'candidate_probability_reference':float(neutral_model[j]),
                    'candidate_probability_delta':float(actual_model[j]-neutral_model[j]),
                    'continuous_goal':float(qd[j]),
                    'total_score_actual':float(score_a[j]),
                    'total_score_reference':float(score_b[j]),
                    'actual_rank_in_top3':(ia.index(j)+1 if j in ia else 0),
                    'reference_rank_in_top3':(ib.index(j)+1 if j in ib else 0)})
        if (k+1)%600==0 or (k+1)==len(oid):
            log(f'精确Top-3对照已完成 {k+1:,}/{len(oid):,}个真实OD')
    if rank_replay_mismatches:
        bad=rank_replay_mismatches[0]
        raise ValueError(f'模型概率已复现，但模块五v4原始Top3无法复现！'
                         f'共{len(rank_replay_mismatches)}个场景OD不匹配，例如{bad}。'
                         f'需检查道路重叠、同OD归一化、原排序参数；为避免无效消融，不发布结果。')
    log('模块五24号全部Top3候选ID与同规则真实画像重算结果100%对应，敏感性比较可用')
    result=pd.DataFrame(rows)
    scores=pd.DataFrame(score_rows)
    # 直接使用模块五27号原本已冻结的strong_conflict标签（不在此处重定义）。
    return result,scores,pred_err


def attach_conflict(df, a):
    p=a.change_file or (a.root/'模块五_双维偏好_三模式Top3多目标推荐'/'formal_K10' /
                       '27_经济舒适场景切换_逐OD五方法详细变化.csv')
    original = readc(p)
    need(original,['trip_id','variant','strong_conflict'], '第五模块强冲突OD标签')
    original = original[original.variant.astype(str)=='proposed'][['trip_id','strong_conflict']]
    if original.trip_id.duplicated().any() or len(original)!=4466:
        raise ValueError('第五模块的强冲突标签OD数量或唯一性错误')
    out=df.merge(original,on='trip_id',how='left',validate='many_to_one')
    if out.strong_conflict.isna().any():
        raise ValueError('存在未对应的强冲突OD')
    if int(original.strong_conflict.sum())!=1224:
        raise ValueError('正式来源强冲突OD数量不是1224')
    return out,p


def clean_values(x):
    v=np.asarray(x,dtype=float)
    return v[np.isfinite(v)]


def ci(x, repeats, seed):
    v=clean_values(x)
    if len(v)==0:return np.nan,np.nan
    if len(v)==1:return float(v[0]),float(v[0])
    rng=np.random.default_rng(seed)
    means=np.empty(repeats)
    # 每次抽取OD级配对差值；多个OD可能来自同车，不能声称车辆簇独立。
    for i in range(repeats):
        means[i]=v[rng.integers(0,len(v),size=len(v))].mean()
    return tuple(map(float,np.percentile(means,[2.5,97.5])))


def summarize(df, a):
    subsets={
        '全部有候选OD':df,
        '强冲突OD':df[df.strong_conflict==1],
        '非强冲突OD':df[df.strong_conflict==0],
        '有历史OD':df[df.history_n_prior>0],
        '完全无历史OD':df[df.history_n_prior==0],
    }
    group_rows=[];quality_rows=[];history_rows=[]
    difference_metrics=[
        'top1_goal_delta_actual_minus_reference',
        'top1_length_delta_km','top1_turns_per_km_delta',
        'top3_mean_len_delta_km','top3_mean_economic_proxy_delta',
        'top3_mean_comfort_proxy_delta','top3_mean_overlap_delta',
    ]
    for (subset_name,part) in subsets.items():
        for sc in SCENARIOS:
            g=part[part.scenario==sc]
            h=g[g.has_3_candidates==1]
            if g.empty:continue
            group_rows.append({
                'sample_group':subset_name,'scenario':sc,'n_scored_od':len(g),
                'n_rank3_eligible':len(h),
                'top1_same_rate':float(g.top1_same.mean()),
                'rank2_same_rate':float(h.rank2_same.mean()) if len(h) else np.nan,
                'rank3_same_rate':float(h.rank3_same.mean()) if len(h) else np.nan,
                'top3_set_same_rate':float(h.top3_set_same.mean()) if len(h) else np.nan,
                'top3_order_same_rate':float(h.top3_order_same.mean()) if len(h) else np.nan,
                'model_argmax_same_rate':float(g.model_rank_same.mean()),
                'mean_model_probability_max_abs_change':float(g.model_choice_prob_max_abs_diff.mean()),
                'n_top1_changed':int((g.top1_same==0).sum()),
                'n_top3_set_changed':int((h.top3_set_same==0).sum()),
                'n_top3_order_changed':int((h.top3_order_same==0).sum()),
                'n_same_set_different_order':int(h.set_same_but_order_changed.sum()),
                'mean_top3_candidates_replaced':float(h.num_replaced_top3.mean()) if len(h) else np.nan,
            })
            for mask,label in [(np.ones(len(g),bool),'全部OD'),
                               (g.top1_same.to_numpy()==0,'Top1发生变化OD'),
                               (g.top3_set_same.to_numpy()==0,'Top3集合发生变化OD')]:
                gg=g.loc[mask]
                for k,met in enumerate(difference_metrics):
                    v=clean_values(gg[met])
                    low,up=ci(v,a.bootstrap,a.seed+k)
                    quality_rows.append({'sample_group':subset_name,'scenario':sc,'selection':label,
                                         'metric':met,'valid_od_n':len(v),
                                         'mean_actual_minus_reference':float(v.mean()) if len(v) else np.nan,
                                         'median_actual_minus_reference':float(np.median(v)) if len(v) else np.nan,
                                         'ci95_low':low,'ci95_high':up})
    for sc,gg in df.groupby('scenario',sort=False):
        for hist,g in gg.groupby('history_group',sort=False):
            hh=g[g.has_3_candidates==1]
            history_rows.append({'scenario':sc,'history_group':hist,'n_od':len(g),
                'n_rank3_eligible':len(hh),
                'top1_same_rate':float(g.top1_same.mean()),
                'top3_set_same_rate':float(hh.top3_set_same.mean()) if len(hh) else np.nan,
                'top3_order_same_rate':float(hh.top3_order_same.mean()) if len(hh) else np.nan,
                'average_prior_trips':float(g.history_n_prior.mean()),
                'average_profile_l1_dist_from_reference':float(g.profile_l1_distance_from_reference.mean()),
                'average_model_probability_max_abs_change':float(g.model_choice_prob_max_abs_diff.mean())})
    return pd.DataFrame(group_rows),pd.DataFrame(quality_rows),pd.DataFrame(history_rows)


def draw(df, summary, hist, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':240})
    names={'economic':'Economic','comfortable':'Comfort','balanced':'Balanced'}
    main=summary[summary.sample_group=='全部有候选OD'].set_index('scenario').reindex(SCENARIOS)
    ys=np.arange(3)
    fig,ax=plt.subplots(figsize=(9,5.2))
    values=np.vstack([main.top1_same_rate.to_numpy(float),
                      main.top3_set_same_rate.to_numpy(float),
                      main.top3_order_same_rate.to_numpy(float)]).T*100
    for j,lab in enumerate(['Same Top-1','Same Top-3 candidate set','Same Top-3 order']):
        ax.bar(ys+(j-1)*.25,values[:,j],width=.23,label=lab)
    ax.set_xticks(ys,[names[t] for t in SCENARIOS]);ax.set_ylim(0,105)
    ax.set_ylabel('Agreement between actual/reference profile (%)')
    ax.set_title('Frozen model: route agreement under profile replacement')
    ax.legend(ncol=2,fontsize=9,loc='lower left');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'05_冻结模型_真实参考画像Top3对照.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.2))
    groups=['0','1-5','6-20','>20'];labels=['0','1–5','6–20','>20']
    for sc in SCENARIOS:
        t=hist[hist.scenario==sc].set_index('history_group').reindex(groups)
        ax.plot(range(4),t.top3_set_same_rate.to_numpy(float)*100,marker='o',label=names[sc])
    ax.set_xticks(range(4),labels);ax.set_ylabel('Top-3 candidate-set agreement (%)')
    ax.set_xlabel('Completed historical trips before OD');ax.set_ylim(0,105)
    ax.set_title('Frozen profile-sensitivity by prior history availability')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'06_按历史轨迹数量分层_画像敏感性.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for i,(met,title) in enumerate([
        ('model_choice_prob_max_abs_diff','Largest within-OD probability change'),
        ('profile_l1_distance_from_reference','Profile L1 distance to reference')]):
        vals=[df.loc[df.scenario==sc,met].to_numpy(float) for sc in SCENARIOS]
        axs[i].boxplot(vals,labels=[names[sc] for sc in SCENARIOS],showfliers=False)
        axs[i].set_title(title);axs[i].grid(axis='y',alpha=.2)
    fig.suptitle('Actual vs fixed reference profile: input and model sensitivity')
    fig.tight_layout();fig.savefig(out/'07_画像输入差异与模型概率变化.png');plt.close(fig)


def save_csv(df, p):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.writing')
    df.to_csv(tmp,index=False,encoding='utf-8-sig',float_format='%.12g')
    tmp.replace(p)


def save_json(o,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf-8-sig')
    tmp.replace(p)


def self_test():
    # 仅验证代码逻辑的固定人工单元测试，不作为论文实验结果或图表。
    assert np.allclose(numeric_score([1,2,3]),[0,.5,1])
    assert np.allclose(numeric_score([2,2]),[.5,.5])
    q=np.array([[.25,.25,.25,.25,.25,.25,.25,.25]],float)
    zz=np.ones((1,12),float)
    assert x_for_model(zz,q,['base','ps_length','econ']).shape==(1,31)
    assert x_for_model(zz,q,['base','ps_length','comf']).shape==(1,31)
    assert x_for_model(zz,q,['base','ps_length','econ','comf']).shape==(1,51)
    assert np.allclose(group_softmax([0.,1.,2.],[2,1]),[1/(1+math.e),math.e/(1+math.e),1])
    assert rank_topk(['b','a'],[.5,.5],np.zeros((2,2)),3)==[1,0]
    log('单元检查通过。注意：这些固定测试值不是论文数据。')


def parse_args():
    p=argparse.ArgumentParser(description='冻结UDPPM++模型的8维画像输入敏感性，严格复现模块四概率+模块五Top3')
    p.add_argument('--phase',choices=['check','analyze','self-test'],default='analyze')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--model',type=Path)
    p.add_argument('--m4-audit',type=Path)
    p.add_argument('--m4-pred',type=Path)
    p.add_argument('--static',type=Path)
    p.add_argument('--profile',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--m5-raw-score',type=Path)
    p.add_argument('--m5-pair',type=Path)
    p.add_argument('--m5-audit',type=Path)
    p.add_argument('--m5-top3',type=Path)
    p.add_argument('--change-file',type=Path)
    p.add_argument('--prediction-tolerance',type=float,default=1e-6,
                   help='重建模型概率必须与第四模块冻结概率一致，默认最大绝对误差<=1e-6')
    p.add_argument('--objective-weight',type=float,default=.70)
    p.add_argument('--diversity-lambda',type=float,default=.18)
    p.add_argument('--maximum-overlap',type=float,default=.8)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-plot',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='仅允许覆盖本程序专属目录，绝不修改模块一至六')
    return p.parse_args()


def main():
    a=parse_args()
    if a.phase=='self-test':
        self_test();return
    if not 0 < a.objective_weight < 1 or abs(a.objective_weight-.7)>1e-12:
        raise ValueError('本实验必须复现第五模块原0.7/0.3权重，不允许调参')
    if abs(a.diversity_lambda-.18)>1e-12 or abs(a.maximum_overlap-.8)>1e-12:
        raise ValueError('本实验必须保持第五模块原有多样性设置')
    if not (0<a.prediction_tolerance<=1e-5):
        raise ValueError('概率复现容差不合法，不能通过放宽验证容差掩盖错误')
    if a.bootstrap<100:raise ValueError('Bootstrap重复至少100次')
    p=paths(a)
    model,m4,m5,ref=validate_inputs(p,a)
    if a.phase=='check':
        log('全部输入文件及模型来源可用；执行 --phase analyze 才读取完整候选并复算Top3')
        return
    a.out = a.out or (a.root/'模块六补充_冻结UDPPM画像输入敏感性'/'formal_K10')
    if a.out.resolve() in {Path(x).parent.resolve() for x in p.values()}:
        raise ValueError('禁止将补充实验输出目录设为原有模型/候选/推荐文件夹')
    planned = ['00_实验来源与重建校验.json','01_逐OD三场景_真实画像与群体参考画像Top3精确对照.csv',
               '02_逐候选三场景_冻结模型效用敏感性.csv',
               '03_全部及强冲突OD_画像敏感性总体统计.csv',
               '04_实际画像与群体参考画像_目标质量配对差与置信区间.csv',
               '05_历史记录数量分层_画像敏感性.csv',
               '06_模型概率复现误差表.csv',
               '05_冻结模型_真实参考画像Top3对照.png','06_按历史轨迹数量分层_画像敏感性.png',
               '07_画像输入差异与模型概率变化.png']
    existing=[x for x in planned if (a.out/x).exists()]
    if existing and not a.overwrite:
        raise FileExistsError(f'专属输出目录已有{existing[0]}，拒绝覆盖；如确认使用 --overwrite')
    log('读取模块三静态候选+模块二画像+模块四冻结模型与概率+模块五冻结目标/重叠')
    raw,lookup,probs,top3=prep_data(p,model,ref)
    df,score_rows,pred_err=replay_and_compare(raw,lookup,probs,top3,model,ref,a)
    df,cp=attach_conflict(df,a)
    summary,quality,hist=summarize(df,a)
    if len(df)!=4466*3 or df.trip_id.nunique()!=4466:
        raise ValueError('逐OD三场景结果数量不正确')
    a.out.mkdir(parents=True,exist_ok=True)
    save_csv(df,a.out/planned[1]);save_csv(score_rows,a.out/planned[2])
    save_csv(summary,a.out/planned[3]);save_csv(quality,a.out/planned[4])
    save_csv(hist,a.out/planned[5])
    save_csv(pd.DataFrame([{'model':k,**v} for k,v in pred_err.items()]),a.out/planned[6])
    if not a.no_plot:draw(df,summary,hist,a.out)
    audit={
        'version':'frozen_UDPPMpp_profile_input_sensitivity_v1.0',
        'created_at':datetime.now().isoformat(),
        'experiment_type':'FROZEN_MODEL_INPUT_SENSITIVITY_NOT_REFIT_ABLATION',
        'selection_only_changes':'8D actual history probabilities replaced by fixed pre-test historical group reference',
        'source_files_sha256':{k:sha(v) for k,v in p.items()},
        'conflict_source_sha256':sha(cp),
        'all_od':4469,'scored_od':4466,'natural_routes':37383,
        'n_od_scenario_rows':len(df),'n_candidate_scenario_rows':len(score_rows),
        'n_strong_conflict_od':1224,
        'rank3_eligible_od':int(df.loc[(df.scenario=='economic')&(df.has_3_candidates==1),'trip_id'].nunique()),
        'reference_file':str(p['reference']),
        'reference_8d':ref.tolist(),
        'scaler':'frozen module4 CandidateScaler median/mean/std; clip [-8,8]',
        'interactions':'(history_dimension_k-0.25)*scaled_road_attribute; econ/comf each 4x5',
        'objective_weight':a.objective_weight,
        'frozen_model_weight':1-a.objective_weight,
        'diversity_lambda':a.diversity_lambda,
        'maximum_overlap':a.maximum_overlap,
        'model_probability_reconstruction_error':pred_err,
        'module5_main_Top3_rank_replay':'MATCHED_ALL_13398_OD_SCENARIOS',
        'future_true_HMM_route_read':'NEVER',
        'model_retrained':'NEVER',
        'natural_candidates_regenerated':'NEVER',
        'interpretation_boundaries':[
            '仅同一冻结模型的画像输入敏感性；不能宣称UDPPM++画像带来因果或泛化收益。',
            '群体参考画像仅由前28天历史数据计算，不读后三天实际行驶路径。',
            '固定的群体参考不等于重新拟合无画像模型。',
            '经济和舒适性来自同OD静态道路结构代理，不是实测油耗/振动/主观舒适。',
            '同一车辆可能有多个OD，OD级Bootstrap未处理车辆内相关性。',
            '后三天测试数据曾用于此前探索，因此本次不称为全新独立盲测。',
            '原模块五Top-3精确复现方可发布本次敏感性结果。',
        ]
    }
    audit['output_sha256']={x:sha(a.out/x) for x in planned[1:] if (a.out/x).exists()}
    save_json(audit,a.out/planned[0])
    log(f'完成：{len(df):,}条OD×场景、{len(score_rows):,}条候选×场景，输出目录：{a.out}')
    print('\n画像输入敏感性摘要（注意：非画像泛化收益）：')
    print(summary.loc[summary.sample_group=='全部有候选OD',
          ['scenario','n_scored_od','n_rank3_eligible','top1_same_rate','top3_set_same_rate',
           'top3_order_same_rate','mean_model_probability_max_abs_change']].to_string(index=False))


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        log('用户中断；未修改任何上游模型/候选/推荐文件');sys.exit(130)
    except Exception as exc:
        log(f'冻结画像敏感性实验停止：{type(exc).__name__}: {exc}');raise
