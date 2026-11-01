# -*- coding: utf-8 -*-
"""
第四章模块六｜连续需求路径推荐：强目标冲突、多方法消融、历史画像增量与独立路线重合评价
适配模块五 5step-55.py --phase demand-soft 的正式v4输出（26—34及24/25）。

默认阶段完全不读取最后3天真实HMM轨迹。
    python 6step-6.py --phase check
    python 6step-6.py --phase summarize
    python 6step-6.py --phase plot
    python 6step-6.py --phase all
仅需要辅助的实际行驶路线重合评价时，先 all，再执行：
    python 6step-6.py --phase evaluate-truth
测试环境：
    python 6step-6.py --phase self-test

不会修改模块一至模块五的任何文件；不重新训练/重搜候选/调节模型参数。
本代码仅使用上传的真实字段；未实测的舒适性、真实司机需求、油耗不作推断。
“需求”由用户事先明确给出。未收集乘客真实需求标签，不能计算真实需求预测准确率。
前面测试OD曾参与v3/v4改进，因此所有研究性比较均属于探索性分析，
不是一个全新独立数据集上的泛化验证。历史画像按vehicle_id汇总，不等于稳定司机身份。

依赖：pip install numpy pandas matplotlib networkx (networkx仅评价真值阶段需要)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

BASE = Path(r'C:\study\轨迹数据预处理')
ROOT = BASE / '曼谷GPS_正式清洗_202412' / 'FULL_全量' / '第四章_28天训练_3天独立测试'
DEFAULT_M5 = ROOT / '模块五_双维偏好_三模式Top3多目标推荐' / 'formal_K10'
DEFAULT_M2 = ROOT / '模块二_UDPPMpp双维历史画像'
DEFAULT_M1 = ROOT / '模块一_时间划分与隔离'
DEFAULT_M3 = ROOT / '模块三_未知OD自然候选与静态道路属性' / 'full_K10'
DEFAULT_M4 = ROOT / '模块四_条件选择六模型与未知OD盲排序' / 'formal_K10'
DEFAULT_OUT = ROOT / '模块六_连续需求_强冲突画像消融与独立评价' / 'formal_K10'
DEFAULT_GRAPH = BASE / 'luwangdata' / '曼谷_路网_兼容旧版.graphml'

SCENARIOS = ('economic', 'comfortable', 'balanced')
VARIANTS = ('proposed', 'no_profile', 'objective_only', 'no_diversity', 'utility_only')
MAIN = 'proposed'
CID = 'candidate_id_TRACE_NOT_FEATURE'
INPUTS = {
    'perod': '26_连续需求_逐OD逐场景逐方法指标.csv',
    'switch': '27_经济舒适场景切换_逐OD五方法详细变化.csv',
    'conflict': '28_目标冲突OD_切换后实际改善统计.csv',
    'summary': '29_连续需求_五方法三场景质量与后悔值汇总.csv',
    'paired': '30_冲突OD_模型及多样性配对消融.csv',
    'audit': '31_连续需求_来源防泄漏与统计审计.json',
    'sensitivity': '34_目标冲突阈值敏感性分析.csv',
}
SHA_KEYS = {'perod':'perod', 'switch':'switch','conflict':'conflict_summary',
            'summary':'method_summary','paired':'paired','sensitivity':'sensitivity'}
M6_FILES = {
    'audit': '01_模块六_严格来源及覆盖审计.json',
    'scenarios': '02_全部OD_三场景五方法质量复算.csv',
    'conflict': '03_冲突OD_路线切换与道路属性变化_Bootstrap.csv',
    'switch': '04_经济舒适_各冲突层级路线切换统计.csv',
    'paired': '05_五方法_同OD配对检验及Holm校正.csv',
    'profile': '06_历史画像分层_有无画像与需求响应.csv',
    'sensitivity': '07_目标冲突阈值_稳健性复算.csv',
    'extremes': '08_强冲突OD_高绕行与转弯反例审计.csv',
    'paper': '09_论文核心结论指标表.csv',
    'notes': '10_实验口径与方法局限.json',
    'truth_perod': '20_最后3天实际路线_独立重合评价逐OD.csv.gz',
    'truth_summary': '21_最后3天实际路线_重合评价汇总.csv',
    'truth_audit': '22_实际路线独立评价_冻结与真值隔离审计.json',
}

# 指标数值均已由模块五计算；在模块六中重新从逐OD表聚合，不使用旧汇总表做“真值”。
METRIC_INFO = {
    'length_increase_km': ('舒适较经济多走距离', 'km', +1),
    'detour_increase': ('舒适较经济绕行比增量', 'ratio', +1),
    'turns_per_km_reduction': ('每公里转弯密度减少', 'turns/km', +1),
    'estimated_total_turns_reduction': ('估算全程转弯数减少', 'estimated turns', +1),
    'segment_density_reduction': ('每公里路段密度减少', 'segments/km', +1),
    'local_ratio_reduction': ('地方道路比例减少', 'share', +1),
    'motorway_ratio_increase': ('快速路比例增加', 'share', +1),
    'economic_proxy_change': ('经济代理分数改变', 'within-OD proxy', +1),
    'comfort_proxy_improvement': ('舒适代理分数提高', 'within-OD proxy', +1),
}
TOP_METRICS = {
    'top1_own_goal_regret': ('目标后悔值', -1),
    'top3_best_goal_regret': ('Top3最佳目标后悔值', -1),
    'top3_mean_pairwise_overlap': ('Top3平均道路重叠', -1),
    'top1_total_len_km': ('Top1长度', -1),
    'top1_turns_per_km': ('Top1转弯密度', -1),
    'top1_comfort_proxy_score': ('Top1舒适代理值', +1),
    'top1_economic_proxy_score': ('Top1经济代理值', +1),
}
PROFILE_COLUMNS = [f'history_{dim}_{i}' for dim in ('econ','comf') for i in range(4)]
PAPER_SCENE = {'economic':'经济优先','comfortable':'舒适优先','balanced':'综合均衡'}


def log(msg: str):
    print(f'[{datetime.now():%H:%M:%S}] {msg}',flush=True)


def required_file(path: Path, name: str):
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f'缺少{name}：{path}\n请确认模块五已执行 --phase demand-soft；不允许用模拟数据替代。')
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda: f.read(4*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def loadjson(path: Path) -> dict:
    return json.loads(required_file(path,'JSON').read_text(encoding='utf-8-sig'))


def readcsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(required_file(path,'CSV'),encoding='utf-8-sig',low_memory=False,
                       dtype={'trip_id':str,CID:str},**kwargs)


def require_cols(df: pd.DataFrame, columns, label: str):
    missing = sorted(set(columns)-set(df.columns))
    if missing:
        raise ValueError(f'{label} 缺少列：{missing}\n实际字段：{df.columns.tolist()}')


def assert_unique(df: pd.DataFrame, keys: list[str], label: str):
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        bad = df[df.duplicated(keys,keep=False)][keys].head(5).to_dict('records')
        raise ValueError(f'{label} 的唯一主键 {keys} 出现缺失/重复：{bad}')


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    df.to_csv(temp,index=False,encoding='utf-8-sig',float_format='%.12g',
              compression='gzip' if path.suffix == '.gz' else None)
    temp.replace(path)


def atomic_json(data: dict, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8-sig')
    temp.replace(path)


def safe_float(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):
        return None


def finite(df: pd.DataFrame, columns: list[str], label: str):
    for c in columns:
        vals = pd.to_numeric(df[c],errors='coerce').to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f'{label} 含缺失或非法数值：{c}')


def paths(a):
    f={k:a.m5/v for k,v in INPUTS.items()}
    f['coverage'] = a.m5/'05_全部测试OD_候选与推荐覆盖审计.csv'
    f['profile'] = a.m2/'05_最后3天盲推理8维画像_只按trip_id.csv'
    f['m4_model'] = a.m4/'06_正式模型参数与训练来源.json'
    f['m4_predictions'] = a.m4/'07_最后3天六模型逐路线盲排序.csv.gz'
    f['candidate_raw'] = a.m3/'01_真实自然候选路径_含有向边序列.csv.gz'
    f['blind_od'] = a.m1/'05_最后3天_纯OD推理输入_绝无真实路线.csv'
    f['truth'] = a.m1/'07_最后3天_真实路径答案_仅评价禁止推理.csv.gz'
    f['m5main'] = a.m5/'24_连续需求_三场景Top3主方法.csv.gz'
    f['m5ablation'] = a.m5/'25_连续需求_四类对照Top3.csv.gz'
    return f


def check_source(a, verbose=True):
    f=paths(a)
    for k in list(INPUTS)+['coverage','profile']:
        required_file(f[k],k)
    audit=loadjson(f['audit'])
    if audit.get('version') != 'chapter4_module5_v4_soft_demand_conflict_analysis':
        raise ValueError('模块五31号审计版本不符，不能读取v3或原版数据')
    if audit.get('mode') != 'FULL_REAL' or audit.get('status') != 'EXPLORATORY_TEST_OD_ALREADY_INSPECTED':
        raise ValueError('需要正式v4探索性全量输出，而非pilot或其他版本')
    if audit.get('module4_retrained') != 'NEVER' or audit.get('future_true_HMM_trajectory_read') != 'NEVER':
        raise ValueError('第五模块真值隔离或模型冻结审计失败')
    # 每个上传的统计文件必须和生成时冻结的哈希一致，防止混用多次运行的CSV。
    for key,sha_key in SHA_KEYS.items():
        expected = audit.get('new_output_sha256',{}).get(sha_key)
        if not expected or sha256(f[key]) != expected:
            raise ValueError(f'模块五输出文件被修改或混用：{f[key].name}')
    if int(audit.get('all_OD',-1)) != 4469 or int(audit.get('OD_with_natural_candidates',-1)) != 4466 \
       or int(audit.get('natural_candidates',-1)) != 37383:
        raise ValueError('第五模块4469/4466/37383正式样本规模与审计不一致')
    if abs(float(audit.get('continuous_demand_objective_weight',-1)) - 0.7)>1e-10:
        raise ValueError('与既有正式v4目标权重0.7不一致，请勿合并不同版本')
    # 模块四文件存在时必须验证冻结版本。缺失时“check/summarize”可以继续使用严格冻结的模块五文件。
    for key,expected in [('m4_model',audit.get('frozen_module4_model_sha256')),
                         ('m4_predictions',audit.get('frozen_module4_predictions_sha256'))]:
        if f[key].exists() and (not expected or sha256(f[key])!=expected):
            raise ValueError(f'模块四真实文件哈希与第五模块运行时不一致：{key}')
    if verbose:
        log('来源审计通过：第五模块v4正式结果，4,469请求OD，4,466有候选，37,383自然候选；本阶段绝不读取真实HMM轨迹')
    return f,audit


def read_base(a, f, audit):
    perod = readcsv(f['perod'])
    change = readcsv(f['switch'])
    coverage = readcsv(f['coverage'])
    profile = readcsv(f['profile'])
    require_cols(perod, ['trip_id','scenario','variant','candidate_count','n_recommended',
         'top1_candidate_id','strong_conflict','top1_own_goal_regret','top3_best_goal_regret',
         'top3_mean_pairwise_overlap','top1_total_len_km','top1_ratio_to_shortest',
         'top1_turns_per_km','top1_segment_density_seg_per_km','top1_ratio_local',
         'top1_ratio_motorway','top1_estimated_total_turns',
         'top1_economic_proxy_score','top1_comfort_proxy_score'], '26号逐OD')
    require_cols(change,['trip_id','variant','candidate_count','top1_switched',
         'strong_conflict','economic_top1_id','comfortable_top1_id','balanced_top1_id',
         'economic_total_len_km','comfortable_total_len_km']+list(METRIC_INFO),'27号切换')
    require_cols(coverage,['trip_id','natural_count'],'05号覆盖')
    require_cols(profile,['trip_id','history_n_prior','is_cold_start']+PROFILE_COLUMNS,'模块二历史画像')
    assert_unique(perod,['trip_id','scenario','variant'],'26号')
    assert_unique(change,['trip_id','variant'],'27号')
    assert_unique(coverage,['trip_id'],'05号')
    assert_unique(profile,['trip_id'],'模块二画像')
    if len(perod)!=int(audit['all_methods_OD_scenario_rows']) or len(change)!=int(audit['conflict_method_OD_rows']):
        raise ValueError('26/27行数不符合31号审计')
    if len(coverage)!=int(audit['all_OD']) or len(profile)!=int(audit['all_OD']):
        raise ValueError('05候选覆盖或模块二历史画像不是完整4469条OD')
    if set(perod.scenario.unique())!=set(SCENARIOS) or set(change.variant.unique())!=set(VARIANTS) \
       or set(perod.variant.unique())!=set(VARIANTS):
        raise ValueError('三场景或五方法缺失')
    if len(perod)!=4466*3*5 or len(change)!=4466*5:
        raise ValueError('五方法×三场景OD数量不匹配')
    sampled=set(perod.trip_id)
    if sampled != set(change.trip_id) or sampled != set(coverage.loc[coverage.natural_count>0,'trip_id']):
        raise ValueError('26/27可评分OD与05号覆盖集合不一致')
    if set(profile.trip_id)!=set(coverage.trip_id):
        raise ValueError('模块二画像与模块三OD不一致')
    for sc in SCENARIOS:
        for method in VARIANTS:
            n=int(((perod.scenario==sc)&(perod.variant==method)).sum())
            if n!=4466:raise ValueError(f'{sc}/{method} 缺OD：{n}')
    if int((pd.to_numeric(coverage.natural_count)<=0).sum())!=3:
        raise ValueError('无候选OD数量与正式审计不同')
    finite(perod,['top1_own_goal_regret','top3_best_goal_regret','top1_total_len_km',
          'top1_turns_per_km','top1_ratio_to_shortest','top1_economic_proxy_score',
          'top1_comfort_proxy_score'], '26号')
    finite(change, list(METRIC_INFO), '27号')
    if ((perod.top1_own_goal_regret<-1e-8)|(perod.top1_own_goal_regret>1+1e-8)).any():
        raise ValueError('目标后悔值超界')
    if (perod.n_recommended>perod.candidate_count).any() or (perod.n_recommended>3).any():
        raise ValueError('推荐数量超过候选或Top3限制')
    if ((perod.strong_conflict==1).sum()!=int(audit['strong_conflict_OD'])*3*5):
        raise ValueError('强冲突OD总数不符')
    v=change.loc[change.variant==MAIN]
    if len(v.loc[v.strong_conflict==1])!=int(audit['strong_conflict_OD']):
        raise ValueError('模块五强冲突OD计数不符')
    if not np.isclose(v.loc[v.strong_conflict==1,'top1_switched'].mean(),
                      float(audit['strong_conflict_top1_switch_fraction']),atol=1e-12):
        raise ValueError('模块五强冲突切换率统计与源审计不一致')
    # 27号配对字段与26号原始Top1重新比对，杜绝由列后缀/错配导致的统计偏差。
    ec=perod.loc[perod.scenario=='economic',['trip_id','variant','top1_candidate_id','top1_total_len_km',
                  'top1_turns_per_km','top1_estimated_total_turns','top1_economic_proxy_score']]
    cc=perod.loc[perod.scenario=='comfortable',['trip_id','variant','top1_candidate_id','top1_total_len_km',
                  'top1_turns_per_km','top1_estimated_total_turns','top1_comfort_proxy_score']]
    x=ec.merge(cc,on=['trip_id','variant'],suffixes=('_econ','_comf'),validate='one_to_one')
    chk=change.merge(x,on=['trip_id','variant'],validate='one_to_one')
    if (chk.economic_top1_id!=chk.top1_candidate_id_econ).any() or \
       (chk.comfortable_top1_id!=chk.top1_candidate_id_comf).any():
        raise ValueError('27号切换推荐路线ID与26号逐OD不一致')
    if not np.allclose(chk.length_increase_km.to_numpy(float),
                       chk.top1_total_len_km_comf.to_numpy(float)-chk.top1_total_len_km_econ.to_numpy(float),
                       atol=1e-8):
        raise ValueError('27号路线距离差与26号Top1不一致')
    if not np.allclose(chk.turns_per_km_reduction.to_numpy(float),
                       chk.top1_turns_per_km_econ.to_numpy(float)-chk.top1_turns_per_km_comf.to_numpy(float),
                       atol=1e-8):
        raise ValueError('27号转弯密度差与26号Top1不一致')
    profile=profile.copy()
    profile['history_n_prior']=pd.to_numeric(profile.history_n_prior,errors='coerce')
    profile['is_cold_start']=pd.to_numeric(profile.is_cold_start,errors='coerce')
    if profile[['history_n_prior','is_cold_start']].isna().any().any():
        raise ValueError('模块二历史画像冷启动标记和历史样本数无效')
    if (profile.history_n_prior<0).any() or not profile.is_cold_start.isin([0,1]).all():
        raise ValueError('模块二历史画像历史样本量或冷启动标签非法')
    if not np.allclose(profile[PROFILE_COLUMNS].apply(pd.to_numeric,errors='coerce').sum(axis=1).to_numpy(),2.,atol=1e-4):
        raise ValueError('8维画像两套概率权重之和不等于2，请检查模块二来源')
    return perod,change,coverage,profile


def boot_ci(x, n_boot:int, rng:np.random.Generator):
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)<2:
        return (None,None)
    out=np.empty(n_boot)
    for i in range(0,n_boot,64):
        k=min(64,n_boot-i)
        out[i:i+k]=x[rng.integers(0,len(x),size=(k,len(x)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(out,[.025,.975]))


def sign_flip_p(x, n_perm, rng):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if len(x)<2 or not np.any(np.abs(x)>1e-12):return 1.
    target=abs(x.mean())
    count=0
    # 随机符号翻转：成对零假设下差值关于0对称，属于探索性检验。
    for i in range(0,n_perm,64):
        k=min(64,n_perm-i)
        signs=rng.choice([-1.,1.],size=(k,len(x)))
        count+=int((np.abs((signs*x).mean(axis=1)) >= target-1e-14).sum())
    return (count+1)/(n_perm+1)


def holm(pvals):
    p=np.asarray(pvals,dtype=float)
    n=len(p)
    order=np.argsort(p)
    out=np.empty(n)
    m=0.
    for rank,pos in enumerate(order):
        m=max(m,float(p[pos])*(n-rank))
        out[pos]=min(1.,m)
    return out.tolist()


def grouped_switch(change):
    cc=change.loc[change.variant==MAIN]
    return {
        'all_scored_OD':cc,
        'strong_conflict_OD':cc.loc[cc.strong_conflict==1],
        'strong_conflict_switched_OD':cc.loc[(cc.strong_conflict==1)&(cc.top1_switched==1)],
        'strong_conflict_not_switched_OD':cc.loc[(cc.strong_conflict==1)&(cc.top1_switched==0)],
        'non_strong_conflict_OD':cc.loc[cc.strong_conflict==0],
    }


def build_tables(a, perod, change, coverage, profile, audit):
    rng=np.random.default_rng(a.seed)
    scored_od=len(set(perod.trip_id))
    full_od=len(coverage)
    # 每个场景/方法/人群分别计算Top1与Top3指标，并显示分母。
    summary=[]
    fields=['top1_own_goal_regret','top3_best_goal_regret','top1_total_len_km','top1_ratio_to_shortest',
            'top1_turns_per_km','top1_segment_density_seg_per_km','top1_ratio_local',
            'top1_ratio_motorway','top1_estimated_total_turns','top1_economic_proxy_score',
            'top1_comfort_proxy_score','top3_mean_length_km','top3_mean_economic_proxy',
            'top3_mean_comfort_proxy','top3_mean_pairwise_overlap']
    for (sc,method),grp in perod.groupby(['scenario','variant'],sort=True):
        for name, subset in [('all_scored_OD',grp),('strong_conflict_OD',grp.loc[grp.strong_conflict==1]),
                            ('non_strong_conflict_OD',grp.loc[grp.strong_conflict==0])]:
            row={'scenario':sc,'variant':method,'population':name,
                 'n_all_requested_od':full_od,'n_scored_od':len(subset),
                 'coverage_rate_of_all_requested_od':len(subset)/full_od,
                 'n_top3_od':int((subset.n_recommended>=3).sum())}
            for k in fields:
                row[f'mean_{k}']=safe_float(pd.to_numeric(subset[k],errors='coerce').mean())
            summary.append(row)
    summary=pd.DataFrame(summary)
    # 强冲突配对差异的均值、置信区间与方向比例，并保留零差值（未切换）。
    differences=[]
    for method,g in change.groupby('variant',sort=True):
        groups={
            'all_scored_OD':g,
            'strong_conflict_OD':g.loc[g.strong_conflict==1],
            'strong_conflict_switched_OD':g.loc[(g.strong_conflict==1)&(g.top1_switched==1)],
            'strong_conflict_not_switched_OD':g.loc[(g.strong_conflict==1)&(g.top1_switched==0)],
            'non_strong_conflict_OD':g.loc[g.strong_conflict==0],
        }
        for label,ss in groups.items():
            for name,(description,unit,_direction) in METRIC_INFO.items():
                vals=ss[name].to_numpy(float)
                lo,hi=boot_ci(vals,a.bootstrap,rng)
                differences.append({
                    'variant':method,'population':label,'metric':name,
                    'metric_description':description,'unit':unit,'n_od':len(vals),
                    'n_switch':int(ss.top1_switched.sum()),
                    'switch_rate':safe_float(ss.top1_switched.mean()),
                    'mean':safe_float(vals.mean()) if len(vals) else None,
                    'median':safe_float(np.median(vals)) if len(vals) else None,
                    'q25':safe_float(np.quantile(vals,.25)) if len(vals) else None,
                    'q75':safe_float(np.quantile(vals,.75)) if len(vals) else None,
                    'bootstrap_ci95_low':lo,'bootstrap_ci95_high':hi,
                    'positive_share':safe_float(np.mean(vals>1e-10)) if len(vals) else None,
                    'negative_share':safe_float(np.mean(vals<-1e-10)) if len(vals) else None,
                    'zero_share':safe_float(np.mean(np.abs(vals)<=1e-10)) if len(vals) else None,
                })
    differences=pd.DataFrame(differences)
    switch=[]
    for method,g in change.groupby('variant',sort=True):
        for pop,sg in [('all_scored_OD',g),('strong_conflict_OD',g[g.strong_conflict==1]),
                       ('non_strong_conflict_OD',g[g.strong_conflict==0])]:
            if len(sg)==0:continue
            switch.append({'variant':method,'population':pop,'n_od':len(sg),
                'n_econ_comfort_top1_switched':int(sg.top1_switched.sum()),
                'econ_comfort_top1_switch_rate':float(sg.top1_switched.mean()),
                'all_three_top1_equal_rate':float(sg.top1_all_three_equal.mean()),
                'mean_cross_scenario_top1_edge_jaccard':float(sg.economic_vs_comfort_top1_jaccard.mean()),
                'mean_cross_scenario_top3_candidate_jaccard':float(sg.economic_vs_comfort_top3_set_jaccard.mean())})
    switch=pd.DataFrame(switch)
    # 正确的对照：在同一OD、同一场景配对比较，逐指标用Holm校正四组消融比较。
    pairs=[]
    for sc in SCENARIOS:
        ps=perod[perod.scenario==sc]
        for pop in ('all_scored_OD','strong_conflict_OD'):
            sg=ps if pop=='all_scored_OD' else ps[ps.strong_conflict==1]
            ref=sg[sg.variant==MAIN].set_index('trip_id')
            for target in ('no_profile','objective_only','no_diversity','utility_only'):
                other=sg[sg.variant==target].set_index('trip_id')
                if set(ref.index)!=set(other.index):
                    raise ValueError(f'配对比较OD集合不相同：{sc}/{pop}/{target}')
                other=other.loc[ref.index]
                same=(ref.top1_candidate_id.values==other.top1_candidate_id.values)
                for metric,(description,direction) in TOP_METRICS.items():
                    dx=(ref[metric]-other[metric]).to_numpy(float)
                    lo,hi=boot_ci(dx,a.bootstrap,rng)
                    # 有益方向统一为“正值越大越有利”；因Regret更小更好乘负号。
                    signed=dx*direction
                    pairs.append({'scenario':sc,'population':pop,'comparison':f'proposed_minus_{target}',
                        'metric':metric,'metric_name':description,
                        'n_pairs':len(dx),'mean_raw_difference':float(dx.mean()),
                        'mean_benefit_direction':float(signed.mean()),
                        'bootstrap_mean_difference_ci_low':lo,'bootstrap_mean_difference_ci_high':hi,
                        'proposed_better_fraction':float((signed>1e-10).mean()),
                        'same_top1_id_fraction':float(same.mean()),
                        'p_sign_flip_exploratory':sign_flip_p(dx,a.permutations,rng)})
    pairs=pd.DataFrame(pairs)
    # 对每一场景×人群×评价指标的四个方法比较做Holm校正，不混淆跨指标检验家族。
    pairs['p_holm']=np.nan
    for _,grp in pairs.groupby(['scenario','population','metric']):
        pairs.loc[grp.index,'p_holm']=holm(grp.p_sign_flip_exploratory.values)
    pairs['holm_significant_005_exploratory']=pairs.p_holm<.05
    # 历史画像冷启动与样本量分层。注意“分层”不是驾驶员/乘客身份或因果推断。
    profile=profile[['trip_id','history_n_prior','is_cold_start']+PROFILE_COLUMNS].copy()
    count=profile.history_n_prior.astype(float)
    profile['history_group']=np.select([count.eq(0),count.between(1,5),count.between(6,20),count.gt(20)],
                                       ['0_无历史','1_1到5条','2_6到20条','3_超过20条'],default='UNASSIGNED')
    grp_count=profile.history_group.value_counts().to_dict()
    profile_merged=perod.merge(profile[['trip_id','history_group','history_n_prior','is_cold_start']],
                               on='trip_id',how='inner',validate='many_to_one')
    rows_profile=[]
    for (sc,hist),group in profile_merged.groupby(['scenario','history_group'],sort=True):
        pp=group[group.variant==MAIN].set_index('trip_id')
        npf=group[group.variant=='no_profile'].set_index('trip_id')
        if set(pp.index)!=set(npf.index):raise ValueError('分层历史画像的OD配对不一致')
        npf=npf.loc[pp.index]
        base={'scenario':sc,'history_group':hist,'n_od':len(pp),
              'n_all_history_group':int(grp_count.get(hist,0)),
              'same_top1_share':float((pp.top1_candidate_id==npf.top1_candidate_id).mean())}
        for metric,(_,direction) in TOP_METRICS.items():
            dx=(pp[metric]-npf[metric]).to_numpy(float)
            lo,hi=boot_ci(dx,a.bootstrap,rng)
            base[f'delta_{metric}_proposed_minus_no_profile']=float(dx.mean())
            base[f'ci95_low_{metric}']=lo
            base[f'ci95_high_{metric}']=hi
        rows_profile.append(base)
    profile_table=pd.DataFrame(rows_profile)
    # 审查容易造成夸大论述的极端样本，保留所有OD，不偷偷剔除离群点。
    hot=change[(change.variant==MAIN)&(change.strong_conflict==1)].copy()
    hot['excess_length_km']=hot.length_increase_km
    hot['more_total_turns_flag']=(hot.estimated_total_turns_reduction<-1e-10).astype(int)
    hot['more_turns_per_km_flag']=(hot.turns_per_km_reduction<-1e-10).astype(int)
    hot['very_short_route_flag']=(hot[['economic_total_len_km','comfortable_total_len_km']].min(axis=1)<.5).astype(int)
    hot['high_excess_length_flag']=(hot.excess_length_km>10).astype(int)
    extreme=hot[['trip_id','candidate_count','top1_switched','strong_conflict','economic_top1_id',
                 'comfortable_top1_id','economic_total_len_km','comfortable_total_len_km',
                 'length_increase_km','detour_increase','turns_per_km_reduction',
                 'estimated_total_turns_reduction','segment_density_reduction','local_ratio_reduction',
                 'motorway_ratio_increase','comfort_proxy_improvement','economic_proxy_change',
                 'very_short_route_flag','high_excess_length_flag','more_total_turns_flag',
                 'more_turns_per_km_flag']].sort_values('length_increase_km',ascending=False)
    # 固定阈值敏感性：使用第五模块已经冻结的交叉目标损失，绝不据结果优化阈值。
    normal=change[change.variant==MAIN].copy()
    sensitivity=[]
    for gap in (.10,.15,.20):
        selected=normal[(normal.different_objective_best==1)&
            (normal.best_econ_vs_best_comfort_economic_gap>=gap-1e-12)&
            (normal.best_econ_vs_best_comfort_comfort_gap>=gap-1e-12)]
        for kind,ss in [('all_conflict_OD',selected),('switched_conflict_OD',selected[selected.top1_switched==1])]:
            row={'conflict_gap':gap,'population':kind,'n_od':len(ss),'n_all_scored_od':scored_od,
                 'switch_rate':safe_float(ss.top1_switched.mean())}
            for name in METRIC_INFO:row['mean_'+name]=safe_float(ss[name].mean())
            sensitivity.append(row)
    sensitivity=pd.DataFrame(sensitivity)
    # 核验28/29/34号先前汇总的关键结果，确保本模块复算与旧表一致。
    old28=readcsv(a.m5/INPUTS['conflict'])
    old29=readcsv(a.m5/INPUTS['summary'])
    old34=readcsv(a.m5/INPUTS['sensitivity'])
    key28=old28[(old28.variant==MAIN)&(old28.population=='strong_conflict_OD')]
    for metric in METRIC_INFO:
        old=key28.loc[key28.metric==metric,'mean']
        new=differences[(differences.variant==MAIN)&(differences.population=='strong_conflict_OD')&
                        (differences.metric==metric)]['mean']
        if len(old)!=1 or len(new)!=1 or not np.isclose(float(old.iloc[0]),float(new.iloc[0]),atol=1e-8):
            raise ValueError(f'28号和新复算数据不一致：{metric}')
    oldtop=old29[(old29.variant==MAIN)&(old29.population=='strong_conflict_OD')]
    for sc in SCENARIOS:
        old=oldtop.loc[oldtop.scenario==sc,'mean_top1_length_km']
        new=summary[(summary.variant==MAIN)&(summary.population=='strong_conflict_OD')&
                    (summary.scenario==sc)]['mean_top1_total_len_km']
        if len(old)!=1 or len(new)!=1 or not np.isclose(float(old.iloc[0]),float(new.iloc[0]),atol=1e-8):
            raise ValueError(f'29号与新复算长度不一致：{sc}')
    for _,r in old34[old34.population=='all_conflict_OD'].iterrows():
        vals=sensitivity[(sensitivity.conflict_gap==float(r.conflict_gap))&
                         (sensitivity.population=='all_conflict_OD')]
        if len(vals)!=1 or int(vals.iloc[0].n_od)!=int(r.n_od):
            raise ValueError('34号冲突阈值OD数量与复算不同')
    # 论文核心展示表：只写真实结果和透明分母，不以阈值命中率冒充预测准确率。
    paper=[]
    keypop=grouped_switch(change)
    for pop in ('all_scored_OD','strong_conflict_OD','strong_conflict_switched_OD'):
        df=keypop[pop]
        paper.append({'section':'场景需求响应','population':pop,'metric':'economic_vs_comfort_top1_switch_rate',
                      'unit':'fraction','n_od':len(df),'value':float(df.top1_switched.mean())})
        for key in ('length_increase_km','turns_per_km_reduction','estimated_total_turns_reduction',
                    'segment_density_reduction','local_ratio_reduction','motorway_ratio_increase',
                    'comfort_proxy_improvement','economic_proxy_change'):
            val=differences[(differences.variant==MAIN)&(differences.population==pop)&
                            (differences.metric==key)].iloc[0]
            paper.append({'section':'舒适相对经济的配对权衡','population':pop,'metric':key,
                          'unit':val.unit,'n_od':int(val.n_od),'value':val['mean'],
                          'ci95_low':val.bootstrap_ci95_low,'ci95_high':val.bootstrap_ci95_high})
    for sc in SCENARIOS:
        for k in ('top1_own_goal_regret','top3_mean_pairwise_overlap'):
            val=summary[(summary.variant==MAIN)&(summary.population=='all_scored_OD')&
                        (summary.scenario==sc)].iloc[0]
            paper.append({'section':sc,'population':'all_scored_OD','metric':k,
                          'unit':'relative','n_od':int(val.n_scored_od),'value':val['mean_'+k]})
    paper=pd.DataFrame(paper)
    return {'scenarios':summary,'conflict':differences,'switch':switch,'paired':pairs,
            'profile':profile_table,'sensitivity':sensitivity,'extremes':extreme,'paper':paper}


def plot_figures(tables, out:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for font in ('Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans'):
        if font in fonts:
            plt.rcParams['font.family']=font
            break
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['figure.dpi']=140
    # 基于真实平均值，用绝对单位对比，而非以很小基数产生的夸张增长百分比。
    df=tables['scenarios']
    arr=df[(df.variant==MAIN)&(df.population=='strong_conflict_OD')]
    groups=[('mean_top1_total_len_km','Length (km)'),
            ('mean_top1_turns_per_km','Turns per km'),
            ('mean_top1_estimated_total_turns','Estimated total turns'),
            ('mean_top1_ratio_motorway','Motorway share')]
    fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,(key,label) in zip(axs.flat,groups):
        vals=[]
        for sc in ('economic','comfortable'):
            vv=arr[arr.scenario==sc]
            vals.append(float(vv.iloc[0][key]))
        ax.bar(['Economic-first','Comfort-first'],vals,color=['#547da9','#248d7b'],width=.54)
        ax.set_title(label,fontsize=11)
        for i,v in enumerate(vals): ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
        ax.margins(y=.16)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Strong-conflict ODs: actual road-attribute trade-off (n={int(arr.n_scored_od.iloc[0]):,})',fontsize=12)
    fig.savefig(out/'11_强冲突OD_经济舒适Top1真实道路属性对照.png',dpi=230,bbox_inches='tight');plt.close(fig)
    # 路线长度与转弯密度二维权衡：原始散点+局部可读范围，并显示离群比例。
    hot=tables['extremes']
    x=hot.length_increase_km.to_numpy(float)
    y=hot.turns_per_km_reduction.to_numpy(float)
    fig,ax=plt.subplots(figsize=(8.6,5.8))
    ax.scatter(x,y,s=11,alpha=.35,edgecolor='none',color='#377ba3',rasterized=True)
    ax.axhline(0,color='#777777',lw=1,ls='--');ax.axvline(0,color='#777777',lw=1,ls='--')
    ax.set_xlabel('Added distance when choosing comfort-first (km)')
    ax.set_ylabel('Reduction in turns per km')
    ax.set_title(f'Strong-conflict ODs: route length–turn density trade-off (n={len(hot):,})')
    fig.savefig(out/'12_强冲突OD_路线长度与转弯密度权衡_全样本.png',dpi=220,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6,5.8))
    ax.scatter(x,y,s=12,alpha=.35,edgecolor='none',color='#377ba3',rasterized=True)
    ax.axhline(0,color='#777777',lw=1,ls='--');ax.axvline(0,color='#777777',lw=1,ls='--')
    # 局部放大范围由真实分位数决定，原始全样本图和原始CSV仍保留。
    xmin,xmax=np.quantile(x,[.01,.99]);ymin,ymax=np.quantile(y,[.01,.99])
    if xmax>xmin:ax.set_xlim(xmin-.03*(xmax-xmin),xmax+.03*(xmax-xmin))
    if ymax>ymin:ax.set_ylim(ymin-.03*(ymax-ymin),ymax+.03*(ymax-ymin))
    ax.set_xlabel('Added distance (km)');ax.set_ylabel('Reduction in turns per km')
    ax.set_title('Same data, view limited to 1st–99th percentiles')
    fig.savefig(out/'13_强冲突OD_长度转弯权衡_分位数局部放大.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # 95%区间：仅画具有明确方向含义的配对差值，符号保留，单位分开。
    selected=tables['conflict']
    sel=selected[(selected.variant==MAIN)&(selected.population=='strong_conflict_OD')]
    metrics=['length_increase_km','turns_per_km_reduction','segment_density_reduction',
             'motorway_ratio_increase','estimated_total_turns_reduction','comfort_proxy_improvement']
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,k in zip(axes.flat,metrics):
        row=sel.loc[sel.metric==k].iloc[0]
        ax.errorbar([float(row['mean'])],[0],
                    xerr=np.array([[float(row['mean'])-float(row.bootstrap_ci95_low)],
                                   [float(row.bootstrap_ci95_high)-float(row['mean'])]]),
                    fmt='o',capsize=4,color='#2c739c')
        ax.axvline(0,ls='--',c='#777777',lw=1)
        ax.set_title(k.replace('_',' '),fontsize=10)
        ax.set_yticks([]);ax.set_xlabel(str(row.unit),fontsize=9)
        ax.spines[['left','right','top']].set_visible(False)
    fig.suptitle(f'Paired changes on {int(sel.n_od.iloc[0]):,} strong-conflict ODs (bootstrap 95% CI)',fontsize=11)
    fig.savefig(out/'14_强冲突OD_配对差值与置信区间.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # 冲突阈值0.1/0.15/0.2敏感性图
    d=tables['sensitivity'];d=d[d.population=='all_conflict_OD'].sort_values('conflict_gap')
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.plot(d.conflict_gap,d.switch_rate*100,marker='o',lw=2,color='#2d7c90')
    for r in d.itertuples(index=False):
        ax.annotate(f'n={int(r.n_od):,}',(r.conflict_gap,100*r.switch_rate),xytext=(0,9),
                    textcoords='offset points',ha='center',fontsize=9)
    ax.set_xlabel('Strong-conflict threshold (within-OD proxy gap)')
    ax.set_ylabel('Economic vs comfort Top-1 switch rate (%)')
    ax.set_title('Threshold sensitivity (predefined 0.10, 0.15, 0.20)')
    ax.set_ylim(max(0,float((d.switch_rate*100).min())-10),min(100,float((d.switch_rate*100).max())+10))
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'15_目标冲突阈值敏感性.png',dpi=220,bbox_inches='tight');plt.close(fig)
    return 5


def summarize(a, f, audit):
    if (a.out/M6_FILES['truth_audit']).exists() and not a.overwrite_truth:
        raise FileExistsError('此前已完成真实路线独立评价。为保护冻结链条，请使用 --phase plot 查看图，或明确 --overwrite-truth 后再重新执行 summarize 和 evaluate-truth。')
    begin=time.perf_counter()
    perod,change,coverage,profile=read_base(a,f,audit)
    tables=build_tables(a,perod,change,coverage,profile,audit)
    a.out.mkdir(parents=True,exist_ok=True)
    for k,df in tables.items():
        atomic_csv(df,a.out/M6_FILES[k])
    source={k:{'path':str(p),'sha256':sha256(p)}
            for k,p in f.items() if k in list(INPUTS)+['coverage','profile']}
    frozen={
        'version':'module6_v4_conflict_and_profile_evaluation_1.0',
        'created_at':datetime.now().isoformat(),
        'phase':'summarize_no_future_truth',
        'source_mode':'FULL_REAL_EXPLORATORY_TEST_ALREADY_INSPECTED',
        'upstream_module5_v4_version':audit['version'],
        'all_od':len(coverage),'scored_od':perod.trip_id.nunique(),
        'no_natural_candidate_od':int((coverage.natural_count==0).sum()),
        'strong_conflict_od':int(((change.variant==MAIN)&(change.strong_conflict==1)).sum()),
        'methods':list(VARIANTS),'scenarios':list(SCENARIOS),
        'main_method':MAIN,'bootstrap_resamples':a.bootstrap,'permutation_resamples':a.permutations,
        'random_seed':a.seed,'input_sources':source,
        'upstream_audit_sha256':sha256(f['audit']),
        'm4_model_sha256_from_v4':audit['frozen_module4_model_sha256'],
        'm4_predictions_sha256_from_v4':audit['frozen_module4_predictions_sha256'],
        'derived_tables_sha256':{k:sha256(a.out/M6_FILES[k]) for k in tables},
        'future_real_HMM_route_read':'NEVER',
        'new_routes_created':'NEVER',
        'model_retrained':'NEVER',
        'elapsed_seconds':round(time.perf_counter()-begin,2),
    }
    atomic_json(frozen,a.out/M6_FILES['audit'])
    notes={
        'scope':'模块五v4已生成的三场景Top3在同OD配对下的属性和多样性比较；不是训练新的推荐模型。',
        'confidence_intervals':'以OD为重抽样单位的percentile Bootstrap，非车辆聚类Bootstrap。一个车辆可能对应多个OD，应避免声称严格独立的统计推断。',
        'paired_tests':'配对随机符号翻转与同一指标下四类方法比较的Holm校正，仅探索性。',
        'not_prediction_accuracy':'不使用硬需求规则100%符合率作为预测准确率；没有乘客真实经济/舒适标签。',
        'no_objective_value_fabrication':'经济代理为同OD相对路径长度，舒适代理为静态道路结构，不是实测油耗或乘坐感受。',
        'total_turns':'estimated_total_turns=turns_per_km*length_km，不是GPS观测的真实转弯事件。',
        'interpret_switch':'切换率表示用户需求变化引起的算法响应；不是预测真实司机下一次将走哪条路的成功率。',
        'profile_scope':'8维软画像来自车辆历史轨迹，不能据此推断司机身份固定、稳定个体意图或单独归因于UDPPM++因果贡献。',
        'profile_ablation':'no_profile以冻结PSL_LengthWeighted概率替换含画像的模型，效用模型本身也发生变化，不是完全纯净的画像删除试验。',
        'test_reuse':'第五模块v4在查看相同后三天测试OD的旧结果后调整，分析需描述为探索性，不可称从未使用过的独立外部测试。',
        'semantic_scope':'上海BERT+KG路网语义未自动输入曼谷此版模块四/五；不能声称推荐性能来自上海语义向量。',
        'no_future_truth_read_in_summarize':True,
        'ground_truth_phase':'仅 --phase evaluate-truth 读取未来真实HMM边序列，且只做历史实际路线重合度辅助比较。'
    }
    atomic_json(notes,a.out/M6_FILES['notes'])
    log('第六模块无真值统计完成。核心汇总：')
    main=change[(change.variant==MAIN)&(change.strong_conflict==1)]
    log(f'强冲突OD={len(main):,}，经济对舒适Top1切换={int(main.top1_switched.sum()):,}/{len(main):,}（{main.top1_switched.mean():.2%}）')
    log(f'平均长度变化={main.length_increase_km.mean():+.4f} km，转弯密度减少={main.turns_per_km_reduction.mean():+.4f} 次/km，估计总转弯数减少={main.estimated_total_turns_reduction.mean():+.4f}')
    log('原模型、候选路径、最后3天真实HMM真值均未修改或读取')
    return tables


def check_frozen_for_truth(a,f,audit):
    p=a.out/M6_FILES['audit']
    frozen=loadjson(p)
    if frozen.get('phase')!='summarize_no_future_truth' or frozen.get('future_real_HMM_route_read')!='NEVER':
        raise ValueError('必须先运行 --phase summarize 冻结无真值评价')
    if frozen.get('upstream_audit_sha256')!=sha256(f['audit']):
        raise ValueError('第五模块31号审计与已冻结的第六模块不一致')
    for key,meta in frozen['input_sources'].items():
        if sha256(Path(meta['path']))!=meta['sha256']:
            raise ValueError(f'冻结后输入被更改：{key}')
    for key,digest in frozen['derived_tables_sha256'].items():
        if sha256(a.out/M6_FILES[key])!=digest:
            raise ValueError(f'无真值复算结果冻结后发生变化：{key}')
    # 真值评价额外依赖已冻结的第5模块Top3顺序；不可将早期v1/v3的推荐混入。
    for key,sha_key in [('m5main','main'),('m5ablation','ablations')]:
        required_file(f[key],key)
        if sha256(f[key])!=audit.get('new_output_sha256',{}).get(sha_key):
            raise ValueError(f'Top3推荐来源混用或变化：{key}')
    required_file(f['candidate_raw'],'模块三原始自然候选')
    required_file(f['blind_od'],'模块一纯OD')
    required_file(f['truth'],'模块一留出的实际HMM真值')
    required_file(a.graphml,'固定GraphML路网')
    # 与源审计里的旧模块五哈希比较路线来源和路网版本。
    old_audit=loadjson(a.m5/'08_模块五_完整性盲测与来源审计.json')
    expected_raw=old_audit.get('input_sources',{}).get('raw',{}).get('sha256')
    expected_graph=old_audit.get('input_sources',{}).get('graph',{}).get('sha256')
    if not expected_raw or sha256(f['candidate_raw'])!=expected_raw:
        raise ValueError('模块三路径边序列与模块五冻结使用的候选数据不一致')
    if not expected_graph or sha256(a.graphml)!=expected_graph:
        raise ValueError('GraphML路网与原来候选生成版本不同')
    return frozen


def parse_edges(raw):
    if pd.isna(raw):raise ValueError('道路序列为空')
    try: data=json.loads(str(raw))
    except json.JSONDecodeError as e:raise ValueError(f'道路序列不是合法JSON：{e}') from e
    if not isinstance(data,list) or not data:raise ValueError('道路边序列必须是非空列表')
    ret=[]
    for e in data:
        if not isinstance(e,(list,tuple)) or len(e)!=3 or any(v is None for v in e):
            raise ValueError('边必须是(u,v,key)三元组')
        ret.append(tuple(map(str,e)))
    if any(a[1]!=b[0] for a,b in zip(ret,ret[1:])):
        raise ValueError('有向道路边序列不连续')
    return ret


def load_edge_lengths(graphml:Path, wanted:set[tuple[str,str,str]]):
    import networkx as nx
    log('正在读取GraphML以建立真值/候选道路长度字典……（仅在真值评价阶段）')
    graph=nx.read_graphml(str(graphml),force_multigraph=True)
    if not graph.is_directed():raise ValueError('要求有向多重路网')
    lengths={}
    for u,v,k,d in graph.edges(keys=True,data=True):
        e=(str(u),str(v),str(k))
        if e in wanted:
            val=safe_float(d.get('length'))
            if val is not None and val>0:lengths[e]=val
    del graph
    return lengths


def evaluate_truth(a,f,audit):
    # 完成所有冻结与哈希检查后，才允许读取最后3天的历史真实边序列。
    frozen=check_frozen_for_truth(a,f,audit)
    if (a.out/M6_FILES['truth_audit']).exists() and not a.overwrite_truth:
        raise FileExistsError('真值评价已经存在；如确需重算，请显式添加 --overwrite-truth')
    log('无真值结果已冻结。现在开始只供评价使用的真实HMM路线匹配。')
    true=readcsv(f['truth'])
    raw=readcsv(f['candidate_raw'])
    blind=readcsv(f['blind_od'])
    require_cols(true,['trip_id','origin_node','destination_node','edge_seq_uvk_json'],'真实答案')
    require_cols(raw,['trip_id',CID,'origin_node','destination_node','edge_seq_uvk_json'],'自然候选')
    require_cols(blind,['trip_id','origin_node','destination_node'],'盲OD')
    assert_unique(true,['trip_id'],'真值');assert_unique(raw,[CID],'自然候选')
    assert_unique(blind,['trip_id'],'盲OD')
    if len(true)!=4469 or len(blind)!=4469 or len(raw)!=37383:
        raise ValueError('真值/候选/盲OD数量与正式来源不一致')
    if set(true.trip_id)!=set(blind.trip_id):raise ValueError('真值OD集合与盲OD集合不相同')
    if not true.set_index('trip_id')[['origin_node','destination_node']].astype(str).sort_index().equals(
           blind.set_index('trip_id')[['origin_node','destination_node']].astype(str).sort_index()):
        raise ValueError('真实OD与盲OD起终节点不匹配')
    seqs={}; wanted=set()
    for row in raw.itertuples(index=False):
        cid=str(getattr(row,CID));seq=parse_edges(row.edge_seq_uvk_json)
        if seq[0][0]!=str(row.origin_node) or seq[-1][1]!=str(row.destination_node):
            raise ValueError('候选路径OD端点不符：'+cid)
        seqs[cid]=seq;wanted.update(seq)
    real={};invalid={}
    for row in true.itertuples(index=False):
        tid=str(row.trip_id)
        try:
            seq=parse_edges(row.edge_seq_uvk_json)
            if seq[0][0]!=str(row.origin_node) or seq[-1][1]!=str(row.destination_node):
                raise ValueError('真实HMM边序列起终节点不符')
            real[tid]=seq;wanted.update(seq)
        except (TypeError,ValueError) as err:
            invalid[tid]=str(err)
    lengths=load_edge_lengths(a.graphml,wanted)
    missing_candidate=[(cid,e) for cid,seq in seqs.items() for e in set(seq) if e not in lengths]
    if missing_candidate:raise ValueError('GraphML找不到候选路线边/长度，拒绝计算：'+str(missing_candidate[0]))
    for tid,seq in real.items():
        missing=[e for e in set(seq) if e not in lengths]
        if missing:invalid[tid]=f'真实路径包含GraphML中找不到的边，示例{missing[0]}'
    selected1=readcsv(f['m5main'])
    selected2=readcsv(f['m5ablation'])
    require_cols(selected1,['trip_id','scenario','variant','rank_in_scenario',CID],'主方法Top3')
    require_cols(selected2,['trip_id','scenario','variant','rank_in_scenario',CID],'消融Top3')
    sel=pd.concat([selected1,selected2],ignore_index=True)
    assert_unique(sel,['trip_id','scenario','variant','rank_in_scenario'],'第五模块Top3')
    assert_unique(sel,['trip_id','scenario','variant',CID],'第五模块Top3')
    if set(sel[CID])-set(seqs):raise ValueError('第五模块Top3存在外部候选ID')
    if set(sel.scenario.unique())!=set(SCENARIOS) or set(sel.variant.unique())!=set(VARIANTS):
        raise ValueError('第五模块Top3缺少场景或方法')
    coverage=readcsv(f['coverage'])
    scored=set(raw.trip_id)
    # 真值的评价单元与本次需求实验保持一致：OD×场景×方法。
    results=[]
    groups={(t,s,v):list(g.sort_values('rank_in_scenario')[CID]) for (t,s,v),g in \
            sel.groupby(['trip_id','scenario','variant'],sort=False)}
    for tid in sorted(set(blind.trip_id)):
        seq=real.get(tid)
        valid=tid not in invalid
        for sc in SCENARIOS:
            for method in VARIANTS:
                chosen=groups.get((tid,sc,method),[])
                if tid in scored and not chosen:
                    raise ValueError(f'{tid}/{sc}/{method} 有自然候选却没有Top3推荐')
                row={'trip_id':tid,'scenario':sc,'variant':method,'n_recommended':len(chosen),
                     'truth_valid':int(valid),'invalid_truth_reason':invalid.get(tid,''),
                     'exact_hit1':None,'exact_hit3':None,
                     'directed_edge_recall_top1':None,'directed_edge_recall_best_top3':None,
                     'length_jaccard_top1':None,'length_jaccard_best_top3':None}
                if valid:
                    truth_set=set(seq)
                    true_len=sum(lengths[e] for e in truth_set)
                    if true_len<=0:raise ValueError('真实路线边长合计为0：'+tid)
                    exact=[];recall=[];js=[]
                    for cid in chosen:
                        candidate=seqs[cid];candset=set(candidate)
                        both=truth_set&candset
                        share=sum(lengths[e] for e in both)
                        union=sum(lengths[e] for e in truth_set|candset)
                        exact.append(int(candidate==seq))
                        recall.append(share/true_len)
                        js.append(share/union if union>0 else 0.)
                    row.update({
                        'exact_hit1':exact[0] if exact else 0,
                        'exact_hit3':max(exact,default=0),
                        'directed_edge_recall_top1':recall[0] if recall else 0.,
                        'directed_edge_recall_best_top3':max(recall,default=0.),
                        'length_jaccard_top1':js[0] if js else 0.,
                        'length_jaccard_best_top3':max(js,default=0.),
                    })
                results.append(row)
    result=pd.DataFrame(results)
    if len(result)!=4469*15:raise ValueError('真值逐OD评价数量不符')
    selected_change=readcsv(f['switch'])
    flag=selected_change.loc[selected_change.variant==MAIN,['trip_id','strong_conflict']]
    result=result.merge(flag,on='trip_id',how='left',validate='many_to_one')
    result['strong_conflict']=result.strong_conflict.fillna(0).astype(int)
    summaries=[]
    metric_cols=['exact_hit1','exact_hit3','directed_edge_recall_top1',
                 'directed_edge_recall_best_top3','length_jaccard_top1','length_jaccard_best_top3']
    for (sc,method),g in result.groupby(['scenario','variant'],sort=True):
        for pop,gg in [('all_requested_OD',g),('strong_conflict_OD',g[g.strong_conflict==1])]:
            validg=gg[gg.truth_valid==1]
            row={'scenario':sc,'variant':method,'population':pop,'n_requested_od':len(gg),
                 'n_truth_valid':len(validg),'n_no_candidate':int((gg.n_recommended==0).sum())}
            for metric in metric_cols:
                row['mean_'+metric+'_over_valid_truth_od']=safe_float(validg[metric].mean())
            summaries.append(row)
    summ=pd.DataFrame(summaries)
    atomic_csv(result,a.out/M6_FILES['truth_perod'])
    atomic_csv(summ,a.out/M6_FILES['truth_summary'])
    audit2={
        'version':'module6_optional_posthoc_real_path_overlap_v1',
        'status':'EXPLORATORY_USED_TEST_OD_NOT_INDEPENDENT_FRESH_TEST',
        'created_at':datetime.now().isoformat(),'input_frozen_sha256':sha256(a.out/M6_FILES['audit']),
        'real_truth_sha256':sha256(f['truth']),
        'graph_sha256':sha256(a.graphml),'candidate_raw_sha256':sha256(f['candidate_raw']),
        'main_top3_sha256':sha256(f['m5main']),'ablations_top3_sha256':sha256(f['m5ablation']),
        'n_all_od':4469,'n_truth_invalid_od':len(invalid),
        'invalid_example':list(invalid.items())[:5],
        'n_evaluation_rows':len(result),'n_unique_road_edges':len(wanted),
        'n_available_road_lengths':len(lengths),
        'output_sha256':{k:sha256(a.out/M6_FILES[k]) for k in ['truth_perod','truth_summary']},
        'no_retraining_after_truth':True,
        'no_simulated_user_demand_from_actual_route':'This phase does not infer passenger demand from realized route.',
        'interpretation':'历史真实路径有向道路重合度仅作行为一致性辅助；严格Exact Hit不是用户需求满意率。',
    }
    atomic_json(audit2,a.out/M6_FILES['truth_audit'])
    log('真实历史路线重合度评价完成。此阶段仅比较冻结推荐与事后真实轨迹，不更新任何模型或权重。')


def self_test():
    rng=np.random.default_rng(42)
    a=np.array([1.,-2.,3.,0.])
    lo,hi=boot_ci(a,100,rng)
    assert lo<=np.mean(a)<=hi
    assert abs(sign_flip_p(np.zeros(10),100,rng)-1)<1e-12
    assert holm([0.01,0.02,0.2])==[.03,.04,.2] or np.allclose(holm([.01,.02,.2]),[.03,.04,.2])
    assert parse_edges('[[1,2,0],[2,3,0]]')==[('1','2','0'),('2','3','0')]
    assert len(PROFILE_COLUMNS)==8
    log('核心统计与真值道路解析自检通过')


def parse_args():
    p=argparse.ArgumentParser(description='第五模块v4连续需求推荐结果的完整第六模块评价程序')
    p.add_argument('--phase',choices=['check','summarize','plot','all','evaluate-truth','self-test'],default='all')
    p.add_argument('--m5',type=Path,default=DEFAULT_M5)
    p.add_argument('--m2',type=Path,default=DEFAULT_M2)
    p.add_argument('--m1',type=Path,default=DEFAULT_M1)
    p.add_argument('--m3',type=Path,default=DEFAULT_M3)
    p.add_argument('--m4',type=Path,default=DEFAULT_M4)
    p.add_argument('--graphml',type=Path,default=DEFAULT_GRAPH)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--permutations',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite-truth',action='store_true')
    p.add_argument('--no-plots',action='store_true')
    return p.parse_args()


def main():
    a=parse_args()
    if a.phase=='self-test':self_test();return
    if a.bootstrap<100 or a.permutations<100:
        raise ValueError('Bootstrap和置换次数至少100')
    f,audit=check_source(a)
    if a.phase=='check':
        perod,change,coverage,profile=read_base(a,f,audit)
        log(f'字段、行数、画像与OD一致性通过：{len(perod):,}逐OD场景方法记录；{len(change):,}配对记录')
        return
    if a.phase in ('all','summarize'):
        tables=summarize(a,f,audit)
        if a.phase=='summarize':return
        if not a.no_plots:
            n=plot_figures(tables,a.out);log(f'已绘制{n}张科研统计图，输出：{a.out}')
    elif a.phase=='plot':
        frozen=loadjson(a.out/M6_FILES['audit'])
        for k,expected in frozen['derived_tables_sha256'].items():
            p=a.out/M6_FILES[k]
            if sha256(required_file(p,k))!=expected:
                raise ValueError('绘图所需第六模块冻结结果已被修改：'+k)
        tables={k:readcsv(a.out/M6_FILES[k]) for k in ('scenarios','conflict','sensitivity','extremes')}
        if not a.no_plots:
            n=plot_figures(tables,a.out);log(f'已绘制{n}张科研统计图')
    elif a.phase=='evaluate-truth':
        evaluate_truth(a,f,audit)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,AssertionError,RuntimeError) as e:
        log(f'第六模块停止：{type(e).__name__}: {e}')
        raise
