# -*- coding: utf-8 -*-
"""
第四章第六模块补充实验｜UDPPM++历史画像对Top-3推荐的真实增量影响
===========================================================================
只读已有的模块五v4正式自然候选Top-3及模块二冻结画像；不训练、不改预测、不读未来真值。
不是“严格仅移除画像”的同模型消融：no_profile 使用不同的冻结PSL效用模型。

Windows运行：
  python.exe .\6step-66.py --phase check
  python.exe .\6step-66.py --phase analyze
  python.exe .\6step-66.py --phase self-test
可指定 --m5 DIR --profile FILE --out DIR，默认路径见下方。
依赖：pip install numpy pandas matplotlib
核心原则：只能依据24/25号逐路线候选ID判断Top-3集合和具体第3名是否变化；
不能从26号的Top-3平均长度推断候选集合一致性。不存在输入文件时明确报错。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r'C:\study\轨迹数据预处理')
ROOT = BASE / '曼谷GPS_正式清洗_202412' / 'FULL_全量' / '第四章_28天训练_3天独立测试'
M5 = ROOT / '模块五_双维偏好_三模式Top3多目标推荐' / 'formal_K10'
PROFILE = ROOT / '模块二_UDPPMpp双维历史画像' / '05_最后3天盲推理8维画像_只按trip_id.csv'
OUT = ROOT / '模块六_历史画像Top3增量实验' / 'formal_K10'
CID = 'candidate_id_TRACE_NOT_FEATURE'
SCENARIOS = ('economic', 'comfortable', 'balanced')
SNAME = {'economic':'经济优先', 'comfortable':'舒适优先', 'balanced':'综合均衡'}
METHODS = ('proposed','no_profile')
INFILES = {
    'main': '24_连续需求_三场景Top3主方法.csv.gz',
    'ablation': '25_连续需求_四类对照Top3.csv.gz',
    'perod': '26_连续需求_逐OD逐场景逐方法指标.csv',
    'switch': '27_经济舒适场景切换_逐OD五方法详细变化.csv',
    'audit': '31_连续需求_来源防泄漏与统计审计.json',
}
# 注意，26号中的Top3属性是三条路线的简单算术平均；不是观测的舒适度/真实经济成本。
METRICS = {
    'top1_own_goal_regret': ('目标后悔值Top1（越低越好）', -1),
    'top3_best_goal_regret': ('目标后悔值Top3最佳（越低越好）', -1),
    'top1_total_len_km': ('Top1路程km（越低越好，仅经济性）', -1),
    'top1_turns_per_km': ('Top1转弯密度（越低越好，仅代理指标）', -1),
    'top1_economic_proxy_score': ('Top1经济代理得分（越高越好）', +1),
    'top1_comfort_proxy_score': ('Top1舒适代理得分（越高越好）', +1),
    'top3_mean_length_km': ('Top3平均里程km（越低越好，仅经济性）', -1),
    'top3_mean_economic_proxy': ('Top3平均经济代理（越高越好）', +1),
    'top3_mean_comfort_proxy': ('Top3平均舒适代理（越高越好）', +1),
    'top3_mean_pairwise_overlap': ('Top3道路重叠（越低越好）', -1),
}
# 同一场景的关键目标质量：仅作代理评价，直接道路规则可能天然优于复杂模型。
MAIN_METRICS = ('top1_own_goal_regret','top3_best_goal_regret',
                'top3_mean_economic_proxy','top3_mean_comfort_proxy',
                'top3_mean_pairwise_overlap')
PROFILE_COLS = [f'history_econ_{j}' for j in range(4)]+[f'history_comf_{j}' for j in range(4)]


def log(s: str):
    print(f'[{datetime.now():%H:%M:%S}] {s}', flush=True)


def must_exist(p:Path, label:str):
    if not p.is_file() or not p.stat().st_size:
        raise FileNotFoundError(f'缺少{label}：{p}\n请先确认5step-55.py的--phase demand-soft正式运行成功。')


def sha256(p:Path) -> str:
    hh=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):hh.update(b)
    return hh.hexdigest()


def read_json(p:Path):
    must_exist(p,'来源审计JSON')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def read_csv(p:Path):
    must_exist(p,'真实实验CSV')
    return pd.read_csv(p,encoding='utf-8-sig',low_memory=False,
                       dtype={'trip_id':'string',CID:'string','top1_candidate_id':'string'})


def require(df:pd.DataFrame, cols, label):
    absent=sorted(set(cols)-set(df.columns))
    if absent:raise ValueError(f'{label}缺少真实字段：{absent}')


def safe_write_csv(df, p:Path):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.part')
    df.to_csv(tmp,index=False,float_format='%.10g',encoding='utf-8-sig')
    tmp.replace(p)


def safe_write_json(obj,p:Path):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.part')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf-8-sig')
    tmp.replace(p)


def history_bin(n):
    n=int(n)
    if n==0:return '0_无历史'
    if n<=5:return '1_1到5条'
    if n<=20:return '2_6到20条'
    return '3_超过20条'


def rank_compare(a:list[str], b:list[str]) -> dict:
    """a和b均为本OD同场景的rank1...rankK，必须由真实24/25号候选ID生成。"""
    if not a or not b or len(a)!=len(b):
        raise ValueError('两方法的TopK有效列表不能为空，且必须拥有相同的推荐数量')
    if len(a)!=len(set(a)) or len(b)!=len(set(b)):
        raise ValueError('同一OD/场景/方法中出现候选ID重复')
    sa,sb=set(a),set(b)
    common=sa & sb
    rank_a={x:i for i,x in enumerate(a)}
    rank_b={x:i for i,x in enumerate(b)}
    shared_sorted=sorted(common,key=lambda x:rank_a[x])
    n_pairs=len(shared_sorted)*(len(shared_sorted)-1)//2
    reversals=sum(rank_b[shared_sorted[i]]>rank_b[shared_sorted[j]]
                  for i in range(len(shared_sorted))
                  for j in range(i+1,len(shared_sorted)))
    return {
        'top1_same':int(a[0]==b[0]),
        'top2_same':int(a[1]==b[1]) if len(a)>=2 else np.nan,
        'top3_same':int(a[2]==b[2]) if len(a)>=3 else np.nan,
        'topk_same_set':int(sa==sb),
        'topk_same_exact_order':int(a==b),
        'topk_same_set_but_reordered':int(sa==sb and a!=b),
        'topk_changed_set':int(sa!=sb),
        'topk_shared_count':len(common),
        'topk_replaced_count':len(a)-len(common),
        'topk_set_jaccard':len(common)/len(sa|sb),
        'topk_rank_pair_inversions_shared':reversals,
        'topk_rank_pair_total_shared':n_pairs,
        'topk_rank_inversion_ratio_shared':reversals/n_pairs if n_pairs else np.nan,
    }


def bootstrap_mean(values, repetitions, seed):
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    if len(v)<2:return (np.nan,np.nan)
    rng=np.random.default_rng(int(seed))
    samples=np.empty(repetitions,dtype=float)
    for off in range(0,repetitions,100):
        m=min(100,repetitions-off)
        ix=rng.integers(0,len(v),size=(m,len(v)))
        samples[off:off+m]=v[ix].mean(axis=1)
    return tuple(np.quantile(samples,[.025,.975]).tolist())


def paired_signflip(values, repetitions, seed):
    """探索性的双侧配对符号翻转检验；有大量完全相同的OD时返回p=1。"""
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    if not len(v):return np.nan
    if np.all(v==0):return 1.0
    obs=abs(v.mean())
    rng=np.random.default_rng(int(seed))
    exceed=0
    for off in range(0,repetitions,100):
        n=min(100,repetitions-off)
        sg=rng.choice(np.array([-1.0,1.0]),size=(n,len(v)))
        means=abs((sg*v).mean(axis=1))
        exceed+=int((means>=obs-1e-14).sum())
    return (exceed+1)/(repetitions+1)


def holm(ps:list[float]):
    """每个场景+组别的一组预声明指标进行Holm-Bonferroni校正。"""
    out=[np.nan]*len(ps)
    ids=[i for i,p in enumerate(ps) if np.isfinite(p)]
    order=sorted(ids,key=lambda i:ps[i])
    prev=0.
    m=len(order)
    for j,i in enumerate(order):
        val=min(1.,max(prev,float(ps[i])*(m-j)))
        out[i]=val;prev=val
    return out


def load_sources(args):
    paths={k:args.m5/v for k,v in INFILES.items()}
    paths['profile']=args.profile
    for label,p in paths.items():must_exist(p,label)
    audit=read_json(paths['audit'])
    if audit.get('version')!='chapter4_module5_v4_soft_demand_conflict_analysis':
        raise ValueError('31号来源审计不是模块五v4连续需求版本，禁止混用旧15—22号结果')
    if audit.get('future_true_HMM_trajectory_read')!='NEVER':
        raise ValueError('第五模块来源审计没有明确声明最后3天真值未读取')
    expected=audit.get('new_output_sha256',{})
    expected_map={'main':'main','ablation':'ablations','perod':'perod','switch':'switch'}
    hashes={}
    for k,p in paths.items():
        hashes[k]=sha256(p)
        ref=expected.get(expected_map.get(k,''))
        if ref and hashes[k]!=ref:
            if not args.allow_hash_mismatch:
                raise ValueError(f'{k}输入文件与31号审计SHA256不一致，拒绝混合版本：\n{p}\n'
                                 '如确实是更新版本，可用 --allow-hash-mismatch 继续，但会记录警告')
            log(f'警告：{k}的哈希与31号冻结记录不一致，将在本次审计中注明')
    expected_profile=4469
    profile=read_csv(paths['profile'])
    require(profile,['trip_id','history_n_prior','is_cold_start']+PROFILE_COLS,'模块二盲画像')
    if profile.trip_id.isna().any() or profile.trip_id.duplicated().any():
        raise ValueError('模块二trip_id缺失或重复')
    if (pd.to_numeric(profile.history_n_prior,errors='coerce').isna()).any():
        raise ValueError('history_n_prior不是合法的数字')
    if (pd.to_numeric(profile.history_n_prior)<0).any():
        raise ValueError('history_n_prior不能为负')
    if not profile.is_cold_start.isin([0,1]).all():
        raise ValueError('is_cold_start必须是0或1')
    if len(profile)!=expected_profile:
        raise ValueError(f'模块二盲画像期望{expected_profile}条，实际{len(profile)}条')
    for col in PROFILE_COLS:
        if not np.isfinite(pd.to_numeric(profile[col],errors='coerce').to_numpy(float)).all():
            raise ValueError(f'模块二画像字段{col}出现缺失、非数值或无穷大')
    profile['history_n_prior']=pd.to_numeric(profile.history_n_prior).astype(int)
    if ((profile.history_n_prior==0).astype(int)!=profile.is_cold_start.astype(int)).any():
        raise ValueError('is_cold_start与history_n_prior==0不一致')
    profile['history_group']=profile.history_n_prior.map(history_bin)
    a=read_csv(paths['main'])
    b=read_csv(paths['ablation'])
    perod=read_csv(paths['perod'])
    switch=read_csv(paths['switch'])
    req_rank=['trip_id','scenario','variant','rank_in_scenario',CID,'strong_conflict']
    for df,name in ((a,'24主方案'),(b,'25消融方案')):
        require(df,req_rank,name)
        if df[['trip_id','scenario',CID]].isna().any().any():
            raise ValueError(f'{name}有缺失的trip_id/scenario/candidate_id')
    require(perod,['trip_id','scenario','variant','candidate_count','n_recommended',
                   'strong_conflict','top1_candidate_id']+list(METRICS),'26逐OD指标')
    require(switch,['trip_id','variant','strong_conflict','top1_switched'],'27切换审计')
    if set(a.variant.unique())!={'proposed'}:
        raise ValueError('24号应当只包含proposed主方案')
    if 'no_profile' not in set(b.variant.unique()):
        raise ValueError('25号缺少no_profile对照，不能开展画像影响分析')
    if set(perod.scenario.dropna().unique())!=set(SCENARIOS):
        raise ValueError('26号三场景不完整')
    if len(perod)!=4466*3*5 or len(a)!=38931:
        raise ValueError(f'正式数据规模与31号审计不符：24号{len(a)},26号{len(perod)}')
    if not set(profile.trip_id.astype(str)).issuperset(set(perod.trip_id.astype(str))):
        raise ValueError('模块五存在无法关联模块二盲画像的trip_id')
    if set(profile.trip_id.astype(str))-set(perod.trip_id.astype(str)) and \
       len(set(profile.trip_id.astype(str))-set(perod.trip_id.astype(str)))!=3:
        raise ValueError('有候选OD之外的画像数量不等于3条无自然候选OD')
    selected_b=b[b.variant.eq('no_profile')].copy()
    rank=pd.concat([a,selected_b],ignore_index=True)
    rank['rank_in_scenario']=pd.to_numeric(rank.rank_in_scenario,errors='coerce')
    if rank.rank_in_scenario.isna().any():
        raise ValueError('Top3名次存在非数值')
    rank['rank_in_scenario']=rank.rank_in_scenario.astype(int)
    if not rank.rank_in_scenario.between(1,3).all():
        raise ValueError('Top3排名超出1~3')
    if rank[['trip_id','scenario','variant','rank_in_scenario']].duplicated().any():
        raise ValueError('相同OD/场景/方法出现重复rank')
    if rank[['trip_id','scenario','variant',CID]].duplicated().any():
        raise ValueError('相同OD/场景/方法出现重复候选ID')
    count_group=rank.groupby(['trip_id','scenario','variant'],sort=False).size().reset_index(name='real_rank_count')
    if len(count_group)!=4466*3*2 or count_group.real_rank_count.sum()!=2*38931:
        raise ValueError('24/25号Top3两方法覆盖与正式审计不一致')
    if perod[['trip_id','scenario','variant']].duplicated().any():
        raise ValueError('26号OD/场景/方法出现重复')
    perod2=perod[perod.variant.isin(METHODS)].copy()
    merged=count_group.merge(perod2,on=['trip_id','scenario','variant'],how='outer',validate='one_to_one',indicator=True)
    if not merged._merge.eq('both').all():
        raise ValueError('24/25号推荐与26号逐OD记录不能一对一完整匹配')
    if not (merged.real_rank_count.astype(int)==merged.n_recommended.astype(int)).all():
        raise ValueError('Top3具体名次记录数量与26号n_recommended不同')
    if not (merged.real_rank_count.astype(int)==np.minimum(3,merged.candidate_count.astype(int))).all():
        raise ValueError('Top3推荐数量与自然候选数量不一致')
    # 核验Top1候选ID，防止误读文件名、旧版本或主键关联错误
    first=rank[rank.rank_in_scenario.eq(1)][['trip_id','scenario','variant',CID]]
    heads=first.merge(perod2[['trip_id','scenario','variant','top1_candidate_id']],
                      on=['trip_id','scenario','variant'],validate='one_to_one')
    if len(heads)!=4466*3*2 or not heads[CID].astype(str).eq(heads.top1_candidate_id.astype(str)).all():
        raise ValueError('24/25号真实Top1候选ID与26号不一致，禁止继续')
    # 26号平均道路属性必须由24/25相同候选的真实属性重算出来。
    mean_cols={'total_len_km':'top3_mean_length_km',
               'economic_proxy_score':'top3_mean_economic_proxy',
               'comfort_proxy_score':'top3_mean_comfort_proxy'}
    if all(k in rank.columns for k in mean_cols):
        av=rank.groupby(['trip_id','scenario','variant'],sort=False)[list(mean_cols)].mean().reset_index()
        check=av.merge(perod2,on=['trip_id','scenario','variant'],validate='one_to_one')
        for orig,dst in mean_cols.items():
            x=pd.to_numeric(check[orig],errors='coerce').to_numpy(float)
            y=pd.to_numeric(check[dst],errors='coerce').to_numpy(float)
            if not np.isfinite(x).all() or not np.allclose(x,y,atol=1e-6,rtol=1e-6):
                raise ValueError(f'逐路线{orig}与26号{dst}均值不符：不能混用实验版本')
    # 用27号再次校验强冲突标签（只有proposed的一行）；不借此臆造其他标签。
    prop27=switch[switch.variant.eq('proposed')]
    if len(prop27)!=4466 or prop27.trip_id.duplicated().any():
        raise ValueError('27号proposed逐OD切换审计数量不符')
    prop26=perod2[(perod2.variant=='proposed')&(perod2.scenario=='economic')]
    chk=prop26[['trip_id','strong_conflict']].merge(
        prop27[['trip_id','strong_conflict']],on='trip_id',suffixes=('_26','_27'),validate='one_to_one')
    if len(chk)!=4466 or not chk.strong_conflict_26.eq(chk.strong_conflict_27).all():
        raise ValueError('26与27号强冲突标签不同')
    if int(chk.strong_conflict_26.sum())!=1224:
        raise ValueError('强冲突OD数量与上游正式审计1224不一致')
    return paths,hashes,audit,profile,rank,perod2


def build_details(profile,rank,perod):
    prof=profile.set_index('trip_id')
    keyed=perod.set_index(['trip_id','scenario','variant'])
    rows=[]
    for (tid,scene),group in rank.groupby(['trip_id','scenario'],sort=True):
        top={}
        for method in METHODS:
            d=group[group.variant.eq(method)].sort_values('rank_in_scenario',kind='mergesort')
            vals=d[CID].astype(str).tolist()
            if not vals:raise ValueError(f'缺少Top3候选：{tid}/{scene}/{method}')
            if d.rank_in_scenario.tolist()!=list(range(1,len(vals)+1)):
                raise ValueError(f'Top3名次不连续：{tid}/{scene}/{method}')
            top[method]=vals
        aa,bb=top['proposed'],top['no_profile']
        p=keyed.loc[(tid,scene,'proposed')]
        n=keyed.loc[(tid,scene,'no_profile')]
        if int(p.strong_conflict)!=int(n.strong_conflict):
            raise ValueError(f'画像有无不应改变强冲突标签：{tid}/{scene}')
        pr=prof.loc[tid]
        row={'trip_id':str(tid),'scenario':str(scene),
             'history_n_prior':int(pr.history_n_prior),
             'history_group':pr.history_group,
             'is_cold_start':int(pr.is_cold_start),
             'strong_conflict':int(p.strong_conflict),
             'candidate_count':int(p.candidate_count),
             'topk_count':len(aa),
             'profile_rank1_id':aa[0],'no_profile_rank1_id':bb[0],
             'profile_rank2_id':aa[1] if len(aa)>=2 else '',
             'no_profile_rank2_id':bb[1] if len(bb)>=2 else '',
             'profile_rank3_id':aa[2] if len(aa)>=3 else '',
             'no_profile_rank3_id':bb[2] if len(bb)>=3 else '',
             'profile_topk_candidate_ids_json':json.dumps(aa,ensure_ascii=False),
             'no_profile_topk_candidate_ids_json':json.dumps(bb,ensure_ascii=False),
             **rank_compare(aa,bb)}
        for metric in METRICS:
            v1=pd.to_numeric(pd.Series([p[metric]]),errors='coerce').iloc[0]
            v0=pd.to_numeric(pd.Series([n[metric]]),errors='coerce').iloc[0]
            row['profile_'+metric]=float(v1) if pd.notna(v1) else np.nan
            row['no_profile_'+metric]=float(v0) if pd.notna(v0) else np.nan
            row['delta_'+metric]=float(v1-v0) if pd.notna(v1) and pd.notna(v0) else np.nan
        rows.append(row)
    result=pd.DataFrame(rows)
    if len(result)!=4466*3 or result[['trip_id','scenario']].duplicated().any():
        raise ValueError('逐OD对比构建的行数或主键不正确')
    for scene in SCENARIOS:
        sub=result[result.scenario==scene]
        if int(sub.strong_conflict.sum())!=1224 or sub.trip_id.nunique()!=4466:
            raise ValueError(f'{scene}场景覆盖或强冲突OD数量异常')
        if int(sub.topk_count.eq(3).sum())!=4192:
            raise ValueError(f'{scene}场景Top3真正有第三名的OD数量不等于4192')
    return result


def group_stats(df, scene, group_name):
    n=len(df)
    eligible3=df[df.topk_count.ge(3)]
    n3=len(eligible3)
    get=lambda c:float(df[c].mean()) if n else np.nan
    get3=lambda c:float(eligible3[c].mean()) if n3 else np.nan
    return {'scenario':scene,'population':group_name,'n_od':n,
            'n_with_rank3':n3,
            'top1_same_fraction':get('top1_same'),
            'top2_same_fraction':get('top2_same'),
            'rank3_same_fraction':get3('top3_same'),
            # “真正的Top3”仅在存在第三条候选的OD上计算，禁止用1/2条冒充。
            'top3_same_set_fraction':get3('topk_same_set'),
            'top3_exact_order_same_fraction':get3('topk_same_exact_order'),
            'top3_same_set_reordered_fraction':get3('topk_same_set_but_reordered'),
            'top3_changed_set_fraction':get3('topk_changed_set'),
            'mean_top3_replaced_candidates':get3('topk_replaced_count'),
            'mean_top3_set_jaccard':get3('topk_set_jaccard'),
            'mean_top3_shared_rank_inversion_ratio':get3('topk_rank_inversion_ratio_shared'),
            # 可用TopK含仅1/2条候选的OD，单独作为覆盖统计，不能标为Top3。
            'available_topk_same_set_fraction':get('topk_same_set'),
            'available_topk_exact_order_same_fraction':get('topk_same_exact_order'),
            'share_history0':float((df.history_n_prior==0).mean()) if n else np.nan}


def summaries(detail):
    out=[];hist=[];cross=[]
    for scene in SCENARIOS:
        part=detail[detail.scenario==scene]
        populations={
            'all_scored_OD':part,
            'strong_conflict_OD':part[part.strong_conflict.eq(1)],
            'non_strong_conflict_OD':part[part.strong_conflict.eq(0)],
            'top3_changed_set_OD':part[part.topk_count.ge(3)&part.topk_changed_set.eq(1)],
            'top3_same_set_reordered_OD':part[part.topk_count.ge(3)&part.topk_same_set_but_reordered.eq(1)],
            'top3_exact_order_unchanged_OD':part[part.topk_count.ge(3)&part.topk_same_exact_order.eq(1)],
        }
        for name,dd in populations.items():out.append(group_stats(dd,scene,name))
        for k,dd in part.groupby('history_group',sort=True):
            hist.append({'history_group':k,**group_stats(dd,scene,'history_group')})
            for c in (0,1):
                cc=dd[dd.strong_conflict.eq(c)]
                cross.append({'history_group':k,'strong_conflict':c,
                              **group_stats(cc,scene,'history_x_conflict')})
    return pd.DataFrame(out),pd.DataFrame(hist),pd.DataFrame(cross)


def quality_stats(detail,args):
    """全部OD是主要结论；变化子集仅作机制诊断，不能只挑改善样本报告。"""
    rows=[]
    for si,scene in enumerate(SCENARIOS):
        all_scene=detail[detail.scenario.eq(scene)]
        populations=[('all_scored_OD',all_scene),
                     ('strong_conflict_OD',all_scene[all_scene.strong_conflict.eq(1)]),
                     ('top3_changed_set_OD',all_scene[all_scene.topk_count.ge(3)&all_scene.topk_changed_set.eq(1)]),
                     ('top3_same_set_reordered_OD',all_scene[all_scene.topk_count.ge(3)&all_scene.topk_same_set_but_reordered.eq(1)]),
                     ('top3_unchanged_exact_OD',all_scene[all_scene.topk_count.ge(3)&all_scene.topk_same_exact_order.eq(1)]),
                     ('strong_conflict_and_top3_changed_OD',
                      all_scene[all_scene.strong_conflict.eq(1)&all_scene.topk_count.ge(3)&all_scene.topk_changed_set.eq(1)])]
        for pop_i,(pop,g) in enumerate(populations):
            for met_i,(name,(label,direction)) in enumerate(METRICS.items()):
                vals=pd.to_numeric(g['delta_'+name],errors='coerce').to_numpy(float)
                valid=vals[np.isfinite(vals)]
                n=len(valid)
                boot=bootstrap_mean(valid,args.bootstrap,args.seed+si*1000+pop_i*100+met_i)
                p=paired_signflip(valid,args.permutation,args.seed+101+si*1000+pop_i*100+met_i) \
                  if pop in ('all_scored_OD','strong_conflict_OD') and name in MAIN_METRICS else np.nan
                rows.append({'scenario':scene,'population':pop,'metric':name,
                             'label':label,'higher_is_better':int(direction>0),
                             'n_od_population':len(g),'n_valid_paired_od':n,
                             'mean_delta_profile_minus_no_profile':float(valid.mean()) if n else np.nan,
                             'median_delta':float(np.median(valid)) if n else np.nan,
                             'ci95_low':boot[0],'ci95_high':boot[1],
                             'share_profile_better':float((direction*valid>1e-12).mean()) if n else np.nan,
                             'share_no_profile_better':float((direction*valid<-1e-12).mean()) if n else np.nan,
                             'share_equal':float((abs(valid)<=1e-12).mean()) if n else np.nan,
                             'p_permutation_two_sided_exploratory':p,
                             'p_holm_within_scene_population':np.nan})
    out=pd.DataFrame(rows)
    for (scene,pop),part in out.groupby(['scenario','population'],sort=False):
        indices=[idx for idx in part.index if out.loc[idx,'metric'] in MAIN_METRICS]
        adjusted=holm(out.loc[indices,'p_permutation_two_sided_exploratory'].tolist())
        for ii,ap in zip(indices,adjusted):out.at[ii,'p_holm_within_scene_population']=ap
    return out


def make_figures(summary,hist,quality,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log('未安装matplotlib，只生成CSV和JSON；可执行pip install matplotlib后重新绘图')
        return []
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'):
        if f in names:
            plt.rcParams['font.sans-serif']=[f,'DejaVu Sans'];break
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['figure.dpi']=155
    files=[]
    d=summary[summary.population.eq('all_scored_OD')].set_index('scenario').loc[list(SCENARIOS)]
    labels=['Economic','Comfort','Balanced']
    x=np.arange(3);width=.24
    fig,ax=plt.subplots(figsize=(9,5))
    keys=[('top1_same_fraction','Same Top-1'),('rank3_same_fraction','Same rank-3'),
          ('top3_same_set_fraction','Same Top-3 set')]
    for j,(key,label) in enumerate(keys):
        vals=100*d[key].to_numpy(float)
        ax.bar(x+(j-1)*width,vals,width=width,label=label)
    ax.set_xticks(x,labels);ax.set_ylim(0,105)
    ax.set_ylabel('Identical recommendations (%)')
    ax.set_title('Historical profile ablation: exact candidate-ID agreement')
    ax.legend(loc='lower left');ax.grid(axis='y',alpha=.18)
    fig.tight_layout()
    p=out/'07_画像消融Top1_第三名_Top3集合一致率.png';fig.savefig(p,dpi=210);plt.close(fig);files.append(p)
    order=['0_无历史','1_1到5条','2_6到20条','3_超过20条']
    fig,ax=plt.subplots(figsize=(9,5))
    for scene in SCENARIOS:
        dd=hist[hist.scenario.eq(scene)].set_index('history_group').reindex(order)
        ax.plot(range(4),100*dd.top3_same_set_fraction.to_numpy(float),marker='o',label=SNAME[scene])
    ax.set_xticks(range(4),['0','1–5','6–20','>20'])
    ax.set_xlabel('Completed historical trips available before OD')
    ax.set_ylabel('Identical Top-3 candidate sets (%)')
    ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend()
    ax.set_title('Top-3 set agreement by history availability')
    fig.tight_layout()
    p=out/'08_不同历史记录数量下Top3集合一致率.png';fig.savefig(p,dpi=210);plt.close(fig);files.append(p)
    fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    metrics=['top1_own_goal_regret','top3_mean_pairwise_overlap','top3_mean_comfort_proxy']
    titles=['Top-1 goal regret','Top-3 route overlap','Top-3 comfort proxy']
    for ax,met,title in zip(axs,metrics,titles):
        qq=quality[(quality.population=='all_scored_OD')&(quality.metric==met)].set_index('scenario').loc[list(SCENARIOS)]
        mean=qq.mean_delta_profile_minus_no_profile.to_numpy(float)
        low=qq.ci95_low.to_numpy(float);high=qq.ci95_high.to_numpy(float)
        err=np.vstack([np.maximum(0,mean-low),np.maximum(0,high-mean)])
        ax.errorbar(range(3),mean,yerr=err,fmt='o',capsize=4)
        ax.axhline(0,ls='--',lw=1)
        ax.set_xticks(range(3),labels,rotation=20)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Profile minus no-profile (OD-paired bootstrap 95% CI)',fontsize=12)
    p=out/'09_历史画像质量增量_配对差与95CI.png';fig.savefig(p,dpi=210);plt.close(fig);files.append(p)
    return files


def self_test():
    assert rank_compare(['A','B','C'],['A','B','C'])['topk_same_exact_order']==1
    z=rank_compare(['A','B','C'],['C','B','A'])
    assert z['topk_same_set']==1 and z['top3_same']==0 and z['topk_rank_pair_inversions_shared']==3
    z=rank_compare(['A','B','C'],['A','D','C'])
    assert z['topk_changed_set']==1 and z['top3_same']==1 and z['topk_replaced_count']==1
    assert np.isnan(rank_compare(['A'],['A'])['top3_same'])
    assert holm([.01,.03,.02])==[.03,.04,.04]
    assert all(history_bin(k)==v for k,v in [(0,'0_无历史'),(3,'1_1到5条'),
                              (6,'2_6到20条'),(21,'3_超过20条')])
    assert paired_signflip([0,0,0],100,42)==1.0
    log('内部Top3候选ID、名次翻转、缺少第三名、Holm校正等单元测试通过。')


def main():
    parser=argparse.ArgumentParser(description='真实数据UDPPM++历史画像Top3候选ID与排序增量分析')
    parser.add_argument('--phase',choices=('check','analyze','self-test'),default='analyze')
    parser.add_argument('--m5',type=Path,default=M5,help='第五模块formal_K10含24—27、31号文件目录')
    parser.add_argument('--profile',type=Path,default=PROFILE,help='模块二05号盲画像CSV')
    parser.add_argument('--out',type=Path,default=OUT,help='新增结果独立目录，不覆盖原模块')
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--permutation',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--allow-hash-mismatch',action='store_true',help='显式放宽上游文件SHA核验并审计，不推荐')
    parser.add_argument('--overwrite',action='store_true',help='明确覆盖本补充程序自己的输出（不涉及上游）')
    args=parser.parse_args()
    if args.phase=='self-test':self_test();return
    if args.bootstrap<100 or args.permutation<100:
        raise ValueError('Bootstrap和符号翻转至少设置100次')
    log('读取真实模块五24/25号候选名次并核验31号来源审计（不读真值）')
    paths,hashes,audit,profile,rank,perod=load_sources(args)
    log(f'来源核验通过：候选推荐行={len(rank):,}；有候选OD={perod.trip_id.nunique():,}；历史画像OD={len(profile):,}')
    if args.phase=='check':
        log('check成功。现在运行：python.exe .\\6step-66.py --phase analyze')
        return
    out=args.out
    files={
        'detail':out/'01_逐OD三场景_有无画像Top3候选ID及排序精确对照.csv',
        'summary':out/'02_全部及强冲突OD_Top3集合与第三名一致率汇总.csv',
        'history':out/'03_历史记录数量分层_Top3候选变化汇总.csv',
        'cross':out/'04_历史记录与强冲突交叉分层_Top3变化.csv',
        'quality':out/'05_有无画像_全部与变化OD道路质量配对检验.csv',
        'changed':out/'06_真实发生Top3变化OD_全部案例及目标权衡.csv',
        'audit':out/'00_来源冻结与画像增量分析审计.json',
    }
    existent=[p for p in files.values() if p.exists()]
    if existent and not args.overwrite:
        raise FileExistsError('本程序已有输出，为防覆盖请指定新--out目录或显式--overwrite：'+str(existent[0]))
    started=time.perf_counter()
    detail=build_details(profile,rank,perod)
    summary,hist,cross=summaries(detail)
    log('精确候选ID分析完成；开始计算质量变化与OD级置信区间')
    quality=quality_stats(detail,args)
    changed=detail[detail.topk_count.ge(3)&(detail.topk_changed_set.eq(1)|detail.topk_same_set_but_reordered.eq(1))].copy()
    for sc in SCENARIOS:
        sub=summary[(summary.scenario==sc)&(summary.population=='all_scored_OD')].iloc[0]
        log(f'{SNAME[sc]}：Top1相同{sub.top1_same_fraction:.2%}，第3名相同{sub.rank3_same_fraction:.2%}，'
            f'Top3集合相同{sub.top3_same_set_fraction:.2%}，顺序完全相同{sub.top3_exact_order_same_fraction:.2%}')
    for k,df in [('detail',detail),('summary',summary),('history',hist),('cross',cross),
                 ('quality',quality),('changed',changed)]:
        safe_write_csv(df,files[k])
    figures=make_figures(summary,hist,quality,out) if not args.no_plots else []
    report={
        'version':'module6_supplement_UDPPM_profile_exact_Top3_ablation_v1.0',
        'created_at':datetime.now().isoformat(),
        'scope':'posthoc exploratory profile vs frozen no_profile PSL comparator',
        'upstream_version':audit['version'],
        'n_all_od':len(profile),'n_scored_od':perod.trip_id.nunique(),
        'n_no_candidate_od':len(profile)-perod.trip_id.nunique(),
        'n_od_scenario_rows':len(detail),
        'n_strong_conflict_od':int(detail[detail.scenario.eq('economic')].strong_conflict.sum()),
        'rank3_denominator_per_scenario':4192,
        'upstream_source_hashes':hashes,
        'hash_validation_overridden':bool(args.allow_hash_mismatch),
        'upstream_frozen_model_sha256':audit.get('frozen_module4_model_sha256'),
        'upstream_frozen_prediction_sha256':audit.get('frozen_module4_predictions_sha256'),
        'output_hashes':{p.name:sha256(p) for p in files.values() if p.is_file() and p!=files['audit']},
        'figures':[p.name for p in figures],
        'elapsed_seconds':round(time.perf_counter()-started,3),
        'future_real_HMM_route_read':'NEVER',
        'new_candidates_generated':'NEVER',
        'historical_or_frozen_models_retrained':'NEVER',
        'limitation':[
            'no_profile以PSL_LengthWeighted冻结模型替换UDPPM模型，不能把变化纯粹归因于8维画像',
            '本补充仅比较现有两个已训练模型输出的真实候选ID与质量，不是重新训练同模型结构消融',
            'Top3集合相同不代表候选顺序相同；第三名相同只统计实际有第三名的OD',
            '所用经济/舒适质量为同OD内静态道路结构代理，非实际乘客偏好、车内实测舒适度或油耗',
            '画像来自车辆历史，不能等同稳定的司机或乘客身份与需求',
            '反复查看测试期后调整第五模块方案，属于探索性结果，不能称未参与开发的独立盲测',
            '变化OD子集的改善与显著性只能作机制描述，不能只选改善案例替代全部OD结果',
            'Bootstrap在OD层面采样，未对同一车辆多个OD的相关性做簇级修正',
        ],
    }
    safe_write_json(report,files['audit'])
    log(f'完成。独立输出：{out}')
    log('优先查看01号逐OD真实候选ID、02号总体一致率、05号质量检验、06号全部变化OD明细。')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        log(f'画像Top3增量分析停止：{type(exc).__name__}: {exc}')
        raise
