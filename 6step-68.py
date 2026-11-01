# -*- coding: utf-8 -*-
"""
第四章补充实验：6step-68.py
历史验证期 UDPPM++ 8维画像的条件路径选择解释能力

只复用模块四已有02号历史条件候选、模块二03号严格历史画像、
模块一01号OD时间索引及模块四的准备审计；不重新地图匹配、不搜索路网、
不读取12月29—31日真实行驶路径，不覆盖原模型。

严格时间方案：
  2024-12-15~23 训练：拟合条件Logit、训练期标准化及训练期群体参考画像。
  2024-12-24~28 验证：同一冻结UDPPM模型，对每个OD分别输入
     A: 该OD出发前真实历史8维画像
     B: 从训练期OD确定的统一8维群体参考画像。
  相同训练/验证条件候选再训练一套PSL_LengthWeighted基线作参考。

注意：历史监督条件候选可能含真实路径补入。NLL/Hit1/MRR均是【已含
真实选择项的条件选择集合】上的验证，不是未来未知OD导航精度。
过去使用过这段验证期选择L2；本分析为探索性复核，不是全新独立验证。

安装：pip install numpy pandas scipy matplotlib
运行：
  python.exe .\\6step-68.py --phase check
  python.exe .\\6step-68.py --phase analyze
可选：--ridge 0.003 --bootstrap 1000 --permutations 2000
不要依据验证结果反复挑选ridge以制造更好的结论。
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
from scipy.optimize import minimize

HOME = Path(r'C:\study\轨迹数据预处理')
ROOT = HOME / '曼谷GPS_正式清洗_202412' / 'FULL_全量' / '第四章_28天训练_3天独立测试'
M1 = ROOT / '模块一_时间划分与隔离'
M2 = ROOT / '模块二_UDPPMpp双维历史画像'
M4 = ROOT / '模块四_条件选择六模型与未知OD盲排序' / 'formal_K10'
DEFAULT_OUT = ROOT / '模块六_严格历史画像条件选择验证' / 'formal_K10'

BASE10 = ['total_len_km','ratio_to_shortest','ratio_motorway','ratio_major',
          'ratio_local','turns_per_km','segment_density_seg_per_km','oneway_ratio',
          'lanes_mean_w','maxspeed_mean_known_kmh']
SCALE_COLS = BASE10 + ['log_ps_count','log_ps_length']
ECON_ROUTE=['total_len_km','ratio_to_shortest','ratio_motorway','ratio_major','ratio_local']
COMF_ROUTE=['turns_per_km','segment_density_seg_per_km','ratio_local','ratio_motorway','oneway_ratio']
PROFILE = [f'history_{dim}_{k}' for dim in ('econ','comf') for k in range(4)]
MODELS = {
    'PSL_LengthWeighted_L2': ['base','ps_length'],
    'UDPPM_Econ_PSL_L2': ['base','ps_length','econ'],
    'UDPPM_Comf_PSL_L2': ['base','ps_length','comf'],
    'UDPPM_Dual_PSL_L2': ['base','ps_length','econ','comf'],
}


def log(s):
    print(f'[{datetime.now():%H:%M:%S}] {s}',flush=True)


def jwrite(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8-sig')
    tmp.replace(path)


def savecsv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    df.to_csv(tmp,index=False,encoding='utf-8-sig',float_format='%.12g',
              compression='gzip' if path.suffix=='.gz' else None)
    tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def need(path,label):
    if not path.is_file() or path.stat().st_size==0:
        raise FileNotFoundError(f'{label}不存在：{path}\n请检查--m1/--m2/--m4，禁止用假数据代替。')


def require_cols(df,names,label):
    missing=sorted(set(names)-set(df.columns))
    if missing:
        raise ValueError(f'{label}缺失真实字段: {missing}；实际字段前40个：{list(df.columns)[:40]}')


def paths(a):
    return {
        'candidates':a.m4/'02_训练验证条件候选与静态属性.csv.gz',
        'profile':a.m2/'03_训练验证逐OD严格历史画像.csv.gz',
        'index':a.m1/'01_全量OD时间分组索引.csv',
        'prepare_audit':a.m4/'01_开发OD候选生成与真实补入审计.csv',
        'prepare_manifest':a.m4/'03_历史候选准备审计.json',
        'm4_final_provenance':a.m4/'06_正式模型参数与训练来源.json',
    }


def load_source(a,read_full=True):
    p=paths(a)
    for k,x in p.items(): need(x,k)
    manifest=read_json(p['prepare_manifest'])
    frozen=read_json(p['m4_final_provenance'])
    if frozen.get('pilot') is not False:
        raise ValueError('模块四来源模型是pilot版本，停止')
    for source_key, expected_key in [('candidates','history_candidate_sha256'),('profile','profile_source_sha256')]:
        expected=frozen.get(expected_key)
        if not expected:
            raise ValueError('模块四正式模型来源缺少必要哈希字段：'+expected_key)
        actual=sha256(p[source_key])
        if actual!=expected:
            raise ValueError(f'{source_key}实际SHA256与模块四正式模型来源不符，可能误读其他版本')
    if manifest.get('mode')!='FULL_REAL':
        raise ValueError('模块四准备记录不是FULL_REAL，禁止把pilot结果写为论文指标')
    if manifest.get('input_source_sha256',{}).get('pilot') is not False:
        raise ValueError('模块四准备审计记录显示pilot不是false')
    if manifest.get('condition_valid_od')!=13402 or manifest.get('conditional_candidate_rows')!=130412:
        raise ValueError('历史候选与此前正式规模不符：需检查是否误读旧版本或试跑数据')
    cols=pd.read_csv(p['candidates'],nrows=0,encoding='utf-8-sig').columns.tolist()
    required=['trip_id','split_stage','chosen','candidate_id_TRACE_NOT_FEATURE']+SCALE_COLS
    if not set(required)<=set(cols):
        raise ValueError('候选文件列不完整：'+str(sorted(set(required)-set(cols))))
    banned={'history_vehicle_id_from_future','test_chosen','future_truth'}
    if banned & set(cols):raise ValueError('输入混入明显未来标签')
    if a.phase=='check':
        pr=pd.read_csv(p['profile'],nrows=0,encoding='utf-8-sig').columns
        require_cols(pd.DataFrame(columns=pr),['trip_id','history_n_prior']+PROFILE,'模块二画像')
        log('来源文件与必要字段存在，正式候选审计通过。check不拟合模型。')
        for k,x in p.items():log(f'{k}: {x}')
        return None,None,None,None,p,manifest
    take=list(dict.fromkeys(required))
    log('读取模块四已缓存的历史训练/验证条件候选（不搜索路网）')
    d=pd.read_csv(p['candidates'],usecols=take,encoding='utf-8-sig',low_memory=False,
                  dtype={'trip_id':str,'candidate_id_TRACE_NOT_FEATURE':str})
    if d.empty:raise ValueError('历史条件候选为空')
    if len(d)!=130412:raise ValueError('候选行数错误，预期130412，实际'+str(len(d)))
    if d.candidate_id_TRACE_NOT_FEATURE.isna().any() or d.candidate_id_TRACE_NOT_FEATURE.duplicated().any():
        raise ValueError('候选ID有空值或重复')
    if set(d.split_stage.dropna().unique())!={'model_train','validation'} or d.split_stage.isna().any():
        raise ValueError('候选集存在校准期/最后3天混入或split_stage为空')
    pp=pd.read_csv(p['profile'],usecols=lambda c:c in (['trip_id','history_n_prior']+PROFILE),
                   encoding='utf-8-sig',low_memory=False,dtype={'trip_id':str})
    require_cols(pp,['trip_id','history_n_prior']+PROFILE,'历史画像')
    if pp.trip_id.isna().any() or pp.trip_id.duplicated().any():raise ValueError('画像trip_id为空或重复')
    d=d.merge(pp[['trip_id','history_n_prior']+PROFILE],on='trip_id',how='left',validate='many_to_one')
    if d[PROFILE+['history_n_prior']].isna().any().any():
        raise ValueError('部分历史候选未连接到模块二严格历史画像')
    ids=pd.read_csv(p['index'],encoding='utf-8-sig',low_memory=False,
                    usecols=['trip_id','split_stage','start_time','VehicleID'],dtype={'trip_id':str,'VehicleID':str})
    if ids.trip_id.duplicated().any():raise ValueError('时间索引trip_id重复')
    odframe=d[['trip_id','split_stage']].drop_duplicates()
    if odframe.trip_id.duplicated().any():raise ValueError('相同trip_id跨训练/验证期，存在泄漏风险')
    check=odframe.merge(ids,on='trip_id',how='left',validate='one_to_one',suffixes=('_cand','_idx'))
    if len(check)!=13402 or check[['start_time','VehicleID']].isna().any().any():
        raise ValueError('有效OD与模块一时间索引无法一一对应')
    if (check.split_stage_cand!=check.split_stage_idx).any():
        raise ValueError('候选OD的时间分组与模块一索引不一致')
    stamp=pd.to_datetime(check.start_time,errors='coerce')
    traincheck=check.split_stage_cand.eq('model_train')
    valcheck=check.split_stage_cand.eq('validation')
    if stamp.isna().any() or not (stamp[traincheck].dt.strftime('%Y-%m-%d').between('2024-12-15','2024-12-23').all()
                  and stamp[valcheck].dt.strftime('%Y-%m-%d').between('2024-12-24','2024-12-28').all()):
        raise ValueError('历史训练、验证阶段的真实日期与2024-12-15~28时间划分不一致')
    if stamp[traincheck].max()>=stamp[valcheck].min():
        raise ValueError('训练时间与验证时间发生交叉')
    d=d.sort_values(['split_stage','trip_id','candidate_id_TRACE_NOT_FEATURE'],kind='mergesort').reset_index(drop=True)
    chosen=pd.to_numeric(d.chosen,errors='coerce')
    if not chosen.isin([0,1]).all():raise ValueError('chosen必须为0/1')
    d['chosen']=chosen.astype('int8')
    n=pd.to_numeric(d.history_n_prior,errors='coerce')
    if not np.isfinite(n.to_numpy(float)).all() or (n<0).any():raise ValueError('历史记录数不合法')
    d['history_n_prior']=n.astype(int)
    for dim in ('econ','comf'):
        arr=d[[f'history_{dim}_{i}' for i in range(4)]].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        if not np.isfinite(arr).all() or (arr<0).any() or (arr>1).any() or not np.allclose(arr.sum(axis=1),1,atol=1e-5):
            raise ValueError(f'模块二{dim}历史软画像不是合法归一化概率')
    audit=pd.read_csv(p['prepare_audit'],encoding='utf-8-sig',low_memory=False,
                      usecols=lambda c:c in ['trip_id','status','real_added_TRACE_NOT_FEATURE'],dtype={'trip_id':str})
    require_cols(audit,['trip_id','status','real_added_TRACE_NOT_FEATURE'],'准备审计')
    if audit.trip_id.duplicated().any():raise ValueError('准备审计存在重复OD')
    aud=check.merge(audit,on='trip_id',how='left',validate='one_to_one')
    if not (aud.status=='valid_labeled').all():
        raise ValueError('被选历史候选OD与valid_labeled审计不吻合')
    tr=d[d.split_stage=='model_train'].copy().reset_index(drop=True)
    va=d[d.split_stage=='validation'].copy().reset_index(drop=True)
    szt,tidt=groupinfo(tr)
    szv,tidv=groupinfo(va)
    if len(tidt)!=8647 or len(tidv)!=4755:
        raise ValueError(f'历史条件OD数不符合正式训练/验证规模：{len(tidt)} / {len(tidv)}')
    info=check[['trip_id','VehicleID','start_time']].merge(audit[['trip_id','real_added_TRACE_NOT_FEATURE']],on='trip_id',validate='one_to_one')
    log(f'时间隔离核验通过：训练OD={len(tidt):,}，验证OD={len(tidv):,}，总候选={len(d):,}，不涉及后三天真值')
    return tr,va,info,manifest,p,manifest


def groupinfo(df):
    if df.empty:raise ValueError('分组样本为空')
    counts=df.groupby('trip_id',sort=False).size()
    seq=np.repeat(counts.index.to_numpy(dtype=str),counts.to_numpy(int))
    if not np.array_equal(seq,df.trip_id.to_numpy(dtype=str)):
        raise ValueError('trip_id不是连续分组，禁止按行计算条件Logit')
    sizes=counts.to_numpy(int)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    if np.min(sizes)<2 or not np.array_equal(np.add.reduceat(df.chosen.to_numpy(float),starts),np.ones(len(sizes))):
        raise ValueError('每OD候选至少2条且chosen必须恰好一条')
    return sizes,counts.index.to_numpy(str)


class CandidateScaler:
    """与模块四原实现一致：训练OD行级中位数填充、均值/std标准化、裁剪[-8,8]。"""
    def fit(self,df):
        self.params=[]
        for c in SCALE_COLS:
            x=pd.to_numeric(df[c],errors='coerce').to_numpy(float)
            ok=np.isfinite(x)
            if ok.sum()<max(100,len(x)*.05):raise ValueError(f'{c}有效值过少')
            med=float(np.median(x[ok]));z=np.where(ok,x,med)
            std=float(z.std())
            self.params.append({'name':c,'median':med,'mean':float(z.mean()),
                                'std':std if std>1e-8 else 1.})
        return self
    def transform(self,df):
        arr=[]
        for p in self.params:
            x=pd.to_numeric(df[p['name']],errors='coerce').to_numpy(float)
            x=np.where(np.isfinite(x),x,p['median'])
            arr.append(np.clip((x-p['mean'])/p['std'],-8,8))
        z=np.stack(arr,axis=1)
        if not np.isfinite(z).all():raise ValueError('标准化结果含NaN/Inf')
        return z


def features(df,sc):
    z=sc.transform(df)
    ix={v:i for i,v in enumerate(SCALE_COLS)}
    b={'base':z[:,:10], 'ps_length':z[:,[ix['log_ps_length']]]}
    for dim,attrs in [('econ',ECON_ROUTE),('comf',COMF_ROUTE)]:
        q=df[[f'history_{dim}_{k}' for k in range(4)]].to_numpy(float)
        b[dim]=np.column_stack([(q[:,k]-0.25)*z[:,ix[col]] for k in range(4) for col in attrs])
    return b


def softmax(x,beta,sizes):
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    grp=np.repeat(np.arange(len(sizes)),sizes)
    v=x@beta
    exp=np.exp(v-np.maximum.reduceat(v,starts)[grp])
    denom=np.add.reduceat(exp,starts)
    p=exp/denom[grp]
    if not np.isfinite(p).all() or not np.allclose(np.add.reduceat(p,starts),1,atol=1e-9):
        raise FloatingPointError('条件概率错误')
    return p


def fit_logit(x,sizes,y,ridge,max_iter):
    x=np.asarray(x,np.float64,order='C');y=np.asarray(y,float)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    grp=np.repeat(np.arange(len(sizes)),sizes)
    n=len(sizes)
    if len(x)!=len(y) or len(y)!=int(sizes.sum()):raise ValueError('Logit行数不一致')
    def func(beta):
        v=x@beta
        mx=np.maximum.reduceat(v,starts)
        e=np.exp(v-mx[grp]);den=np.add.reduceat(e,starts);p=e/den[grp]
        loss=float(((mx+np.log(den)).sum()-y@v)/n+0.5*ridge*(beta@beta))
        grad=x.T@(p-y)/n+ridge*beta
        return loss,grad
    ans=minimize(func,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options={'maxiter':max_iter,'gtol':1e-6,'ftol':1e-10,'maxls':30})
    if not ans.success or not np.isfinite(ans.fun) or not np.isfinite(ans.x).all():
        raise RuntimeError('模型拟合未收敛（不允许伪装成成功结果）：'+str(ans.message))
    return ans


def evaluate(prob,sizes,tids,y,model,variant):
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    out=[]
    for trip,s,n in zip(tids,starts,sizes):
        t=y[s:s+n]
        sel=np.flatnonzero(t==1)
        if len(sel)!=1:raise ValueError('标签不是唯一选择')
        ix=int(sel[0]);p=float(prob[s+ix]);order=np.argsort(-prob[s:s+n],kind='mergesort')
        rank=int(np.flatnonzero(order==ix)[0]+1)
        out.append((trip,model,variant,int(n),p,rank,int(rank==1),1/rank,-math.log(max(p,1e-15))))
    return pd.DataFrame(out,columns=['trip_id','model','input_profile','n_candidates',
                                     'chosen_probability','chosen_rank','hit1','mrr','nll'])


def ref_from_train(train):
    """使用训练阶段独立OD的8D历史画像均值，绝不使用12月24—28验证真实路线/验证画像。"""
    once=train[['trip_id']+PROFILE].drop_duplicates('trip_id')
    v=once[PROFILE].to_numpy(float).mean(axis=0)
    for offset in [0,4]:
        if abs(v[offset:offset+4].sum()-1)>1e-6:raise ValueError('参考画像未归一化')
    if not np.isfinite(v).all():raise ValueError('参考画像含NaN')
    return v


def history_bin(n):
    return '0' if n==0 else ('1-5' if n<=5 else ('6-20' if n<=20 else '>20'))


def pair_merge(od,base,ref):
    first=od.merge(base[['trip_id','nll','mrr','hit1','chosen_probability','chosen_rank']],
                   on='trip_id',validate='one_to_one')
    first=first.merge(ref[['trip_id','nll','mrr','hit1','chosen_probability','chosen_rank']],
                      on='trip_id',suffixes=('_actual','_reference'),validate='one_to_one')
    first['nll_gain_reference_minus_actual']=first.nll_reference-first.nll_actual
    first['mrr_gain_actual_minus_reference']=first.mrr_actual-first.mrr_reference
    first['hit1_gain_actual_minus_reference']=first.hit1_actual-first.hit1_reference
    return first


def sample_ci(d,a,level='vehicle'):
    """同一VehicleID多个OD按车辆聚类Bootstrap；差值定义为“越大代表真实画像越好”。"""
    values=pd.to_numeric(d['diff'],errors='coerce').to_numpy(float)
    ok=np.isfinite(values)
    if not ok.any():return {'n_od':0,'n_vehicles':0,'mean':None,'ci_low':None,'ci_high':None,'p_two_sided':None}
    d=d.loc[ok].copy();values=values[ok]
    vehicle=d.VehicleID.astype(str).to_numpy()
    codes,keys=pd.factorize(vehicle,sort=False)
    s=np.bincount(codes,weights=values).astype(float)
    n=np.bincount(codes).astype(float)
    k=len(s);rng=np.random.default_rng(a.seed + int(len(values)) + int(abs(values.mean())*1e9)%10007)
    boots=[]
    for _ in range(a.bootstrap):
        take=rng.integers(0,k,size=k) if level=='vehicle' else None
        if take is not None:boots.append(float(s[take].sum()/n[take].sum()))
        else:boots.append(float(values[rng.integers(0,len(values),len(values))].mean()))
    ci=np.quantile(boots,[0.025,0.975])
    obs=float(values.mean())
    # 符号翻转检验：按车辆汇总独立地随机改变符号，检验“均值无方向性”探索性假设。
    exceed=0
    for _ in range(a.permutations):
        if level=='vehicle':
            signs=rng.choice([-1.,1.],size=k)
            stat=float((s*signs).sum()/len(values))
        else:
            signs=rng.choice([-1.,1.],size=len(values))
            stat=float(np.mean(values*signs))
        exceed+=abs(stat)>=abs(obs)-1e-14
    p=(exceed+1)/(a.permutations+1)
    return {'n_od':int(len(values)),'n_vehicles':int(k),'mean':obs,
            'ci_low':float(ci[0]),'ci_high':float(ci[1]),'p_two_sided':float(p),
            'frac_positive':float((values>1e-12).mean()),
            'frac_negative':float((values< -1e-12).mean()),
            'frac_tie':float((np.abs(values)<=1e-12).mean())}


def holm(ps):
    x=np.asarray(ps,float)
    idx=np.argsort(x)
    ans=np.zeros(len(x),float)
    prev=0.
    for rank,i in enumerate(idx):
        prev=max(prev,min(1.,(len(x)-rank)*x[i]));ans[i]=prev
    return ans


def make_plot(out,compare,paired):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        have={f.name for f in font_manager.fontManager.ttflist}
        for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans']:
            if f in have:
                plt.rcParams['font.sans-serif']=[f,'DejaVu Sans'];break
        plt.rcParams['axes.unicode_minus']=False
    except ImportError:
        log('matplotlib未安装：仅跳过图，不影响任何真实统计CSV');return
    subset=compare[compare.input_profile.isin(['actual','reference','baseline'])]
    pivot=subset.pivot(index='model',columns='input_profile',values='mean_nll')
    fig,ax=plt.subplots(figsize=(9.7,4.8))
    x=np.arange(len(pivot));w=.25
    for k,(label,offset) in enumerate([('actual',-.25),('reference',0),('baseline',.25)]):
        # 只在含该变体的模型绘制；不凭空填入基线值。
        vals=pivot[label].to_numpy(float) if label in pivot else np.full(len(pivot),np.nan)
        good=np.isfinite(vals)
        if good.any():ax.bar(x[good]+offset,vals[good],width=w,label={'actual':'真实画像','reference':'群体参考画像','baseline':'基础PSL'}[label])
    ax.set_xticks(x,labels=[q.replace('UDPPM_','').replace('_PSL_L2','').replace('PSL_LengthWeighted_L2','PSL') for q in pivot.index])
    ax.set_ylabel('验证期NLL（越低越好）');ax.set_title('历史验证期：真实画像与参考画像的选择概率解释能力')
    ax.grid(axis='y',alpha=.16);ax.legend();fig.tight_layout()
    fig.savefig(out/'07_验证期真实参考画像_NLL对比.png',dpi=220,bbox_inches='tight');plt.close(fig)
    sub=paired[(paired.comparison_type=='真实画像_vs_参考画像') & (paired.history_group=='全部')]
    if not sub.empty:
        fig,ax=plt.subplots(figsize=(9.2,4.7))
        y=np.arange(len(sub));v=sub['mean_gain'].to_numpy(float)
        low=sub['ci_low'].to_numpy(float);hi=sub['ci_high'].to_numpy(float)
        ax.errorbar(v,y,xerr=np.vstack([v-low,hi-v]),fmt='o',capsize=4)
        ax.axvline(0,color='gray',linestyle='--',linewidth=1)
        ax.set_yticks(y,[s.replace('UDPPM_','').replace('_PSL_L2','') for s in sub.model])
        ax.set_xlabel('NLL改善：参考画像NLL − 真实画像NLL（>0说明真实画像更好）')
        ax.set_title('模型输入画像敏感性：验证OD配对均值与95%置信区间')
        ax.grid(axis='x',alpha=.18);fig.tight_layout()
        fig.savefig(out/'08_真实画像NLL增量_车辆聚类Bootstrap95CI.png',dpi=220,bbox_inches='tight');plt.close(fig)


def analyze(a):
    tr,va,odinfo,manifest,files,_=load_source(a)
    tr_size,tr_ids=groupinfo(tr);va_size,va_ids=groupinfo(va)
    log('仅从模型训练期有效OD历史画像估计群体参考分布，不读取验证期画像计算均值')
    reference=ref_from_train(tr)
    log('训练期群体参考画像：'+', '.join(f'{x:.6f}' for x in reference))
    sc=CandidateScaler().fit(tr)
    tx=features(tr,sc)
    vx=features(va,sc)
    va_ref=va.copy()
    for c,value in zip(PROFILE,reference):va_ref[c]=float(value)
    vr=features(va_ref,sc)
    ytr=tr.chosen.to_numpy(float);yv=va.chosen.to_numpy(float)
    valmeta=va[['trip_id','history_n_prior']].drop_duplicates('trip_id').copy()
    valmeta['history_group']=valmeta.history_n_prior.map(history_bin)
    valmeta=valmeta.merge(odinfo[['trip_id','VehicleID','real_added_TRACE_NOT_FEATURE']],on='trip_id',validate='one_to_one')
    fitted=[];valids=[];pairs=[];refits=[];coeff=[]
    baseline=None
    for model,blocks in MODELS.items():
        X=np.column_stack([tx[q] for q in blocks])
        V=np.column_stack([vx[q] for q in blocks])
        log(f'训练期拟合 {model}，参数维度={X.shape[1]}，L2={a.ridge}')
        result=fit_logit(X,tr_size,ytr,a.ridge,a.max_iter)
        prob=softmax(V,result.x,va_size)
        actual=evaluate(prob,va_size,va_ids,yv,model,'actual' if model!='PSL_LengthWeighted_L2' else 'baseline')
        valids.append(actual)
        fitted.append({'model':model,'n_coefficients':len(result.x),'iterations':int(result.nit),
                       'train_objective':float(result.fun),'converged':bool(result.success)})
        coeff.append({'model':model,'beta':result.x.tolist(),'n_coefficients':len(result.x)})
        log(f'   验证 NLL={actual.nll.mean():.6f} MRR={actual.mrr.mean():.6f} Hit1={actual.hit1.mean():.6f}')
        if model=='PSL_LengthWeighted_L2':baseline=actual;continue
        R=np.column_stack([vr[q] for q in blocks])
        p_ref=softmax(R,result.x,va_size)
        ref=evaluate(p_ref,va_size,va_ids,yv,model,'reference')
        valids.append(ref)
        pairs.append(pair_merge(valmeta,actual,ref))
        # 记录一条OD实际路线模型概率变化，不复用先前已看过验证结果的全期重拟合模型
        delta=ref[['trip_id','chosen_probability']].merge(actual[['trip_id','chosen_probability']],
               on='trip_id',suffixes=('_reference','_actual'),validate='one_to_one')
        delta['model']=model
        delta['delta_actual_minus_reference']=delta.chosen_probability_actual-delta.chosen_probability_reference
        refits.append(delta)
    # 汇总所有方法与两类画像输入，验证条目必须一一对应。
    allval=pd.concat(valids,ignore_index=True)
    if allval.groupby(['model','input_profile']).size().nunique()!=1:
        raise AssertionError('不同方案验证OD数量不一致')
    summary=allval.groupby(['model','input_profile'],sort=False).agg(
        validation_od=('trip_id','nunique'),mean_nll=('nll','mean'),mean_mrr=('mrr','mean'),
        hit1=('hit1','mean'),mean_chosen_probability=('chosen_probability','mean')).reset_index()
    overall=[];strata=[];compare_types=[]
    for model,pair in zip(list(MODELS)[1:],pairs):
        d=pair.copy();d['model']=model
        for group in ['全部','0','1-5','6-20','>20']:
            s=d if group=='全部' else d[d.history_group==group]
            if s.empty:continue
            for metric,label in [('nll_gain_reference_minus_actual','nll'),
                                  ('mrr_gain_actual_minus_reference','mrr'),
                                  ('hit1_gain_actual_minus_reference','hit1')]:
                x=s[['trip_id','VehicleID',metric]].rename(columns={metric:'diff'})
                stat=sample_ci(x,a)
                row={'comparison_type':'真实画像_vs_参考画像','model':model,'history_group':group,
                     'metric':label,'meaning':'正值表示真实画像优于群体参考','mean_gain':stat['mean'],
                     'ci_low':stat['ci_low'],'ci_high':stat['ci_high'],
                     'p_two_sided':stat['p_two_sided'],'n_od':stat['n_od'],
                     'n_vehicles':stat['n_vehicles'],'frac_improve':stat.get('frac_positive'),
                     'frac_worsen':stat.get('frac_negative'),'frac_tie':stat.get('frac_tie')}
                (overall if group=='全部' else strata).append(row)
    # 与长度加权PSL做严格相同训练数据/同ridge的模型增量比较，不更换评价OD。
    basen=baseline[['trip_id','nll','mrr','hit1']].rename(columns={'nll':'nll_baseline','mrr':'mrr_baseline','hit1':'hit1_baseline'})
    for model,pair in zip(list(MODELS)[1:],pairs):
        d=pair.merge(basen,on='trip_id',validate='one_to_one')
        for metric,expr in [('nll',d.nll_baseline-d.nll_actual),
                            ('mrr',d.mrr_actual-d.mrr_baseline),
                            ('hit1',d.hit1_actual-d.hit1_baseline)]:
            x=d[['VehicleID']].copy();x['diff']=expr
            stat=sample_ci(x,a)
            overall.append({'comparison_type':'UDPPM真实画像模型_vs_基础PSL','model':model,
                'history_group':'全部','metric':metric,'meaning':'正值表示带画像交互模型优于基础PSL',
                'mean_gain':stat['mean'],'ci_low':stat['ci_low'],'ci_high':stat['ci_high'],
                'p_two_sided':stat['p_two_sided'],'n_od':stat['n_od'],
                'n_vehicles':stat['n_vehicles'],'frac_improve':stat.get('frac_positive'),
                'frac_worsen':stat.get('frac_negative'),'frac_tie':stat.get('frac_tie')})
    overall=pd.DataFrame(overall)
    strata=pd.DataFrame(strata)
    # Holm控制同一主分析中六个预先列出的NLL比较，不能把点估计当独立显著性证据。
    mask=(overall.metric=='nll')
    overall['p_holm_nll_family']=np.nan
    overall.loc[mask,'p_holm_nll_family']=holm(overall.loc[mask,'p_two_sided'].to_numpy(float))
    # 旧模块四条件模型验证分数允许做【描述性】对照：不替代本实验独立复算。
    old_path=a.m4/'05B_六模型验证均值指标.csv'
    old=pd.DataFrame()
    if old_path.exists():
        old=pd.read_csv(old_path,encoding='utf-8-sig')
    audit2=read_json(files['prepare_manifest'])
    val_audit=odinfo[odinfo.trip_id.isin(set(va_ids))]
    n_added=int(pd.to_numeric(val_audit.real_added_TRACE_NOT_FEATURE,errors='coerce').fillna(0).sum())
    out=a.out
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f'输出目录非空：{out}；为了保护已有结果请改 --out，确认覆盖才加 --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    saved={}
    def put(df,name):
        target=out/name;savecsv(df,target);saved[name]=sha256(target)
    # 不导出含真实选择线路边序列/VehicleID的原候选明细
    put(allval,'01_历史验证逐OD_条件选择NLL_MRR_Hit1.csv.gz')
    put(summary,'02_四模型_真实与参考画像_验证期指标汇总.csv')
    put(overall,'03_画像增量及PSL对照_配对统计车辆Bootstrap.csv')
    put(strata,'04_按历史轨迹数量分层_画像增量统计.csv')
    put(pd.concat(refits,ignore_index=True),'05_验证OD真实与参考画像_真实选择路线概率变化.csv.gz')
    put(pd.DataFrame(fitted),'06_四模型_训练收敛和目标值.csv')
    if not old.empty:put(old,'07_原模块四历史验证指标_仅描述性参照.csv')
    # 保存本次仅训练期拟合出的参数，便于后来逐行复算；不覆盖模块四原始模型。
    parameter_file=out/'09_历史验证模型与标准化参数_可复现.json'
    jwrite(parameter_file, {'training_period':'2024-12-15..23',
                            'validation_period':'2024-12-24..28',
                            'ridge':a.ridge,'feature_order':SCALE_COLS,
                            'profile_column_order':PROFILE,
                            'scaler_params':sc.params,
                            'train_only_reference_profile_8d':reference.tolist(),
                            'model_parameters':coeff})
    saved[parameter_file.name]=sha256(parameter_file)
    manifest_out={
        'version':'m68_historical_validation_profile_conditional_choice_1.0',
        'created_at':datetime.now().isoformat(),
        'experiment':'MODEL_TRAIN_2024_12_15_TO_23__VALIDATION_2024_12_24_TO_28',
        'never_read_2024_12_29_31_truth':True,
        'training_candidate_generation_rerun':False,
        'validation_used_for_training_or_reference':False,
        'reference_origin':'mean of unique MODEL_TRAIN OD prior 8D profiles (not validation, not pre-28 aggregate)',
        'reference_profile_8d':reference.tolist(),
        'same_frozen_model_actual_vs_reference':True,
        'scaler_fit_only_on_training':True,
        'ridge_fixed_without_reselection':a.ridge,
        'ridge_note':'0.003曾在旧实验验证集选过；本实验沿用属探索性复核，不能宣称全新独立验证。',
        'bootstrap_unit':'VehicleID cluster',
        'permutation_unit':'VehicleID sign-flip',
        'n_train_od':int(len(tr_ids)),'n_validation_od':int(len(va_ids)),
        'n_train_candidates':int(len(tr)),'n_validation_candidates':int(len(va)),
        'validation_real_added_od':n_added,
        'validation_real_added_fraction':float(n_added/len(va_ids)),
        'source_path_and_sha256':{key:{'path':str(file),'sha256':sha256(file)} for key,file in files.items()},
        'old_prepare_audit_mode':audit2.get('mode'),
        'model_fits':fitted,
        'output_sha256':saved,
        'method_limitations':[
            '历史验证条件候选可补入真实HMM路线，因此NLL/Hit1/MRR不能称为未知OD纯自然候选预测率。',
            '该验证日期曾用于模块四旧方案正则化选择；本补充研究是探索性而非未经使用的全新测试。',
            '同一冻结模型的真实/群体参考输入比较是画像敏感性，不代表重新拟合无画像模型。',
            '与PSL基线的差异包含更多交互特征和重新拟合参数，不是UDPPM++聚类算法本身的因果增益。',
            '群体参考画像只用训练期OD的历史软归属均值，非真实经济/舒适需求标签。',
            '同车多个OD使用VehicleID聚类重抽样，但观测非随机试验，统计结果是探索性的。',
        ]}
    jwrite(out/'00_严格时间与来源冻结审计.json',manifest_out)
    make_plot(out,summary,overall)
    log('==== 真实验证集主要结果（NLL越低越好） ====')
    print(summary[['model','input_profile','validation_od','mean_nll','mean_mrr','hit1']].to_string(index=False),flush=True)
    print('\n=== NLL主比较（改善>0表示画像更好） ===',flush=True)
    print(overall[overall.metric=='nll'][['comparison_type','model','n_od','mean_gain','ci_low','ci_high','p_holm_nll_family']].to_string(index=False),flush=True)
    log('完成：读取的是已指定的历史候选和时间索引；实验解释需查看00号审计中的研究局限。输出 '+str(out))


def self_test():
    # 纯内存公式测试，不输出数据/图，不当作科研实验数据。
    size=np.array([3,2]);x=np.array([[.1,1.],[.5,0.],[.2,-.5],[.9,.6],[.3,1.]])
    beta=np.array([.2,-.1]);p=softmax(x,beta,size)
    assert np.allclose([p[:3].sum(),p[3:].sum()],[1,1],atol=1e-12)
    y=np.array([0,1,0,1,0],float)
    fit=fit_logit(x,size,y,.1,100)
    assert fit.success and np.isfinite(fit.fun)
    # 确认交互项每维center=.25，且无歧义的数组列顺序
    assert len(MODELS['UDPPM_Dual_PSL_L2'])==4
    print('公式单元测试通过（仅软件测试；未产生论文结果）')


def main():
    parser=argparse.ArgumentParser(description='补充实验：历史验证期真实8维画像 vs 训练期群体参考画像，条件NLL与PSL对照')
    parser.add_argument('--phase',choices=['check','analyze','self-test'],default='check')
    parser.add_argument('--m1',type=Path,default=M1)
    parser.add_argument('--m2',type=Path,default=M2)
    parser.add_argument('--m4',type=Path,default=M4)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--ridge',type=float,default=0.003)
    parser.add_argument('--max-iter',type=int,default=350)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--permutations',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--overwrite',action='store_true')
    a=parser.parse_args()
    if a.ridge<=0 or not math.isfinite(a.ridge):raise ValueError('--ridge必须为正且有限')
    if a.max_iter<10 or a.bootstrap<100 or a.permutations<100 or a.seed<0:
        raise ValueError('迭代次数、重抽样次数或随机种子不合法')
    if a.phase=='self-test':self_test();return
    if a.phase=='check':load_source(a);return
    analyze(a)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        log('用户终止：原模块数据与模型均未修改')
        raise SystemExit(130)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        raise SystemExit(f'6step-68停止：{type(ex).__name__}: {ex}')
