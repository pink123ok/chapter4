# -*- coding: utf-8 -*-
"""
第四章模块二：原第三章UDPPM++经济/舒适6+6行为特征的时间无泄漏历史画像

输入来自【第四章模块一】的真实输出：
  01_全量OD时间分组索引.csv
  02_四阶段样本规模统计.csv
  05_最后3天_纯OD推理输入_绝无真实路线.csv
  06_最后3天_历史画像匹配索引_受控.csv
  09_前28天_第三章原始6加6特征.csv.gz
  11_完整性防泄漏审计.json
不读取 07_最后3天真实路径答案。

重要：
1. 本脚本完整复用第三章 step7_udppm_all_experiments.py 中的 TrainOnlyMAD、
   UDPPMSoft、squared_norm_dist 实现，并严格保留 6+6 主行为特征，K=4+4。
2. 该实验使用第三章行为特征版（semantic_mode='none'），不使用原第三章整月
   PCA8语义数据，避免语义PCA已见未来数据导致泄漏。不是在声称重现第三章含
   BERT+KG辅助向量的正式融合模型；如论文要求完全一致，需先重新生成
   严格训练期拟合的语义PCA后扩展本实验，不能直接套用整月语义特征。
3. calibration仅用于原型训练；model_train历史画像只看出发前已结束的路径；
   validation画像冻结在12月24日00:00；future_test画像冻结在12月29日00:00。
4. 最后3天不读取真实轨迹或测试行为特征；供下游使用的测试画像不含VehicleID。
5. 本程序不生成或杜撰任何论文实验数值；必须从真实模块一文件计算。

安装：python -m pip install -U numpy pandas scipy scikit-learn matplotlib
执行：python "第四章_模块二_UDPPMpp历史双维偏好画像_严格时间隔离.py"
重跑：在命令末尾加 --overwrite
查看路径：python 脚本.py --help
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'C:\study\轨迹数据预处理\曼谷GPS_正式清洗_202412\FULL_全量\第四章_28天训练_3天独立测试')
MODULE1 = ROOT / '模块一_时间划分与隔离'
MODULE2 = ROOT / '模块二_UDPPMpp双维历史画像'
ECON = ['path_len_km','travel_time_min','avg_speed_kmh_geom',
        'ratio_motorway','ratio_major','ratio_local']
COMF = ['slow_ratio','turns_per_km','ratio_local','ratio_motorway',
        'oneway_ratio','segment_density']
TRANSFORM_LOG={'path_len_km','travel_time_min','segment_density','turns_per_km','detour_index'}
STAGES=['calibration','model_train','validation','future_test']
CAL_END=pd.Timestamp('2024-12-15 00:00:00')
TRAIN_END=pd.Timestamp('2024-12-24 00:00:00')
TEST_BEGIN=pd.Timestamp('2024-12-29 00:00:00')
TEST_END=pd.Timestamp('2025-01-01 00:00:00')
PROB_COLS=[f'history_econ_{i}' for i in range(4)]+[f'history_comf_{i}' for i in range(4)]
Q_COLS=[f'econ_prob_{i}' for i in range(4)]+[f'comf_prob_{i}' for i in range(4)]


def log(s):
    print(f'[{datetime.now():%H:%M:%S}] {s}',flush=True)

def numeric_col(df,k):
    return pd.to_numeric(df[k],errors='coerce').astype(float).to_numpy()


class TrainOnlyMAD:
    """先按训练集1%-99%截尾，再log1p（对指定非负列），训练集median/MAD缩放。
    缺失按训练集median补全，常量列设零；严格防止验证集统计泄漏。
    """
    def __init__(self, cols, clip=8.):
        self.cols=list(cols); self.clip=float(clip)
    def fit(self, frame):
        self.parts=[]
        for name in self.cols:
            x=numeric_col(frame,name);finite=np.isfinite(x)
            if not finite.any():
                raise ValueError(f'特征列全缺失：{name}')
            raw=x[finite]
            lo,hi=np.quantile(raw,[.01,.99])
            x=np.clip(x,lo,hi)
            if name in TRANSFORM_LOG:
                x=np.log1p(np.maximum(0,x))
            yy=x[np.isfinite(x)];med=float(np.median(yy))
            mad=float(np.median(np.abs(yy-med)))
            scale=1.4826*mad
            if scale<1e-7:
                q=np.quantile(np.abs(yy-med),.75)
                scale=float(q/0.6745) if q>1e-7 else 1.
            self.parts.append({'name':name,'clip_low':float(lo),'clip_high':float(hi),
                               'median':med,'scale':scale,'log':name in TRANSFORM_LOG})
        return self
    def transform(self,frame):
        out=np.empty((len(frame),len(self.cols)),dtype=np.float32)
        for j,p in enumerate(self.parts):
            v=numeric_col(frame,p['name'])
            v=np.clip(v,p['clip_low'],p['clip_high'])
            if p['log']:v=np.log1p(np.maximum(0,v))
            v[~np.isfinite(v)]=p['median']
            out[:,j]=np.clip((v-p['median'])/p['scale'],-self.clip,self.clip)
        return out
    def fit_transform(self,df):return self.fit(df).transform(df)


def squared_norm_dist(X,C):
    xx=(X.astype(np.float64)**2).sum(1)[:,None]
    cc=(C.astype(np.float64)**2).sum(1)[None,:]
    return np.maximum(0,xx+cc-2*(X.astype(np.float64)@C.astype(np.float64).T))/X.shape[1]


class UDPPMSoft:
    """可运行的温度化 K-means + 三种可关停正则。

    J = mean_i sum_k q_ik ||x_i-c_k||^2/d  + tau mean_i sum_k q_ik log q_ik
        + λ_bal KL(mean_i q_i || uniform)
        + λ_sep mean_{k<l} exp(-||c_k-c_l||²/(d alpha))
        + λ_sm mean_{(i,j)属于同车相邻} ||q_i-q_j||²/2

    q 更新采用在固定中心下的镜像/固定点迭代；中心更新由软均值与
    分离梯度修正组合，迭代到收敛。它是近似交替优化，非全局最优保证。
    时间正则仅在训练集上使用；测试路径预测不读取该车辆未来轨迹。
    """
    def __init__(self,k=4,tau=.8,bal=.03,sep=.02,sm=.01,
                 iters=35, seed=42):
        self.k=int(k);self.tau=float(tau);self.bal=float(bal)
        self.sep=float(sep);self.sm=float(sm);self.iters=int(iters);self.seed=int(seed)
    def fit(self,X,pairs=(None,None)):
        X=np.asarray(X,dtype=np.float32)
        n,d=X.shape
        if self.k>=n:raise ValueError('k不能大于样本数')
        ini=KMeans(n_clusters=self.k,n_init=3,max_iter=90,random_state=self.seed)
        ini.fit(X)
        C=ini.cluster_centers_.astype(np.float64)
        d2=squared_norm_dist(C,C)
        top=d2[np.triu_indices(self.k,1)]
        alpha=max(float(np.median(top)),.1)
        a,b=pairs
        if a is None:a=np.array([],dtype=np.int32);b=np.array([],dtype=np.int32)
        a=np.asarray(a,dtype=np.int64);b=np.asarray(b,dtype=np.int64)
        Q=softmax(-squared_norm_dist(X,C)/self.tau,axis=1)
        self.history_=[]
        for t in range(self.iters):
            D=squared_norm_dist(X,C)
            p=np.maximum(Q.mean(0),1e-8)
            # 均衡项导数来自KL(pi || 1/K)：lambda_bal*log(pi*K)，平移常数在softmax中抵消
            penalty=self.bal*np.log(p*self.k)[None,:]
            if self.sm and len(a):
                grad=np.zeros_like(Q)
                np.add.at(grad,a,Q[a]-Q[b]);np.add.at(grad,b,Q[b]-Q[a])
                # 单个软分配的梯度与目标中mean_i的尺度对齐
                penalty=penalty+self.sm*n/max(len(a),1)*grad
            Qnext=softmax(-(D+penalty)/self.tau,axis=1)
            Q=.45*Q+.55*Qnext
            w=np.maximum(Q.sum(0),1e-9)
            means=(Q.T@X)/w[:,None]
            Cnew=means.astype(np.float64)
            if self.sep>0:
                dd=squared_norm_dist(C,C)
                expo=np.exp(-dd/alpha)
                np.fill_diagonal(expo,0.)
                # ∂ Jsep / ∂ck = -2*sep/(pairs*alpha*d) Σ exp(-dist²/(d alpha))(ck-cl)
                gradsep=-2*self.sep/(max(self.k*(self.k-1)/2,1)*alpha*d)*(
                    (expo.sum(axis=1)[:,None]*C)-(expo@C))
                # 拟合项对中心梯度 2*pi/d*(C-mean)，由此得到修正量
                drift=-gradsep*d/(2*np.maximum(w/n,.03)[:,None])
                drift=np.clip(drift,-.25,.25)
                Cnew+=drift
            delta=float(np.mean(np.abs(Cnew-C)))
            C=.35*C+.65*Cnew
            if t%3==0 or t==self.iters-1:
                top_cur=squared_norm_dist(C,C)[np.triu_indices(self.k,1)]
                self.history_.append({'epoch':t+1,'change':delta,
                                      'fit':float(np.mean((Q*D).sum(1))),
                                      'balance_kl':float(np.sum(p*np.log(p*self.k))),
                                      'sm_pair_mse':float(np.mean((Q[a]-Q[b])**2)) if len(a) else np.nan,
                                      'sep_kernel':float(np.mean(np.exp(-top_cur/alpha)))})
            if delta<1e-5 and t>=10:break
        self.centers_=C.astype(np.float32)
        self.train_q_=Q.astype(np.float32)
        self.alpha_=alpha
        return self
    def predict_proba(self,X):
        # 验证集/新车辆：不使用同车邻接信息，避免测试未来观测影响当前预测
        return softmax(-squared_norm_dist(np.asarray(X,dtype=np.float32),self.centers_)/self.tau,axis=1).astype(np.float32)


def read_csv_real(path, *, required=None, dtype=None, usecols=None):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'真实实验输入缺失或空文件：{path}\n请先完成模块一并确认该文件存在。')
    h=pd.read_csv(path,nrows=0,encoding='utf-8-sig').columns.tolist()
    if len(h)!=len(set(h)):
        raise ValueError(f'输入表有重复列名：{path}')
    missing=sorted(set(required or [])-set(h))
    if missing:
        raise ValueError(f'{path.name}缺少必要字段：{missing}\n实际字段：{h}')
    df=pd.read_csv(path,encoding='utf-8-sig',low_memory=False,dtype=dtype,usecols=usecols)
    if df.empty:raise ValueError(f'真实文件没有记录：{path}')
    return df


def parse_local_time(s, label):
    t=pd.to_datetime(s,errors='coerce')
    if t.isna().any():
        raise ValueError(f'{label}存在无法解析的日期；例：{s[t.isna()].head(5).tolist()}')
    if t.dt.tz is not None:
        # 模块一按曼谷日期分组；本模块所有比较都统一到曼谷本地无时区时钟。
        t=t.dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
    return t.astype('datetime64[ns]')


def stable_ids(df, label):
    if df['trip_id'].isna().any() or df['trip_id'].duplicated().any():
        raise ValueError(f'{label}存在空/重复trip_id，例：{df.loc[df.trip_id.duplicated(),"trip_id"].head().tolist()}')
    if not df['trip_id'].astype(str).str.strip().ne('').all():
        raise ValueError(f'{label}存在空字符串trip_id')


def check_probability(mat, label):
    a=np.asarray(mat,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=4 or not np.isfinite(a).all():
        raise ValueError(f'{label}概率维数或有限性异常：shape={a.shape}')
    if (a< -1e-7).any() or (a>1+1e-7).any() or not np.allclose(a.sum(1),1,atol=2e-5):
        raise ValueError(f'{label}概率未归一化或超出[0,1]')


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_csv(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.9g',compression='infer')


def save_json(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8-sig')


def validate_module1(a):
    directory=a.module1
    files={
       'index':directory/'01_全量OD时间分组索引.csv',
       'stat':directory/'02_四阶段样本规模统计.csv',
       'blind':directory/'05_最后3天_纯OD推理输入_绝无真实路线.csv',
       'ctl':directory/'06_最后3天_历史画像匹配索引_受控.csv',
       'feat':directory/'09_前28天_第三章原始6加6特征.csv.gz',
       'audit':directory/'11_完整性防泄漏审计.json',
    }
    for p in files.values():
        if not p.is_file():raise FileNotFoundError(f'模块一缺少：{p}\n请先确保上传前的本机完整输出存在。')
    meta=json.loads(files['audit'].read_text(encoding='utf-8-sig'))
    if meta.get('version')!='chapter4_module1_chrono_split_v1.0':
        raise ValueError('模块一审计版本不匹配，禁止混用不同数据批次')
    policy=meta.get('time_policy',{})
    if (policy.get('calibration')!='2024-12-01 through 2024-12-14' or
        policy.get('model_train')!='2024-12-15 through 2024-12-23' or
        policy.get('validation')!='2024-12-24 through 2024-12-28' or
        policy.get('future_test')!='2024-12-29 through 2024-12-31'):
        raise ValueError('模块一阶段边界与本模块约定不同，禁止自动推测')
    if meta.get('feature_policy',{}).get('econ_exact_6')!=ECON or meta.get('feature_policy',{}).get('comfort_exact_6')!=COMF:
        raise ValueError('第三章6+6特征顺序不匹配，禁止改动后静默继续')
    if not all(meta.get('checks',{}).get(k) is True for k in [
         'dev_test_trip_id_disjoint','master_ids_subset_of_step6',
         'master_vehicle_and_timestamps_match_step6','blind_od_columns_exact_whitelist']):
        raise ValueError('模块一存在未通过的数据来源或隔离审计项')
    ids_dtype={'trip_id':'string','VehicleID':'string',
               'origin_node':'string','destination_node':'string'}
    idx=read_csv_real(files['index'],dtype=ids_dtype,required=[
      'trip_id','VehicleID','start_time','end_time','split_stage'])
    stat=read_csv_real(files['stat'],required=['stage','n_od'])
    blind=read_csv_real(files['blind'],dtype={'trip_id':'string'},required=[
      'trip_id','start_time','origin_node','destination_node'])
    ctl=read_csv_real(files['ctl'],dtype={'trip_id':'string','VehicleID':'string'},required=[
      'trip_id','VehicleID','start_time','has_history_vehicle'])
    # 除了用户许可的必要控制列，必须拒绝测试真实标签或行为字段进入下游画像输入。
    forbidden={'chosen','edge_seq_uvk_json','path_len_km','travel_time_min',
      'duration_min','slow_ratio','avg_speed_kmh_geom','is_real_added','real_added'}
    if set(blind.columns)&forbidden:
        raise ValueError('纯OD输入含真实答案或行为字段：'+repr(sorted(set(blind.columns)&forbidden)))
    if 'VehicleID' in blind.columns or 'end_time' in blind.columns:
        raise ValueError('纯OD输入包含车辆身份或结束时间，违反模块一隔离白名单')
    feat_cols=['trip_id','VehicleID','start_time','end_time','split_stage']+list(dict.fromkeys(ECON+COMF))
    feat=read_csv_real(files['feat'],dtype={'trip_id':'string','VehicleID':'string'},required=feat_cols)
    stable_ids(idx,'模块一时间索引');stable_ids(feat,'模块一6+6行为特征')
    stable_ids(blind,'模块一盲测OD');stable_ids(ctl,'模块一受控画像索引')
    for x in [idx,feat,blind,ctl]:
        x['start_time']=parse_local_time(x['start_time'], 'start_time')
    for x in [idx,feat]:
        x['end_time']=parse_local_time(x['end_time'], 'end_time')
    if (idx.end_time<=idx.start_time).any() or (feat.end_time<=feat.start_time).any():
        raise ValueError('存在结束时间不晚于开始时间的行程')
    if not idx.split_stage.isin(STAGES).all() or not feat.split_stage.isin(STAGES[:3]).all():
        raise ValueError('时间索引或历史行为数据存在未知阶段，或历史行为出现测试集')
    exp=meta.get('counts',{}).get('stage_od',{})
    counts=idx.split_stage.value_counts().to_dict()
    for s in STAGES:
        expected=int(exp.get(s,-1))
        if counts.get(s,0)!=expected:
            raise ValueError(f'{s}真实行数 {counts.get(s,0)} 与模块一审计 {expected} 不一致')
        statrow=stat.loc[stat.stage==s,'n_od']
        if len(statrow)!=1 or int(statrow.iloc[0])!=expected:
            raise ValueError(f'四阶段统计文件{s}与索引不一致')
    dev=idx.loc[idx.split_stage!='future_test'].copy()
    test=idx.loc[idx.split_stage=='future_test'].copy()
    if set(dev.trip_id)!=set(feat.trip_id):
        raise ValueError(f'前28天行为特征trip_id与索引不一致，左右差集数量：'
                         f'{len(set(dev.trip_id)-set(feat.trip_id))} / {len(set(feat.trip_id)-set(dev.trip_id))}')
    if set(test.trip_id)!=set(ctl.trip_id) or set(test.trip_id)!=set(blind.trip_id):
        raise ValueError('最后3天纯OD、受控索引和主时间索引ID不一致')
    # 对照模块一1/2/3期的原始ID、车辆、时间以及阶段；不使用测试真值或测试行为。
    merged=dev[['trip_id','VehicleID','start_time','end_time','split_stage']].merge(
       feat,on='trip_id',validate='one_to_one',suffixes=('_idx',''))
    for c in ['VehicleID','start_time','end_time','split_stage']:
        if not (merged[c].astype(str)==merged[f'{c}_idx'].astype(str)).all():
            raise ValueError(f'模块一特征与主索引的{c}存在映射不一致')
    hist=merged.drop(columns=[f'{c}_idx' for c in ['VehicleID','start_time','end_time','split_stage']])
    # 只取查询需要的受控字段，明示不读取/保留测试完成时间或真实路径。
    safe_ctl=ctl[['trip_id','VehicleID','start_time','has_history_vehicle']].copy()
    safe_test=test[['trip_id','start_time']].merge(safe_ctl,on='trip_id',validate='one_to_one',suffixes=('_idx',''))
    if not (safe_test.start_time_idx==safe_test.start_time).all():
        raise ValueError('最后3天受控索引开始时间与模块一不一致')
    safe_test=safe_test.drop(columns=['start_time_idx'])
    if not (blind.set_index('trip_id').loc[safe_test.trip_id,'start_time'].to_numpy()==
            safe_test.start_time.to_numpy()).all():
        raise ValueError('最后3天纯OD与受控画像索引的开始时间不一致')
    # 严格复核各阶段边界：出发与结束都必须在其对应窗口内。
    bounds={
      'calibration':('2024-12-01','2024-12-15'),
      'model_train':('2024-12-15','2024-12-24'),
      'validation':('2024-12-24','2024-12-29'),
      'future_test':('2024-12-29','2025-01-01')}
    for s,(lo,hi) in bounds.items():
        z=idx[idx.split_stage==s]
        lo,hi=pd.Timestamp(lo),pd.Timestamp(hi)
        if not ((z.start_time>=lo)&(z.start_time<hi)&(z.end_time>lo)&(z.end_time<hi)).all():
            raise ValueError(f'时间阶段{s}存在越界行程，先修改模块一')
    return files,meta,hist,safe_test,blind


def feature_quality(hist):
    cal=hist[hist.split_stage=='calibration']
    out=[]
    for col in list(dict.fromkeys(ECON+COMF)):
        vals=pd.to_numeric(hist[col],errors='coerce').to_numpy(float)
        vc=pd.to_numeric(cal[col],errors='coerce').to_numpy(float)
        x=vc[np.isfinite(vc)]
        rate=float(np.isfinite(vals).mean())
        if rate<.90 or len(x)<100:
            raise ValueError(f'{col}有效率={rate:.2%}, 校准期有效样本={len(x)}；不能安全拟合UDPPM++')
        if col in ('path_len_km','travel_time_min') and (x<=0).any():
            raise ValueError(f'{col}存在非正值；先审核真实数据')
        out.append({'feature':col,'valid_rate':rate,
                    'calibration_valid_n':len(x),'calibration_median':float(np.median(x)),
                    'calibration_p01':float(np.quantile(x,.01)),
                    'calibration_p99':float(np.quantile(x,.99))})
    return pd.DataFrame(out)


def fit_models(hist,a):
    cal=hist.loc[hist.split_stage=='calibration'].sort_values(
       ['VehicleID','start_time','trip_id'],kind='mergesort').reset_index(drop=True)
    if len(cal)<100:
        raise ValueError('原型校准期真实路径不足100条')
    # 完整沿用第三章time_pairs: 同车辆、连续开始时间差不超过240min。
    # 第三章原代码的time_pairs默认按开始时间排序；校准数据在此先排序。
    v=cal.VehicleID.to_numpy(dtype=str)
    ts=cal.start_time.to_numpy(dtype='datetime64[ns]')
    delta=(ts[1:]-ts[:-1])/np.timedelta64(1,'m')
    aa=np.flatnonzero((v[1:]==v[:-1])&(delta>=0)&(delta<=a.smooth_gap_min))
    pairs=(aa,aa+1)
    models={};scalers={};summary={}
    for dim,cols,seed in [('econ',ECON,a.seed),('comf',COMF,a.seed+1)]:
        log(f'训练{dim} 4类原型：真实校准行程{len(cal):,}条、时间相邻对{len(aa):,}')
        pre=TrainOnlyMAD(cols).fit(cal)
        X=pre.transform(cal)
        mod=UDPPMSoft(k=4,tau=a.tau,bal=a.bal,sep=a.sep,sm=a.sm,
                      iters=a.iterations,seed=seed).fit(X,pairs)
        if mod.centers_.shape!=(4,6):raise ValueError(f'{dim}模型中心shape异常')
        check_probability(mod.predict_proba(X),dim)
        models[dim]=mod;scalers[dim]=pre
        summary[dim]={
            'features':cols,'params':pre.parts,
            'centers_scaled':mod.centers_.astype(float).tolist(),
            'alpha':float(mod.alpha_),
            'fit_history':[{k:(None if isinstance(v,float) and not np.isfinite(v) else v)
                             for k,v in rec.items()} for rec in mod.history_],
            'train_n':len(cal), 'pair_n':len(aa), 'seed':seed}
    return models,scalers,summary


def all_hist_probs(hist,models,scalers):
    r=hist[['trip_id','VehicleID','start_time','end_time','split_stage']].copy()
    for dim in ('econ','comf'):
        q=models[dim].predict_proba(scalers[dim].transform(hist))
        check_probability(q,dim)
        for k in range(4):r[f'{dim}_prob_{k}']=q[:,k]
        r[f'{dim}_cluster_id']=q.argmax(axis=1)
    return r


def prior_asof(hist,queries,cutoff=None):
    """按车辆结束时间搜索：只累计end_time < min(query.start, cutoff)的历史。
    验证期仅允许calibration/model_train；最终测试期仅允许前28天。
    保持train-only的行为概率、未读取任何测试阶段真实路径。
    """
    out=queries[['trip_id','VehicleID','start_time']].copy().reset_index(drop=True)
    out['history_n_prior']=np.zeros(len(out),dtype=np.int32)
    for c in PROB_COLS:out[c]=np.nan
    hist=hist.copy()
    hcols=Q_COLS
    hist=hist.sort_values(['VehicleID','end_time','trip_id'],kind='mergesort')
    lookup={key:grp for key,grp in hist.groupby('VehicleID',sort=False)}
    for vid, loc in out.groupby('VehicleID',sort=False).groups.items():
        loc=np.asarray(list(loc),dtype=np.int64)
        if vid not in lookup:
            out.loc[loc,'history_n_prior']=0
            out.loc[loc,PROB_COLS]=.25
            continue
        g=lookup[vid]
        end=g.end_time.to_numpy(dtype='datetime64[ns]')
        X=g[hcols].to_numpy(float)
        pre=np.vstack([np.zeros((1,8),dtype=float),np.cumsum(X,axis=0)])
        qt=out.loc[loc,'start_time'].to_numpy(dtype='datetime64[ns]')
        if cutoff is not None:
            qt=np.minimum(qt,np.datetime64(cutoff,'ns'))
        # strictly < cutoff，保守避免同时结束/开始的观测作为已知历史。
        n=np.searchsorted(end,qt,side='left')
        z=np.full((len(loc),8),.25,dtype=float)
        hit=n>0
        z[hit]=pre[n[hit]]/n[hit,None]
        out.loc[loc,'history_n_prior']=n.astype(np.int32)
        out.loc[loc,PROB_COLS]=z
    mat=out[PROB_COLS].to_numpy(float)
    check_probability(mat[:,:4],'历史经济画像')
    check_probability(mat[:,4:],'历史舒适画像')
    out['is_cold_start']=(out.history_n_prior==0).astype('int8')
    return out


def make_cluster_report(hist,prob):
    cal=hist[hist.split_stage=='calibration'].copy()
    cal=cal.merge(prob[['trip_id']+Q_COLS],on='trip_id',validate='one_to_one')
    rows=[]
    for dim,fields in [('econ',ECON),('comf',COMF)]:
        q=cal[[f'{dim}_prob_{i}' for i in range(4)]].to_numpy(float)
        hard=q.argmax(axis=1)
        for k in range(4):
            weights=q[:,k]
            row={'dimension':dim,'prototype_id':k,'calibration_n':len(cal),
                 'hard_assign_n':int((hard==k).sum()),
                 'soft_membership_sum':float(weights.sum()),
                 'mean_soft_membership':float(weights.mean())}
            for f in fields:
                v=pd.to_numeric(cal[f],errors='coerce').to_numpy(float)
                mask=np.isfinite(v)
                row[f'{f}_median_hard']=float(np.median(v[mask & (hard==k)])) if np.any(mask & (hard==k)) else np.nan
                row[f'{f}_mean_soft']=float(np.average(v[mask],weights=weights[mask])) if np.any(mask) and weights[mask].sum()>0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_cluster_counts(report,out):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,(dim,title) in zip(axes,[('econ','经济性'),('comf','舒适性')]):
        sub=report[report.dimension==dim].sort_values('prototype_id')
        ax.bar(sub.prototype_id.astype(int),sub.hard_assign_n)
        ax.set_xticks([0,1,2,3]);ax.set_xlabel('Prototype ID')
        ax.set_title(title+' · 校准期硬主类规模')
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('真实行程数量')
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(out/f'图4-X_校准期UDPPMpp双维原型规模.{ext}',
                    dpi=450 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='模块二：原第三章6+6输入与UDPPM++历史画像，28天训练/3天冻结盲测')
    parser.add_argument('--module1',type=Path,default=MODULE1)
    parser.add_argument('--out',type=Path,default=MODULE2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--tau',type=float,default=.8)
    parser.add_argument('--bal',type=float,default=.03)
    parser.add_argument('--sep',type=float,default=.02)
    parser.add_argument('--sm',type=float,default=.01)
    parser.add_argument('--iterations',type=int,default=28)
    parser.add_argument('--smooth-gap-min',type=float,default=240.)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.tau<=0 or args.iterations<1 or min(args.bal,args.sep,args.sm)<0:
        parser.error('tau必须>0、iterations必须>=1、正则系数不可为负')
    start=time.perf_counter()
    out=args.out
    outputs={
      'model':'01_训练期UDPPMpp原型及预处理参数.json',
      'prob':'02_前28天已完成路径8维软隶属概率.csv.gz',
      'asof':'03_训练验证逐OD严格历史画像.csv.gz',
      'test_control':'04_最后3天受控车辆画像_禁止进入候选生成.csv.gz',
      'test_input':'05_最后3天盲推理8维画像_只按trip_id.csv',
      'group':'06_前28天冻结群体画像.json',
      'cluster':'07_校准期原型道路特征解释.csv',
      'quality':'08_原始6加6特征质量统计.csv',
      'audit':'09_模块二完整性与时间隔离审计.json',
      'readme':'README_模块二使用边界.txt',
    }
    out.mkdir(parents=True,exist_ok=True)
    for fn in list(outputs.values())+['图4-X_校准期UDPPMpp双维原型规模.png',
                                     '图4-X_校准期UDPPMpp双维原型规模.svg']:
        if (out/fn).exists() and not args.overwrite:
            raise FileExistsError(f'检测到已有输出：{out/fn}\n需要重跑请加 --overwrite，避免覆盖已归档正式实验。')
    log('1/6 检查模块一输入和数据完整性')
    files,source_meta,hist,test,blind=validate_module1(args)
    if len(hist)!=int(source_meta['counts']['development_28d']):
        raise ValueError('历史数据数量与模块一审计不一致')
    if len(test)!=int(source_meta['counts']['future_3d']):
        raise ValueError('测试OD数量与模块一审计不一致')
    quality=feature_quality(hist)
    log(f'前28天历史路径={len(hist):,}；最后3天仅OD索引={len(test):,}')
    log('2/6 用第三章原有6+6特征及完全一致的UDPPMSoft类训练校准期原型')
    models,scalers,fit_summary=fit_models(hist,args)
    config={
      'source_algorithm':'第三章Step7原始TrainOnlyMAD及UDPPMSoft实现，原函数/类原文复制',
      'feature_input':'exact_6+6_behavior','semantic_mode':'none',
      'semantic_warning':'未使用整月PCA8语义辅助，避免未来时间泄漏；不可声称等同第三章fusion正式版本',
      'calibration_window':'2024-12-01..2024-12-14',
      'num_prototypes_economic':4,'num_prototypes_comfort':4,
      'tau':args.tau,'bal':args.bal,'sep':args.sep,'sm':args.sm,
      'iterations':args.iterations,'seed':args.seed,'smooth_gap_min':args.smooth_gap_min,
      'economy':fit_summary['econ'],'comfort':fit_summary['comf']}
    log('3/6 仅对前28天已完成真实路径计算软概率，不读取未来3天的真实行为')
    q=all_hist_probs(hist,models,scalers)
    assert not set(q.trip_id).intersection(set(test.trip_id))
    hq=q.copy()
    for i in range(4):
        hq[f'history_econ_{i}']=hq[f'econ_prob_{i}']
        hq[f'history_comf_{i}']=hq[f'comf_prob_{i}']
    log('4/6 严格时间画像：训练期逐OD历史、验证期冻结于24日、测试期冻结于29日')
    # 校准期数据只用于原型训练，不被作为第4章条件选择模型的监督训练OD。
    qs=[]
    for stage in ['model_train','validation']:
        target=hist[hist.split_stage==stage][['trip_id','VehicleID','start_time']].copy()
        if stage=='model_train':
            allowed=hq[hq.split_stage.isin(['calibration','model_train'])]
            portrait=prior_asof(allowed,target)
            portrait['history_policy']='completed_before_OD_start'
        else:
            allowed=hq[hq.split_stage.isin(['calibration','model_train'])]
            portrait=prior_asof(allowed,target,cutoff=TRAIN_END)
            portrait['history_policy']='frozen_2024-12-24_00:00_Bangkok'
        portrait['split_stage']=stage
        qs.append(portrait)
    dev_portrait=pd.concat(qs,ignore_index=True).sort_values(['start_time','trip_id'])
    # 测试只能访问前28天q，没有任何2024-12-29..31的真实行为特征。
    test_portrait=prior_asof(hq,test[['trip_id','VehicleID','start_time']],cutoff=TEST_BEGIN)
    test_portrait['history_policy']='frozen_2024-12-29_00:00_Bangkok'
    if len(test_portrait)!=len(test) or set(test_portrait.trip_id)!=set(blind.trip_id):
        raise ValueError('测试画像与最后3天纯OD匹配失败')
    cold=int(test_portrait.is_cold_start.sum())
    declared=int(source_meta['counts']['test_cold_start_count'])
    if cold!=declared:
        raise ValueError(f'测试冷启动{cold}与模块一{declared}不一致；检查画像历史时间与源数据')
    # 根据完整历史判断各车冻结画像；与测试集车辆画像保持同一个计算规则。
    freeze_vehicles=hq[['VehicleID']].drop_duplicates().copy()
    freeze_vehicles['trip_id']='summary_'+freeze_vehicles['VehicleID'].astype(str)
    freeze_vehicles['start_time']=TEST_BEGIN
    freeze_profile=prior_asof(hq,freeze_vehicles,cutoff=TEST_BEGIN)
    group_q=hq[Q_COLS].to_numpy(float)
    meanq=group_q.mean(axis=0)
    assert np.allclose(meanq[:4].sum(),1,atol=1e-5)
    assert np.allclose(meanq[4:].sum(),1,atol=1e-5)
    group_profile={'as_of':'2024-12-29 00:00:00 Asia/Bangkok',
       'source':'仅前28天真实历史已完成路径的UDPPMSoft预测概率均值',
       'n_history_paths':len(hq),
       'economic_probabilities':meanq[:4].tolist(),
       'comfort_probabilities':meanq[4:].tolist(),
       'cold_start_prior':[.25,.25,.25,.25],
       'warning':'群体平均画像是默认/中性场景参考，不是已校验的经济或舒适偏好标签'}
    log('5/6 再次检查时间与概率、输出真实实验参数、画像及特征报告')
    if not (hq.end_time < TEST_BEGIN).all():
        raise ValueError('用于测试画像的历史数据包含未来时间')
    if not (dev_portrait.loc[dev_portrait.split_stage=='validation','history_policy']
            == 'frozen_2024-12-24_00:00_Bangkok').all():
        raise AssertionError('验证集历史冻结失败')
    if int(test_portrait.history_n_prior.sum())<0:raise AssertionError('历史计数不合理')
    # 避免任何测试VehicleID或真实结束时间被带入通用路径候选生成。
    test_public=test_portrait[['trip_id','history_n_prior','is_cold_start']+PROB_COLS].copy()
    assert 'VehicleID' not in test_public.columns and 'end_time' not in test_public.columns
    # 受控输出仅给评价画像匹配使用，不交给候选生成器。
    report=make_cluster_report(hist,q)
    save_json(config,out/outputs['model'])
    save_csv(q,out/outputs['prob'])
    save_csv(dev_portrait,out/outputs['asof'])
    save_csv(test_portrait,out/outputs['test_control'])
    save_csv(test_public,out/outputs['test_input'])
    save_json(group_profile,out/outputs['group'])
    save_csv(report,out/outputs['cluster'])
    save_csv(quality,out/outputs['quality'])
    plot_cluster_counts(report,out)
    # 验证阶段画像固定：校准、模型训练两阶段；测试阶段使用所有前28天完成路径。
    audit={
       'module':'chapter4_module2_behavior_UDPPMpp_no_future_v1.0',
       'created_local':datetime.now().isoformat(timespec='seconds'),
       'source_manifest_version':source_meta['version'],
       'source_file_hashes':{k:file_hash(v) for k,v in files.items() if k!='stat'},
       'inputs':{k:str(v) for k,v in files.items()},
       'outputs':{k:str(out/v) for k,v in outputs.items()},
       'time_policy':{
         'calibration_fit':'2024-12-01..2024-12-14',
         'choice_train':'2024-12-15..2024-12-23',
         'validation_snapshot':'2024-12-24T00:00:00+07:00',
         'test_snapshot':'2024-12-29T00:00:00+07:00',
         'history_rule':'end_time < min(OD_start, snapshot); no future actual routes',
         'timezone_confirmation_pending':True,
         'module1_timezone':source_meta.get('time_policy',{}).get('naive_source_timezone')},
       'feature_policy':{'ECON_6':ECON,'COMF_6':COMF,'semantic_mode':'none'},
       'counts':{'calibration':int((hist.split_stage=='calibration').sum()),
          'model_train':int((hist.split_stage=='model_train').sum()),
          'validation':int((hist.split_stage=='validation').sum()),
          'test_od':len(test),'q_dev_rows':len(q),
          'dev_prior_rows':len(dev_portrait), 'test_prior_rows':len(test_portrait),
          'test_cold_start':cold,'test_with_history':len(test)-cold,
          'frozen_vehicle_n':len(freeze_profile)},
       'checks':{
          'same_exact_6+6':True,'no_test_features_read':True,
          'test_n_matches_module1':True,'test_cold_start_matches_module1':True,
          'test_public_has_no_vehicle_id':True,'no_true_route_used':True,
          'fit_only_calibration':True,'validation_snapshot_frozen':True,
          'test_snapshot_frozen':True,'profile_probabilities_normalized':True},
       'limitations':[
         '第三章正式结果可能带BERT-KG语义辅助。本实验仅复用原有6+6行为主特征和UDPPMSoft原算法，不复用整月语义PCA。',
         '校准期在拟合原型时已整体可见；不应将校准期路径画像用于同一批次监督选择模型性能评价。',
         '本模块画像对象以VehicleID为单位，不足以确定真实驾驶员身份。',
         '最后三天画像完全冻结，不使用未来3天完成的新行程更新。',
         '本模块不计算新OD推荐准确率，也不能将软隶属概率视为路径被选概率。',
         '模块一源时间是否UTC/曼谷当地时区仍应核对上游GPS时间定义。'
       ],
       'elapsed_seconds':round(time.perf_counter()-start,2)}
    save_json(audit,out/outputs['audit'])
    (out/outputs['readme']).write_text(
       '【模块二数据使用边界】\n'
       '01为仅校准期拟合的UDPPM++原型、预处理参数，6+6行为特征不变。\n'
       '02为前28天完成路径的8维概率，只可用于历史画像，不得作为未知候选的已观测属性。\n'
       '03用于第四章模型训练与内部验证，校准行程不进入监督选择模型。\n'
       '04是受控输出，含VehicleID，只给审计，不给纯OD候选生成器。\n'
       '05是最后3天按trip_id对接的无VehicleID盲推理画像；必须在生成候选后与OD按trip_id关联。\n'
       '06群体冻结画像只用于中性/群体画像实验，不代表特定主观偏好。\n'
       '07为校准期原型物理特征解释，cluster ID没有预定义好坏。\n'
       '最后三天严格不读取其真实HMM道路路径或真实6+6特征，真实结果只允许推荐完成后评价。\n'
       '此脚本使用第三章UDPPMSoft原算法+6+6行为输入（不使用整月语义PCA）。\n'
       '时区口径需由原始数据说明最终确认，不能将审计通过视为源时区已证实。\n',
       encoding='utf-8-sig')
    log('6/6 模块二完成，统计全部来自真实模块一分割后的路径特征。')
    log(f'原型训练={audit["counts"]["calibration"]:,}; 开发期概率={len(q):,}; '
        f'监督训练与验证画像={len(dev_portrait):,}; 测试盲画像={len(test_public):,}; 冷启动={cold:,}')
    log('后续候选生成只允许读模块一05，后续偏好评分从模块二05按trip_id匹配。')
    log('输出目录：'+str(out))


if __name__=='__main__':
    try:main()
    except Exception as ex:
        log('【模块二失败】'+str(ex))
        raise
