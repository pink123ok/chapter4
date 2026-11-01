# -*- coding: utf-8 -*-
"""
第四章模块四｜真实条件路径选择模型 + 2024年12月29~31日新OD自然候选盲排序

第三章经济性6项与舒适性6项特征完全不变；消费模块二已经冻结的8维历史画像。
修复版v1.2：修复EdgeInfo实例化、旧SQLite错误缓存隔离，增加自检和失败审计。
第4部分从模块一开发期OD的真实路网端点生成自然候选，监督训练时若真实HMM
道路边序列未在自然候选中，单独补入作标签；重新计算含真实路线条件集的PS。
所有额外加入真实路线的样本显式输出real_added_TRACE_NOT_FEATURE审计，
严禁将其作为模型特征，不能将条件候选Hit@1解释为盲测推荐成功率。
对最后三天**只读取**模块三自然候选特征及模块二盲画像，绝不打开最后三天真值。

依赖：pip install numpy pandas scipy networkx shapely
用法：
  python 第四章_模块四_六模型训练与后三天自然候选盲排序.py --phase check
  python 第四章_模块四_六模型训练与后三天自然候选盲排序.py --phase prepare
  python 第四章_模块四_六模型训练与后三天自然候选盲排序.py --phase fit
  python 第四章_模块四_六模型训练与后三天自然候选盲排序.py --phase predict
  python 第四章_模块四_六模型训练与后三天自然候选盲排序.py --phase all

prepare使用SQLite按OD断点续跑，参数/源文件或路网变化会拒绝混用旧库。
如只想测试运行逻辑，可用 --pilot --pilot-od 100；试跑数据不能写进论文。
训练耗时主要来自重新为20,720个OD搜索候选，停止后重复原命令即可续跑。
若尚未确认模块一的曼谷时区口径，论文正式实验前仍需核实。
小样本：python 4step-4.py --phase prepare --pilot --pilot-od 20 --reset-db
旧进度诊断：python 4step-4.py --phase diagnose --pilot --pilot-od 20
正式实验（小样本通过后）：python 4step-4.py --phase prepare --reset-db
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,heapq,json,math,re,sqlite3,sys,time,random
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import networkx as nx
from shapely import wkt
from scipy.optimize import minimize

BASE=Path(r'C:\study\轨迹数据预处理')
FULL=BASE/'曼谷GPS_正式清洗_202412'/'FULL_全量'
ROOT=FULL/'第四章_28天训练_3天独立测试'
M1=ROOT/'模块一_时间划分与隔离'
M2=ROOT/'模块二_UDPPMpp双维历史画像'
M3=ROOT/'模块三_未知OD自然候选与静态道路属性'/'full_K10'
OUT=ROOT/'模块四_条件选择六模型与未知OD盲排序'
GRAPH=BASE/'luwangdata'/'曼谷_路网_兼容旧版.graphml'
DEV_FIELDS=['trip_id','start_time','origin_node','destination_node','od_straight_km',
            'origin_lon','origin_lat','destination_lon','destination_lat']
ECON_ROUTE=['total_len_km','ratio_to_shortest','ratio_motorway','ratio_major','ratio_local']
COMF_ROUTE=['turns_per_km','segment_density_seg_per_km','ratio_local','ratio_motorway','oneway_ratio']
BASE10=['total_len_km','ratio_to_shortest','ratio_motorway','ratio_major','ratio_local',
        'turns_per_km','segment_density_seg_per_km','oneway_ratio',
        'lanes_mean_w','maxspeed_mean_known_kmh']
SCALE_COLS=BASE10+['log_ps_count','log_ps_length']
PROFILE=[f'history_{d}_{j}' for d in ('econ','comf') for j in range(4)]
MODELS={
 'MNL_L2': ['base'],
 'PSL_Count_L2':['base','ps_count'],
 'PSL_LengthWeighted_L2':['base','ps_length'],
 'UDPPM_Econ_PSL_L2':['base','ps_length','econ'],
 'UDPPM_Comf_PSL_L2':['base','ps_length','comf'],
 'UDPPM_Dual_PSL_L2':['base','ps_length','econ','comf'],
}
MODEL_ATTRS=['total_len_km','n_segments','segment_density_seg_per_km','ratio_motorway',
 'ratio_major','ratio_local','ratio_other','oneway_ratio','oneway_attribute_coverage',
 'lanes_mean_w','lanes_attribute_coverage','maxspeed_mean_known_kmh',
 'maxspeed_attribute_coverage','turn_count_45deg','u_turn_count_150deg',
 'turns_per_km','wkt_geometry_length_fraction','node_line_fallback_length_fraction',
 'ps_count','log_ps_count','ps_length','log_ps_length','shared_edge_length_ratio',
 'unique_edge_length_ratio','ratio_to_shortest','detour_ratio_vs_straight']
ROAD_PRIORITY=['motorway','motorway_link','trunk','trunk_link','primary','primary_link',
 'secondary','secondary_link','tertiary','tertiary_link','residential','living_street',
 'service','unclassified','road','track','busway','footway','pedestrian','path','cycleway']
NONMOTOR={'footway','pedestrian','path','cycleway','steps','bridleway',
          'platform','corridor','construction','proposed','busway'}
try:csv.field_size_limit(sys.maxsize)
except OverflowError:csv.field_size_limit(2**31-1)

def info(s):print(f'[{datetime.now():%H:%M:%S}] {s}',flush=True)
def jsonsafe(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=str)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readjson(p):
 require(p,'JSON');return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def writejson(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.writing')
 q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8-sig')
 q.replace(p)
def require(p,label):
 if not Path(p).is_file() or Path(p).stat().st_size==0:
  raise FileNotFoundError(f'缺少{label}: {p}；禁止使用模拟或猜测数据')
def savecsv(df,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.writing')
 df.to_csv(q,index=False,encoding='utf-8-sig',float_format='%.10g',compression='gzip' if p.suffix=='.gz' else None)
 q.replace(p)
def csvopen(p):return gzip.open(p,'rt',encoding='utf-8-sig',newline='') if str(p).endswith('.gz') else open(p,'r',encoding='utf-8-sig',newline='')

def check_inputs(a):
 mf=readjson(a.m1/'11_完整性防泄漏审计.json')
 pm=readjson(a.m2/'09_模块二完整性与时间隔离审计.json')
 tm=readjson(a.m3/'07_模块三完整性_路网及数据隔离审计.json')
 src={
 'dev':a.m1/'04_前28天开发OD与真实路径_仅训练.csv.gz',
 'dev_profiles':a.m2/'03_训练验证逐OD严格历史画像.csv.gz',
 'test_blind':a.m1/'05_最后3天_纯OD推理输入_绝无真实路线.csv',
 'test_features':a.m3/'02_未知OD候选静态属性与PathSize_模型输入.csv.gz',
 'test_profiles':a.m2/'05_最后3天盲推理8维画像_只按trip_id.csv',
 'test_audit':a.m3/'03_全部OD搜索与Top3审计.csv',
 'graphml':a.graphml,
 }
 for k,p in src.items():require(p,k)
 if mf.get('version')!='chapter4_module1_chrono_split_v1.0':raise ValueError('模块一版本错误')
 if pm.get('module')!='chapter4_module2_behavior_UDPPMpp_no_future_v1.0':raise ValueError('模块二版本错误')
 if not tm.get('is_formal_full') or tm.get('selected_od')!=4469:
  raise ValueError('模块三不是全部4,469条正式测试OD；拒绝把pilot当正式结果')
 config=tm.get('source_config',{})
 expected= config.get('input_source',{}).get('graphml',{}).get('sha256')
 if a.verify_graph_sha:
  actual=sha(a.graphml)
  if expected!=actual:raise ValueError('训练路网和后三天模块三路网SHA256不一致')
 elif a.phase=='check':info('提醒：本次 --no-graph-hash 略过了路网哈希；正式训练建议开启')
 if config.get('k')!=a.k or config.get('max_attempts')!=a.max_attempts or \
    config.get('max_route_ratio')!=a.max_route_ratio or \
    config.get('penalty_start')!=a.penalty_start or config.get('penalty_step')!=a.penalty_step or \
    config.get('max_expanded')!=a.max_expanded or config.get('vehicle_filter')!=a.vehicle_filter or \
    config.get('turn_angle')!=a.turn_angle or config.get('min_straight_km')!=0.0 or \
    config.get('max_node_coordinate_error_m')!=a.max_node_coordinate_error_m:
  raise ValueError('训练候选参数与模块三后三天不一致；请保持同一搜路及属性口径')
 if int(mf['counts']['stage_od']['model_train'])!=13383 or int(mf['counts']['stage_od']['validation'])!=7337:
  raise ValueError('模块一训练/验证OD数与已核对的真实来源不符；请重新审核版本')
 if int(pm['counts']['dev_prior_rows'])!=20720:raise ValueError('模块二开发期画像行数不符')
 if int(pm['counts']['test_prior_rows'])!=int(mf['counts']['future_3d']):raise ValueError('测试画像数量不符')
 with csvopen(src['test_features']) as f:
  header=next(csv.reader(f))
  if not set(BASE10+['log_ps_count','log_ps_length','trip_id','candidate_id_TRACE_NOT_FEATURE'])<=set(header):
   raise ValueError('模块三静态候选特征字段不足')
  if {'chosen','VehicleID','edge_seq_uvk_json'} & set(header):
   raise ValueError('模块三盲测候选特征混入真实结果字段')
 return src,mf,pm,tm


# 以下有向A*/Dijkstra与路段属性原样复用已运行成功的模块三核心实现
def finite_num(v):
    try:
        z=float(v)
        return z if math.isfinite(z) else math.nan
    except (ValueError,TypeError):
        return math.nan


def haversine_m(a,b):
    lon1,lat1=a;lon2,lat2=b
    la1,la2=math.radians(lat1),math.radians(lat2)
    z=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin(math.radians(lon2-lon1)/2)**2
    return 2*6371008.8*math.asin(min(1.,math.sqrt(max(0.,z))))


def highway_code(raw):
    """沿用Step43的高速公路/干线/地方道路分类解析。"""
    tokens=re.findall(r'[a-z][a-z0-9_]*',str(raw).lower())
    for k in ROAD_PRIORITY:
        if k in tokens:return k
    return tokens[0] if tokens else 'unknown'


def road_group(h):
    if h in {'motorway','motorway_link','trunk','trunk_link'}:return 'motorway'
    if h in {'primary','primary_link','secondary','secondary_link','tertiary','tertiary_link'}:return 'major'
    if h in {'residential','living_street','service','unclassified'}:return 'local'
    return 'other'


def parse_bool(s):
    s=str(s).strip().lower()
    if s in {'true','yes','1'}:return 1
    if s in {'false','no','0'}:return 0
    return None


def lanes_parse(s):
    vals=[int(x) for x in re.findall(r'(?<!\d)\d{1,2}(?!\d)',str(s))]
    vals=[v for v in vals if 1<=v<=16]
    return min(vals) if vals else None


def speed_parse(s):
    if s is None:return None
    t=str(s).lower().strip()
    if any(x in t for x in ['none','signals','walk','national','urban','variable','unknown','nan']):return None
    vals=[float(x) for x in re.findall(r'(?<!\w)\d+(?:\.\d+)?',t)]
    if not vals:return None
    v=min(vals)
    if 'mph' in t:v*=1.609344
    return v if 5<=v<=150 else None


def heading(a,b):
    if haversine_m(a,b)<0.2:return None
    x1,y1=map(math.radians,a);x2,y2=map(math.radians,b)
    d=x2-x1
    y=math.sin(d)*math.cos(y2)
    x=math.cos(y1)*math.sin(y2)-math.sin(y1)*math.cos(y2)*math.cos(d)
    return (math.degrees(math.atan2(y,x))+360)%360


def tangents(coords):
    if len(coords)<2:return None,None
    j=1
    while j<len(coords)-1 and haversine_m(coords[0],coords[j])<5:j+=1
    k=len(coords)-2
    while k>0 and haversine_m(coords[k],coords[-1])<5:k-=1
    return heading(coords[0],coords[j]),heading(coords[k],coords[-1])


def explicitly_nonmotor(d):
    """只禁止明确的非汽车道路/明确的禁行标记；不保证所有法规和转向限制。"""
    h=highway_code(d.get('highway','unknown'))
    if h in NONMOTOR:return True
    for attr in ('motor_vehicle','vehicle','access'):
        val=str(d.get(attr,'')).strip().lower()
        # 对未知属性不做主观补全；对明确的禁行标记谨慎过滤。
        if val in {'no','private'}:return True
    return False


@dataclass(slots=True)
class EdgeInfo:
    length:float
    group:str
    one_way:int|None
    lanes:float|None
    maxspeed:float|None
    start_heading:float|None
    end_heading:float|None
    precise_geom:bool


class RoadNetwork:
    """GraphML有向(u,v,key)搜索；路由继承Step42，属性提取继承Step43。"""
    def __init__(self,graphml,vehicle_filter):
        info(f'读取正式有向GraphML：{graphml}')
        g=nx.read_graphml(str(graphml),force_multigraph=True)
        if not g.is_directed():raise RuntimeError('GraphML不是有向图，不允许改成无向图')
        self.g=g
        self.xy={}
        for n,d in g.nodes(data=True):
            x,y=finite_num(d.get('x')),finite_num(d.get('y'))
            if math.isfinite(x) and math.isfinite(y) and -180<=x<=180 and -90<=y<=90:
                self.xy[str(n)]=(x,y)
        self.adj=defaultdict(list)
        self.edges={}
        self.feature_cache={}
        filt=Counter()
        factor=1.0
        missing_coordinate=False
        for u,v,key,d in g.edges(keys=True,data=True):
            u,v,key=str(u),str(v),str(key)
            L=finite_num(d.get('length'))
            if not math.isfinite(L) or L<=0:
                filt['nonpositive_or_missing_length']+=1
                continue
            if vehicle_filter=='exclude_nonmotor' and explicitly_nonmotor(d):
                filt['nonmotor_or_access_excluded']+=1
                continue
            e=(u,v,key)
            self.adj[u].append((v,key,L))
            self.edges[e]=L
            if u in self.xy and v in self.xy:
                f=haversine_m(self.xy[u],self.xy[v])
                if f>1:factor=min(factor,L/f)
            else:missing_coordinate=True
        if not self.edges:raise RuntimeError('GraphML筛选后没有可用有向边')
        self.hfactor=0. if missing_coordinate else max(0.,min(1.,factor))*.999
        self.stats={
            'graph_nodes':g.number_of_nodes(), 'graph_directed_edges':g.number_of_edges(),
            'usable_directed_edges':len(self.edges),'nodes_with_xy':len(self.xy),
            'vehicle_filter':vehicle_filter,'excluded_edges':dict(filt),
            'admissible_geometric_heuristic_factor':self.hfactor,
            'directed':True,'graph_length_unit':'meters'}
        info('路网装载完毕：可搜有向边'+f'{len(self.edges):,}，'+
             f'坐标节点{len(self.xy):,}，A*启发系数{self.hfactor:.6f}')

    def astar(self,origin,dest,penalties,max_expanded):
        if origin==dest:return [],0.,0,'same_node'
        if origin not in self.g:return [],math.nan,0,'origin_not_in_graph'
        if dest not in self.g:return [],math.nan,0,'destination_not_in_graph'
        if origin not in self.adj:return [],math.nan,0,'origin_no_usable_outgoing_edge'
        if self.hfactor and dest in self.xy:
            end=self.xy[dest]
            def heuristic(n):
                return self.hfactor*haversine_m(self.xy[n],end) if n in self.xy else 0.
        else:
            heuristic=lambda n:0.
        openq=[(heuristic(origin),0.,0,origin)]
        dist={origin:0.}
        parents={}
        serial=0;expanded=0
        while openq:
            _,cost,_,u=heapq.heappop(openq)
            if cost>dist.get(u,math.inf)+1e-7:continue
            if u==dest:
                seq=[]
                while u!=origin:
                    prev,k=parents[u]
                    seq.append((prev,u,k));u=prev
                seq.reverse()
                return seq,cost,expanded,'found'
            expanded+=1
            if max_expanded>0 and expanded>max_expanded:
                return [],math.nan,expanded,'budget_exceeded'
            for v,key,L in self.adj.get(u,[]):
                nd=cost+L*(1+penalties.get((u,v,key),0.))
                if nd+1e-8<dist.get(v,math.inf):
                    dist[v]=nd;parents[v]=(u,key)
                    serial+=1
                    heapq.heappush(openq,(nd+heuristic(v),nd,serial,v))
        return [],math.nan,expanded,'no_directed_connection'

    def edge_info(self,e):
        if e in self.feature_cache:return self.feature_cache[e]
        if e not in self.edges:raise RuntimeError(f'候选边不在可用路网中：{e}')
        u,v,k=e
        edge_options=self.g[u][v]
        if k in edge_options:
            d=edge_options[k]
        else:
            matching=[value for key,value in edge_options.items() if str(key)==k]
            if len(matching)!=1:
                raise RuntimeError(f'GraphML边key无法唯一定位：{e}')
            d=matching[0]
        if u not in self.xy or v not in self.xy:
            raise RuntimeError(f'候选边端点无有效坐标，无法计算几何转弯：{e}')
        endpoints=[self.xy[u],self.xy[v]]
        coords=endpoints
        precise=False
        raw=d.get('geometry')
        if raw:
            try:
                geom=wkt.loads(str(raw))
                if geom.geom_type=='LineString' and len(geom.coords)>=2:
                    pts=[(float(c[0]),float(c[1])) for c in geom.coords]
                    fw=haversine_m(pts[0],endpoints[0])+haversine_m(pts[-1],endpoints[1])
                    bw=haversine_m(pts[-1],endpoints[0])+haversine_m(pts[0],endpoints[1])
                    if bw<fw:pts.reverse()
                    coords=pts;precise=True
            except Exception:
                precise=False;coords=endpoints
        s,t=tangents(coords)
        ei=EdgeInfo(self.edges[e],road_group(highway_code(d.get('highway','unknown'))),
                    parse_bool(d.get('oneway')),lanes_parse(d.get('lanes')),
                    speed_parse(d.get('maxspeed')),s,t,precise)
        self.feature_cache[e]=ei
        return ei


def route_hash(seq):
    data=jsonsafe(seq)
    return hashlib.sha1(data.encode('utf-8')).hexdigest()[:20]


def build_natural_routes(road,origin,dest,k,max_trials,penalty_start,penalty_step,
                         max_ratio,max_expanded):
    """Step42的自然搜索逻辑，但完全没有obs实走路线参数和任何真实重叠字段。"""
    routes=[];seen=set();used=Counter();shortest=None
    attempts=0;expanded=0;states=Counter();budget=False
    for i in range(max_trials):
        if len(routes)>=k:break
        strength=0 if i==0 else penalty_start+penalty_step*(i-1)
        penalty={e:strength*c for e,c in used.items()} if i else {}
        seq,_,nodes,state=road.astar(origin,dest,penalty,max_expanded)
        attempts+=1;expanded+=nodes;states[state]+=1
        if state=='budget_exceeded':
            budget=True
            if i==0:break
            continue
        if state!='found':
            if i==0:break
            continue
        length_m=sum(road.edges[e] for e in seq)
        if shortest is None:shortest=length_m
        if length_m>shortest*max_ratio+1e-5:continue
        sig=route_hash(seq)
        if sig in seen:continue
        seen.add(sig)
        routes.append({'rank':len(routes)+1,'path':seq,'hash':sig,'length_m':length_m})
        used.update(seq)
    status=('ok_k_reached' if len(routes)==k else
            'partial_budget' if routes and budget else
            'fewer_than_k' if routes else
            'first_search_budget' if budget else
            next(iter(states),'no_natural_route'))
    return routes,{'status':status,'shortest_m':shortest,'attempts':attempts,
                   'expanded_total':expanded,'states':dict(states),'budget_hit':budget}


def route_properties(seq,roads,angle_threshold):
    """按Step43(2).py相同的几何转角及长度加权统计口径计算。"""
    L=0.; group=Counter()
    dir_known=dir_true=lane_len=lane_sum=speed_len=speed_sum=wkt_len=0.
    turns=u_turns=0;prev=None;prevkey=None
    for e in seq:
        edge=roads.edge_info(e)
        l=edge.length;L+=l;group[edge.group]+=l
        if edge.one_way is not None:
            dir_known+=l;dir_true+=l*edge.one_way
        if edge.lanes is not None:
            lane_len+=l;lane_sum+=l*edge.lanes
        if edge.maxspeed is not None:
            speed_len+=l;speed_sum+=l*edge.maxspeed
        if edge.precise_geom:wkt_len+=l
        if prevkey is not None:
            if prevkey[1]!=e[0]:raise ValueError('候选有向道路边序列不连续')
            if prev is not None and edge.start_heading is not None:
                delta=abs((edge.start_heading-prev+180)%360-180)
                if delta>=angle_threshold:turns+=1
                if delta>=150:u_turns+=1
        prev=edge.end_heading;prevkey=e
    if L<=0:raise ValueError('候选路径长度小于等于零')
    lk=L/1000.
    return {
        'total_len_km':lk,'n_segments':len(seq),'segment_density_seg_per_km':len(seq)/lk,
        'ratio_motorway':group['motorway']/L,'ratio_major':group['major']/L,
        'ratio_local':group['local']/L,'ratio_other':group['other']/L,
        'oneway_ratio':dir_true/dir_known if dir_known else None,
        'oneway_attribute_coverage':dir_known/L,
        'lanes_mean_w':lane_sum/lane_len if lane_len else None,
        'lanes_attribute_coverage':lane_len/L,
        'maxspeed_mean_known_kmh':speed_sum/speed_len if speed_len else None,
        'maxspeed_attribute_coverage':speed_len/L,
        'turn_count_45deg':turns,'u_turn_count_150deg':u_turns,
        'turns_per_km':turns/lk,
        'wkt_geometry_length_fraction':wkt_len/L,
        'node_line_fallback_length_fraction':1-wkt_len/L
    }


def path_sizes(seqs,roads):
    """沿用Step43定义：相同(u,v,key)有向边在多少条不同候选中出现。"""
    counts=Counter()
    for seq in seqs:counts.update(set(seq))
    out=[]
    for seq in seqs:
        length=sum(roads.edges[e] for e in seq)
        cnt=len(seq)
        c=sum(1/counts[e] for e in seq)/cnt
        p=sum(roads.edges[e]/counts[e] for e in seq)/length
        shared=sum(roads.edges[e] for e in seq if counts[e]>1)/length
        if not (0<c<=1+1e-9 and 0<p<=1+1e-9):
            raise ValueError('计算出非法Path Size值')
        out.append({'ps_count':min(1.,c),'log_ps_count':math.log(c),
                    'ps_length':min(1.,p),'log_ps_length':math.log(p),
                    'shared_edge_length_ratio':shared,'unique_edge_length_ratio':1-shared})
    return out


def one_od(road,row,args):
    t0=time.perf_counter()
    t=row['trip_id'];origin=row['origin_node'];dest=row['destination_node']
    base={'trip_id':t,'start_time':row['start_time'],'origin_node':origin,
          'destination_node':dest,'od_straight_km':row['od_straight_km'],
          'natural_count':0,'top3_available':0,'shortest_length_km':None,
          'candidate_min_km':None,'candidate_max_km':None,
          'attempts':0,'expanded_total':0,'search_budget_hit':0,
          'search_states_json':'{}','reason':''}
    try:
        if origin==dest:
            base.update(status='same_node',reason='起终点相同，不能产生正长路线')
            return base,[],[]
        if float(row['od_straight_km']) < args.min_straight_km:
            base.update(status='od_below_min_straight',reason='预设OD距离筛选；不依赖真实行程')
            return base,[],[]
        if origin not in road.g or dest not in road.g:
            base.update(status='od_node_missing_from_graph',reason='起点或终点不在指定GraphML节点集')
            return base,[],[]
        for which,node in [('origin',origin),('destination',dest)]:
            if node not in road.xy:
                base.update(status='od_node_missing_coordinates',reason=which)
                return base,[],[]
            a=road.xy[node];b=(float(row[which+'_lon']),float(row[which+'_lat']))
            delta=haversine_m(a,b)
            if delta>args.max_node_coordinate_error_m:
                base.update(status='od_coordinate_mismatch',reason=f'{which}坐标与图节点相差{delta:.3f}米')
                return base,[],[]
        routes,diag=build_natural_routes(road,origin,dest,args.k,args.max_attempts,
                                        args.penalty_start,args.penalty_step,
                                        args.max_route_ratio,args.max_expanded)
        base.update(status=diag['status'],natural_count=len(routes),
                    top3_available=int(len(routes)>=3),
                    shortest_length_km=(diag['shortest_m']/1000 if diag['shortest_m'] else None),
                    attempts=diag['attempts'],expanded_total=diag['expanded_total'],
                    search_budget_hit=int(diag['budget_hit']),
                    search_states_json=jsonsafe(diag['states']))
        if not routes:
            base['reason']='无法找到自然候选或搜索预算耗尽'
            return base,[],[]
        base['candidate_min_km']=min(x['length_m'] for x in routes)/1000
        base['candidate_max_km']=max(x['length_m'] for x in routes)/1000
        paths=[x['path'] for x in routes]
        vals=path_sizes(paths,road)
        raw=[];safe=[]
        for rr,ps in zip(routes,vals):
            seq=rr['path']
            if seq[0][0]!=origin or seq[-1][1]!=dest:
                raise ValueError('生成的有向路线端点与OD不一致')
            for e1,e2 in zip(seq,seq[1:]):
                if e1[1]!=e2[0]:raise ValueError('生成路线有向边不连续')
            prop=route_properties(seq,road,args.turn_angle)
            prop.update(ps)
            prop['ratio_to_shortest']=rr['length_m']/diag['shortest_m']
            prop['detour_ratio_vs_straight']=prop['total_len_km']/float(row['od_straight_km'])
            if abs(prop['total_len_km']*1000-rr['length_m'])>1e-5:
                raise RuntimeError('图边长度统计不一致')
            if abs(sum(prop[k] for k in ('ratio_motorway','ratio_major','ratio_local','ratio_other'))-1)>1e-6:
                raise RuntimeError('道路等级长度占比不守恒')
            candidate_id=f'{t}_natural_{rr["rank"]:02d}_{rr["hash"]}'
            raw.append({'trip_id':t,'start_time':row['start_time'],
                        'origin_node':origin,'destination_node':dest,
                        'candidate_id_TRACE_NOT_FEATURE':candidate_id,
                        'candidate_rank_TRACE_NOT_FEATURE':rr['rank'],
                        'edge_seq_uvk_json':jsonsafe(seq),
                        'road_edge_count':len(seq),'route_length_km':rr['length_m']/1000,
                        'ratio_to_shortest':prop['ratio_to_shortest'],
                        'route_hash_TRACE_NOT_FEATURE':rr['hash']})
            safe.append({'trip_id':t,'start_time':row['start_time'],
                         'candidate_id_TRACE_NOT_FEATURE':candidate_id,**prop})
        if len({x['route_hash_TRACE_NOT_FEATURE'] for x in raw})!=len(raw):
            raise RuntimeError('同OD出现重复有向路径')
        return base,raw,safe
    except Exception as exc:
        # 编程类错误必须立即报出，不得当作真实OD不可用而吞掉；
        # 合法的OD级数据/搜索异常仍保留原因，便于真实数据质量审计。
        if isinstance(exc, (TypeError, AttributeError, NameError, ImportError)):
            raise RuntimeError(
                f'程序级异常，已停止以避免无效OD批量缓存；trip_id={t}, '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        # OD级失败要完整审计；路径或特征阶段失败不能留下半套候选。
        base.update(status='od_error',reason=f'{type(exc).__name__}: {exc}',
                    natural_count=0,top3_available=0,
                    candidate_min_km=None,candidate_max_km=None)
        return base,[],[]
    finally:
        base['elapsed_seconds']=round(time.perf_counter()-t0,6)


# ======== 历史候选构建：仅历史阶段可接触HMM观察路径 =========

def read_development(a,man):
    file=a.m1/'04_前28天开发OD与真实路径_仅训练.csv.gz'
    require(file,'模块一前28天真实历史OD')
    req=DEV_FIELDS+['VehicleID','split_stage','edge_seq_uvk_json']
    with csvopen(file) as f:
        rd=csv.DictReader(f)
        missing=set(req)-set(rd.fieldnames or [])
        if missing:raise ValueError('开发期真实OD缺少字段：'+str(sorted(missing)))
        result=[];counts=Counter();idset=set()
        for r in rd:
            stage=r['split_stage']
            if stage not in ('model_train','validation'):continue
            tid=r['trip_id']
            if tid in idset:raise ValueError(f'开发期trip_id重复：{tid}')
            idset.add(tid);counts[stage]+=1
            if not r['edge_seq_uvk_json'] or not r['VehicleID']:
                raise ValueError(f'{tid}缺少真实行驶序列或VehicleID')
            result.append(r)
    for stage in ['model_train','validation']:
        if counts[stage]!=int(man['counts']['stage_od'][stage]):
            raise ValueError(f'{stage}数量={counts[stage]},模块一审计={man["counts"]["stage_od"][stage]}')
    if a.pilot:
        rng=np.random.default_rng(a.seed)
        selected=[]
        for stage in ['model_train','validation']:
            pool=[r for r in result if r['split_stage']==stage]
            picked=set(rng.choice(len(pool),min(a.pilot_od,len(pool)),replace=False).tolist())
            selected.extend(r for j,r in enumerate(pool) if j in picked)
        result=selected
        info(f'仅真实OD功能试跑：训练{min(a.pilot_od,counts["model_train"])}、验证{min(a.pilot_od,counts["validation"])}')
    return result,counts


def observed_edges(text):
    raw=json.loads(text)
    if not isinstance(raw,list) or not raw:
        raise ValueError('HMM真实道路序列为空或不是数组')
    edges=[]
    for z in raw:
        if not isinstance(z,(list,tuple)) or len(z)!=3 or any(x is None for x in z):
            raise ValueError('HMM道路序列存在非(u,v,key)三元组')
        edges.append(tuple(map(str,z)))
    return edges


def historical_one_od(road,r,a):
    """训练/验证历史OD单独使用真实HMM路线作监督标签；不适用于后三天。"""
    assert r['split_stage'] in ('model_train','validation')
    t=r['trip_id'];stage=r['split_stage'];vid=r['VehicleID']
    od={f:r[f] for f in DEV_FIELDS}
    natural_audit,raw,natural_safe=one_od(road,od,a)
    aud={'trip_id':t,'split_stage':stage,'VehicleID_TRACE_NOT_FEATURE':vid,
         'status':'','reason':'','natural_count':len(raw),'natural_status':natural_audit['status'],
         'real_added_TRACE_NOT_FEATURE':0,'real_in_natural_TRACE_NOT_FEATURE':0,
         'condition_count':0,'observed_vs_shortest':None,
         'natural_search_seconds':natural_audit['elapsed_seconds']}
    if len(raw)<1:
        aud.update(status='no_natural',reason=natural_audit['status']+':'+str(natural_audit['reason']))
        return aud,[]
    try:
        obs=observed_edges(r['edge_seq_uvk_json'])
        if obs[0][0]!=str(r['origin_node']) or obs[-1][1]!=str(r['destination_node']):
            raise ValueError('HMM首末端点与当前OD节点不一致')
        if any(x[1]!=y[0] for x,y in zip(obs,obs[1:])):
            raise ValueError('HMM边序列不是连续的有向路')
        bad=[e for e in obs if e not in road.edges]
        if bad:
            raise ValueError(f'观察路径包含{len(bad)}条不属于可用机动车路网的边；示例={bad[0]}')
        observed_length=sum(road.edges[e] for e in obs)
        shortest=natural_audit['shortest_length_km']*1000
        if shortest<=0:raise ValueError('没有可用最短路长度')
        obs_ratio=observed_length/shortest
        aud['observed_vs_shortest']=float(obs_ratio)
        if obs_ratio>a.max_route_ratio+1e-8:
            aud.update(status='observed_exceeds_length_cap',
                       reason=f'观察路径与最短路比值{obs_ratio:.4f}超过统一约束{a.max_route_ratio}')
            return aud,[]
        if obs_ratio<1-1e-5:
            aud.update(status='observed_shorter_than_shortest',
                       reason=f'图搜索最短路比观察路径长，怀疑路网路径/方向不一致，ratio={obs_ratio:.7f}')
            return aud,[]
        seqs=[observed_edges(x['edge_seq_uvk_json']) for x in raw]
        same=[j for j,seq in enumerate(seqs) if seq==obs]
        if len(same)>1:raise ValueError('自然候选出现重复真实序列')
        if same:
            chosen_idx=same[0];aud['real_in_natural_TRACE_NOT_FEATURE']=1
        else:
            chosen_idx=len(seqs);seqs.append(obs)
            aud['real_added_TRACE_NOT_FEATURE']=1
        ps=path_sizes(seqs,road)
        rows=[]
        for j,(seq,pv) in enumerate(zip(seqs,ps)):
            prop=route_properties(seq,road,a.turn_angle)
            prop.update(pv)
            prop['ratio_to_shortest']=sum(road.edges[e] for e in seq)/shortest
            prop['detour_ratio_vs_straight']=prop['total_len_km']/float(r['od_straight_km'])
            if j<len(raw):cid=raw[j]['candidate_id_TRACE_NOT_FEATURE']
            else:cid=f'{t}_observed_augmented_{route_hash(seq)}'
            rows.append({'trip_id':t,'VehicleID_TRACE_NOT_FEATURE':vid,
                         'split_stage':stage,'start_time':r['start_time'],
                         'candidate_id_TRACE_NOT_FEATURE':cid,
                         'chosen':int(j==chosen_idx),
                         'real_added_TRACE_NOT_FEATURE':int(j==chosen_idx and not same),
                         **prop})
        if len(rows)<2:
            aud.update(status='conditional_only_one',reason='可用条件选择路径不足2条')
            return aud,[]
        if sum(z['chosen'] for z in rows)!=1:raise ValueError('chosen数量不是1')
        aud.update(status='valid_labeled',condition_count=len(rows))
        return aud,rows
    except Exception as exc:
        if isinstance(exc, (TypeError, AttributeError, NameError, ImportError)):
            raise RuntimeError(
                f'历史监督候选发生程序级异常，禁止按坏OD吞掉：trip_id={t}, '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        aud.update(status='invalid_observed_or_features',reason=f'{type(exc).__name__}: {exc}')
        return aud,[]


def program_self_check():
    """只做结构自检，不生成/修改任何实验数据。"""
    try:
        v = EdgeInfo(123.0, 'major', 1, 2.0, 60.0, None, None, False)
    except Exception as e:
        raise RuntimeError('代码自检失败：EdgeInfo构造器异常；请确认已替换旧版脚本') from e
    if v.length!=123.0 or v.group!='major' or v.precise_geom is not False:
        raise RuntimeError('代码自检失败：EdgeInfo存储字段不正确')
    info('程序结构自检通过：EdgeInfo已可正常实例化')


def real_graph_preflight(road, turn_angle):
    """使用当前真实GraphML中的一条实际道路边作快速特征自检，不构造实验数据。"""
    edge = next((e for e in road.edges
                 if e[0] in road.xy and e[1] in road.xy and e[0]!=e[1]), None)
    if edge is None:
        raise RuntimeError('真实GraphML没有可用的带坐标有向边，无法计算候选属性')
    try:
        prop = route_properties([edge], road, turn_angle)
        ps = path_sizes([[edge]], road)[0]
        if abs(prop['total_len_km']*1000-road.edges[edge])>1e-5:
            raise AssertionError('单边距离不守恒')
        if abs(ps['ps_length']-1)>1e-9:
            raise AssertionError('单条路线的PathSize应等于1')
    except Exception as exc:
        raise RuntimeError(
            f'实际GraphML道路边特征自检失败：edge={edge}, '
            f'{type(exc).__name__}: {exc}'
        ) from exc
    info('真实GraphML单边特征自检通过：道路属性和Path Size可正常计算')


def diagnose_existing_db(a):
    """只读旧SQLite缓存，输出此前逐OD真实失败原因。不会加载路网。"""
    p = a.work/'00_历史条件候选断点续跑.sqlite3'
    require(p,'旧版SQLite候选缓存')
    db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
    counts=Counter(); reasons=Counter(); samples=defaultdict(list); n=0
    try:
        for (raw,) in db.execute('SELECT audit FROM od'):
            z=json.loads(raw);n+=1
            st=z.get('status','unknown');reason=str(z.get('reason',''))
            counts[st]+=1;reasons[(st,reason)]+=1
            if len(samples[st])<5:
                samples[st].append({'trip_id':z.get('trip_id'),
                                    'natural_status':z.get('natural_status'),
                                    'natural_count':z.get('natural_count'),
                                    'reason':reason})
        meta=db.execute('SELECT c FROM meta WHERE id=1').fetchone()
        saved_config=json.loads(meta[0]) if meta else None
    finally:
        db.close()
    obj={'mode':'READONLY_CACHE_DIAGNOSTIC','n_od':n,
         'status_counts':dict(counts),
         'reasons':[{'status':st,'reason':reason,'n_od':count}
                    for (st,reason),count in reasons.most_common()],
         'samples':dict(samples),'cached_config':saved_config}
    a.work.mkdir(parents=True,exist_ok=True)
    writejson(a.work/'00_SQLite旧记录只读诊断.json',obj)
    info(f'历史SQLite已处理 {n:,} 条OD；状态分布：{dict(counts)}')
    for (st,reason),count in reasons.most_common(12):
        info(f'  {count:,}条 [{st}] {reason[:240]}')
    info('只读诊断已保存：'+str(a.work/'00_SQLite旧记录只读诊断.json'))


def prep_db(path,config,reset):
    if reset and path.exists():
        path.unlink()
        for s in ('-wal','-shm'):
            p=Path(str(path)+s)
            if p.exists():p.unlink()
    db=sqlite3.connect(str(path));db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS meta(id INTEGER PRIMARY KEY,c TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS od(trip_id TEXT PRIMARY KEY,audit TEXT NOT NULL,rows TEXT NOT NULL)')
    enc=jsonsafe(config)
    val=db.execute('SELECT c FROM meta WHERE id=1').fetchone()
    if val is None:db.execute('INSERT INTO meta VALUES(1,?)',(enc,));db.commit()
    elif val[0]!=enc:
        db.close();raise RuntimeError('已有旧版/不同配置的SQLite进度（旧版可能缓存了全部失败OD），请先运行 --phase diagnose，然后使用 --reset-db 重建；禁止混用')
    return db


def prepare(a,sources,man,tm):
    out=a.work;out.mkdir(parents=True,exist_ok=True)
    dev,stages=read_development(a,man)
    source_spec={
        'step41_dev_sha256':sha(sources['dev']),
        'graph_sha256':tm['source_config']['input_source']['graphml']['sha256'],
        'module1_manifest_sha256':sha(a.m1/'11_完整性防泄漏审计.json'),
        'k':a.k,'max_attempts':a.max_attempts,'max_route_ratio':a.max_route_ratio,
        'penalty_start':a.penalty_start,'penalty_step':a.penalty_step,
        'max_expanded':a.max_expanded,'turn_angle':a.turn_angle,
        'vehicle_filter':a.vehicle_filter,'min_straight_km':0.,
        'max_node_coordinate_error_m':a.max_node_coordinate_error_m,
        'pilot':a.pilot,'pilot_od':a.pilot_od,'seed':a.seed,
        'n_input_od':len(dev),'version':'module4_condition_choice_build_v1.2_edgeinfo_and_audit'}
    db=prep_db(out/'00_历史条件候选断点续跑.sqlite3',source_spec,a.reset_db)
    done={z[0] for z in db.execute('SELECT trip_id FROM od')}
    target={r['trip_id'] for r in dev}
    if not done<=target:raise RuntimeError('SQLite含本批次之外的OD，拒绝续跑')
    info(f'前28天训练+验证历史OD {len(dev):,}条；已经完成 {len(done):,}；剩余 {len(dev)-len(done):,}')
    road=None
    if len(done)<len(dev):
        road=RoadNetwork(a.graphml,a.vehicle_filter)
        if road.stats['graph_directed_edges'] !=tm['graph_stats']['graph_directed_edges']:
            raise ValueError('当前路网有向边数与模块三不一致')
        real_graph_preflight(road, a.turn_angle)
        n=0
        accumulated_status=Counter()
        for row in dev:
            if row['trip_id'] in done:continue
            audit,rows=historical_one_od(road,row,a)
            accumulated_status[audit['status']]+=1
            with db:db.execute('INSERT INTO od VALUES(?,?,?)',(
                row['trip_id'],jsonsafe(audit),jsonsafe(rows)))
            n+=1
            if n%20==0 or n==len(dev)-len(done):
                info(f'真实历史候选准备进度 {len(done)+n:,}/{len(dev):,}，'
                     f'当前OD={row["trip_id"]}，自然={audit["natural_count"]}，条件={audit["condition_count"]}，'
                     f'状态={audit["status"]}，原因={str(audit["reason"])[:130]}，阶段统计={dict(accumulated_status)}')
    # 完整导出（不能把逐OD失败静默删除）
    observed=list(db.execute('SELECT trip_id,audit,rows FROM od'))
    if len(observed)!=len(dev):raise RuntimeError('SQLite历史OD尚未准备完整，拒绝发布正式数据')
    lookup={t:(json.loads(au),json.loads(rr)) for t,au,rr in observed}
    all_a=[];all_cond=[]
    for r in dev:
        au,rr=lookup[r['trip_id']];all_a.append(au);all_cond.extend(rr)
    db.close()
    audit=pd.DataFrame(all_a)
    cond=pd.DataFrame(all_cond)
    # 先保存全部样本状态，即使一条有效样本也没有，也必须能追查真正失败原因。
    savecsv(audit,out/'01_开发OD候选生成与真实补入审计.csv')
    fail_stats=audit.groupby(['split_stage','status','reason'],dropna=False).size().rename('n_od').reset_index()
    fail_stats=fail_stats.sort_values('n_od',ascending=False).reset_index(drop=True)
    savecsv(fail_stats,out/'00_开发OD全部状态和失败原因统计.csv')
    info('历史候选状态汇总：'+str(audit.status.value_counts().to_dict()))
    for z in fail_stats.head(12).itertuples(index=False):
        info(f'  状态={z.status} 数量={z.n_od:,} 原因={str(z.reason)[:160]}')
    if cond.empty:
        writejson(out/'00_无有效监督样本_诊断信息.json',{
            'selected_od':len(dev),
            'status_counts':audit.status.value_counts().to_dict(),
            'top_failures':fail_stats.head(40).to_dict(orient='records'),
            'warning':'这不是模型训练结果；必须修复错误而不能伪造或人为放宽路径标签'
        })
        raise RuntimeError('未生成任何有效条件训练样本！已导出01_开发OD候选生成与真实补入审计.csv、'
                           '00_开发OD全部状态和失败原因统计.csv和00_无有效监督样本_诊断信息.json，'
                           '请先核查真实失败原因，不可继续fit')
    sizes=cond.groupby('trip_id',sort=False).agg(n=('chosen','size'),hit=('chosen','sum'))
    if ((sizes.n<2)|(sizes.hit!=1)).any():raise ValueError('条件OD候选数或唯一真实选择标签错误')
    if cond['candidate_id_TRACE_NOT_FEATURE'].duplicated().any():raise ValueError('候选ID重复')
    # 显式为审计文件保留候选来源信息；入模时排除全部TRACE、chosen和ID字段。
    savecsv(cond,out/'02_训练验证条件候选与静态属性.csv.gz')
    report={
        'mode':'PILOT_NOT_FOR_PAPER' if a.pilot else 'FULL_REAL',
        'training_od_total':stages['model_train'],'validation_od_total':stages['validation'],
        'selected_od':len(dev),'condition_valid_od':int((audit.status=='valid_labeled').sum()),
        'conditional_candidate_rows':len(cond),
        'natural_count_total':int(audit.natural_count.sum()),
        'real_added_n':int((audit.real_added_TRACE_NOT_FEATURE==1).sum()),
        'real_natural_hit_n':int((audit.real_in_natural_TRACE_NOT_FEATURE==1).sum()),
        'status_counts':audit.status.value_counts().to_dict(),
        'mean_cond_candidates':float(sizes.n.mean()),
        'input_source_sha256':source_spec,
        'warning':[
            '训练/验证条件候选集合允许真实路线补入；该条件集合不等于未知OD自然候选集合。',
            '条件选择Hit@1只衡量历史标签识别，不能当作真实导航命中率。',
            '不使用任何后三天真实路线作为训练标签。',
            '候选超过长度上限、非法观察边或生成失败的OD均有原因记录。',
        ]}
    writejson(out/'03_历史候选准备审计.json',report)
    info(f'准备完成：有效历史条件OD {report["condition_valid_od"]:,} / {len(dev):,}；'
         f'含观察补入的OD {report["real_added_n"]:,}；逐条静态候选 {len(cond):,}')
    return report

# ======== 保持原Step457的特征交互与精确条件Logit =========

class CandidateScaler:
    def fit(self,df):
        self.params=[]
        for col in SCALE_COLS:
            if col not in df.columns:raise ValueError(f'候选数据缺少原Step457特征列：{col}')
            x=pd.to_numeric(df[col],errors='coerce').to_numpy(float)
            mask=np.isfinite(x)
            if mask.sum()<max(100,len(x)*.05):
                raise ValueError(f'{col}训练期可用数太少：{mask.sum()} / {len(x)}')
            med=float(np.median(x[mask]))
            z=np.where(mask,x,med)
            avg=float(z.mean());std=float(z.std())
            self.params.append({'name':col,'median':med,'mean':avg,
                                'std':std if std>1e-8 else 1.,
                                'train_missing_n':int((~mask).sum()),
                                'train_missing_rate':float((~mask).mean())})
        return self
    def transform(self,df):
        cols=[]
        for p in self.params:
            if p['name'] not in df.columns:raise ValueError('缺少模型入参：'+p['name'])
            z=pd.to_numeric(df[p['name']],errors='coerce').to_numpy(float)
            z=np.where(np.isfinite(z),z,p['median'])
            cols.append(np.clip((z-p['mean'])/p['std'],-8,8))
        x=np.stack(cols,axis=1)
        if not np.isfinite(x).all():raise FloatingPointError('标准化后仍存在NaN/inf')
        return x
    @classmethod
    def from_params(cls,ps):
        z=cls();z.params=ps
        if [t['name'] for t in ps]!=SCALE_COLS:raise ValueError('模型标准化字段顺序不同于约定结构')
        return z


def features_for_models(df,sc):
    z=sc.transform(df)
    idx={s:j for j,s in enumerate(SCALE_COLS)}
    blocks={'base':z[:,:10],
            'ps_count':z[:,[idx['log_ps_count']]],
            'ps_length':z[:,[idx['log_ps_length']]]}
    for dim,attrs in [('econ',ECON_ROUTE),('comf',COMF_ROUTE)]:
        fields=[f'history_{dim}_{j}' for j in range(4)]
        if not set(fields)<=set(df.columns):raise ValueError(f'{dim}画像缺失：{sorted(set(fields)-set(df.columns))}')
        q=df[fields].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        if not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(axis=1),1,atol=1e-5):
            raise ValueError(f'{dim}历史软概率非法或未归一化')
        blocks[dim]=np.column_stack([(q[:,k]-.25)*z[:,idx[c]]
                                      for k in range(4) for c in attrs])
    return blocks


def by_groups(df,needs_label=True):
    if df.empty:raise ValueError('候选数据为空')
    size=df.groupby('trip_id',sort=False).size()
    expected=np.repeat(size.index.to_numpy(dtype=str),size.to_numpy())
    if not np.array_equal(expected,df.trip_id.astype(str).to_numpy()):
        raise ValueError('候选没有按trip_id连续排列，不能用分段softmax')
    sizes=size.to_numpy(int)
    if (sizes<1).any():raise ValueError('OD没有路线')
    if needs_label:
        if (sizes<2).any():raise ValueError('条件训练OD候选数少于2')
        starts=np.r_[0,np.cumsum(sizes)[:-1]]
        labels=df['chosen'].to_numpy(float)
        if not np.array_equal(np.add.reduceat(labels,starts),np.ones(len(sizes))):
            raise ValueError('训练组chosen不是恰好1条真实路线')
    return sizes,size.index.to_numpy(dtype=str)


def prob_softmax(x,beta,sizes):
    sizes=np.asarray(sizes,int)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    grp=np.repeat(np.arange(len(sizes)),sizes)
    s=x@np.asarray(beta,float)
    shifted=np.exp(s-np.maximum.reduceat(s,starts)[grp])
    den=np.add.reduceat(shifted,starts)
    probs=shifted/den[grp]
    if not np.isfinite(probs).all() or not np.allclose(np.add.reduceat(probs,starts),1,atol=1e-8):
        raise ValueError('条件概率数值异常')
    return s,probs


def logit_fit(x,sizes,labels,ridge,max_iter):
    x=np.asarray(x,np.float64,order='C');labels=np.asarray(labels,np.float64)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    grp=np.repeat(np.arange(len(sizes)),sizes)
    n=len(sizes)
    if len(x)!=labels.size or len(x)!=int(np.sum(sizes)):
        raise ValueError('模型行数与OD候选数不对应')
    if not np.array_equal(np.add.reduceat(labels,starts),np.ones(n)):
        raise ValueError('条件Logit训练标签错误')
    def obj(beta):
        v=x@beta
        vmax=np.maximum.reduceat(v,starts)
        ex=np.exp(v-vmax[grp]);den=np.add.reduceat(ex,starts)
        p=ex/den[grp]
        loss=float((np.sum(vmax+np.log(den))-np.dot(labels,v))/n +ridge*.5*np.dot(beta,beta))
        grad=x.T@(p-labels)/n+ridge*beta
        return loss,grad
    res=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options={'maxiter':max_iter,'gtol':1e-6,'ftol':1e-10,'maxls':30})
    if not np.isfinite(res.x).all() or not np.isfinite(res.fun):
        raise RuntimeError(f'条件Logit出现NaN/Inf: {res.message}')
    return {'beta':res.x,'converged':bool(res.success),'iterations':int(res.nit),
            'objective':float(res.fun),'optimizer_message':str(res.message)}


def view_cond(a):
    p=a.work/'02_训练验证条件候选与静态属性.csv.gz'
    require(p,'模块四已准备的条件选择候选')
    d=pd.read_csv(p,encoding='utf-8-sig',low_memory=False,dtype={'trip_id':'string'})
    if d.empty:raise ValueError('条件候选表为空')
    required=['trip_id','split_stage','chosen','candidate_id_TRACE_NOT_FEATURE']+SCALE_COLS
    if not set(required)<=set(d.columns):raise ValueError('条件候选表缺失'+str(sorted(set(required)-set(d.columns))))
    if d['candidate_id_TRACE_NOT_FEATURE'].duplicated().any():raise ValueError('重复候选ID')
    if not set(d.split_stage.unique())<=set(['model_train','validation']):
        raise ValueError('条件训练集混入了未知测试阶段')
    p=a.m2/'03_训练验证逐OD严格历史画像.csv.gz'
    require(p,'模块二训练验证逐OD历史画像')
    pp=pd.read_csv(p,encoding='utf-8-sig',low_memory=False,dtype={'trip_id':'string'})
    requiredp=['trip_id','history_n_prior']+PROFILE
    if not set(requiredp)<=set(pp.columns):
        raise ValueError('模块二训练验证画像缺失：'+str(sorted(set(requiredp)-set(pp.columns))))
    if pp.trip_id.duplicated().any():raise ValueError('模块二训练验证画像trip_id重复')
    d=d.merge(pp[requiredp],on='trip_id',how='left',validate='many_to_one')
    if d[PROFILE].isna().any().any():raise ValueError('部分历史条件候选找不到偏好画像')
    d=d.sort_values(['split_stage','trip_id','candidate_id_TRACE_NOT_FEATURE'],kind='mergesort').reset_index(drop=True)
    by_groups(d,True)
    ainfo=pd.read_csv(a.work/'01_开发OD候选生成与真实补入审计.csv',encoding='utf-8-sig')
    valid=set(ainfo.loc[ainfo.status=='valid_labeled','trip_id'].astype(str))
    if valid!=set(d.trip_id.astype(str)):
        raise ValueError('条件候选OD和准备审计中有效OD集合不一致')
    return d


def validation_rows(model,df,prob):
    sizes,tids=by_groups(df,True)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    chosen=df.chosen.to_numpy(int)
    rows=[]
    for t,st,n in zip(tids,starts,sizes):
        selected=int(np.flatnonzero(chosen[st:st+n]==1)[0])
        order=np.argsort(-prob[st:st+n],kind='mergesort')
        rank=int(np.flatnonzero(order==selected)[0]+1)
        p=float(prob[st+selected])
        rows.append({'trip_id':t,'model':model,'n_candidates':int(n),
                     'chosen_rank':rank,'chosen_probability':p,
                     'hit1':int(rank==1),'mrr':1/rank,'nll':-math.log(max(p,1e-15))})
    return pd.DataFrame(rows)


def fit(a):
    out=a.work
    model_path=out/'06_正式模型参数与训练来源.json'
    if model_path.exists() and not a.overwrite_results:
        raise FileExistsError(f'已有正式模型输出：{model_path}；确认重训请加 --overwrite-results')
    cond=view_cond(a)
    train=cond[cond.split_stage=='model_train'].reset_index(drop=True)
    val=cond[cond.split_stage=='validation'].reset_index(drop=True)
    if train.empty or val.empty:raise ValueError('训练或验证集合为空')
    ntrain=by_groups(train,True)[0];nval=by_groups(val,True)[0]
    if len(ntrain)<20 or len(nval)<20:
        raise ValueError('有效训练或验证OD过少，不能正式估计')
    scaler=CandidateScaler().fit(train)
    ct=features_for_models(train,scaler);cv=features_for_models(val,scaler)
    try:ridges=sorted({float(x.strip()) for x in a.ridge_grid.split(',') if x.strip()})
    except ValueError as e:raise ValueError('--ridge-grid需要如0.003,0.01,0.03') from e
    if not ridges or min(ridges)<=0 or not np.isfinite(ridges).all():
        raise ValueError('L2候选值必须是正数且有限')
    ytrain=train.chosen.to_numpy(float)
    candidate_rows=[];validation_predictions=[];chosen_ridges={};logs=[]
    for name,blocks in MODELS.items():
        xx=np.hstack([ct[b] for b in blocks]);vv=np.hstack([cv[b] for b in blocks])
        best=None
        for ridge in ridges:
            t0=time.perf_counter()
            ff=logit_fit(xx,ntrain,ytrain,ridge,a.max_iter)
            if not ff['converged'] and not a.allow_nonconverged:
                raise RuntimeError(f'{name},ridge={ridge}:条件Logit未收敛({ff["optimizer_message"]})；'
                                   '建议提高 --max-iter，不能伪称已收敛')
            _,pv=prob_softmax(vv,ff['beta'],nval)
            valid=validation_rows(name,val,pv)
            score=float(valid.nll.mean())
            row={'model':name,'ridge':ridge,'n_train_od':len(ntrain),
                 'n_validation_od':len(nval),'hit1':float(valid.hit1.mean()),
                 'mrr':float(valid.mrr.mean()),'nll':score,
                 'train_objective':ff['objective'],'iterations':ff['iterations'],
                 'converged':ff['converged'],'elapsed_seconds':time.perf_counter()-t0}
            candidate_rows.append(row)
            info(f'验证 {name} ridge={ridge}: NLL={score:.6f} MRR={row["mrr"]:.6f}, Hit1={row["hit1"]:.6f}')
            if best is None or (score,ridge)<(best[0],best[1]):best=(score,ridge,valid)
        chosen_ridges[name]=float(best[1]);validation_predictions.append(best[2]);
    # 最终重拟合仅允许 model_train + validation，严格不包含12月29-31真实路径。
    combined=cond.reset_index(drop=True)
    sfinal=CandidateScaler().fit(combined)
    cf=features_for_models(combined,sfinal)
    nf=by_groups(combined,True)[0];yf=combined.chosen.to_numpy(float)
    saved={}
    for name,blocks in MODELS.items():
        xf=np.hstack([cf[b] for b in blocks]);ridge=chosen_ridges[name]
        ff=logit_fit(xf,nf,yf,ridge,a.max_iter)
        if not ff['converged'] and not a.allow_nonconverged:
            raise RuntimeError(f'最终{name}未收敛：{ff["optimizer_message"]}')
        saved[name]={'blocks':blocks,'ridge':ridge,'beta':ff['beta'].tolist(),
                     'n_coefficients':len(ff['beta']),'converged':ff['converged'],
                     'iterations':ff['iterations'],'train_objective':ff['objective'],
                     'optimizer_message':ff['optimizer_message']}
        info(f'最终重拟合 {name}：OD={len(nf):,}，系数={len(ff["beta"])}，收敛={ff["converged"]}')
    allval=pd.concat(validation_predictions,ignore_index=True)
    valsummary=allval.groupby('model',sort=False).agg(n_od=('trip_id','size'),
        hit1=('hit1','mean'),mrr=('mrr','mean'),nll=('nll','mean')).reset_index()
    valsummary['best_ridge']=valsummary.model.map(chosen_ridges)
    if valsummary.n_od.nunique()!=1:raise ValueError('六模型验证OD数量不相同')
    savecsv(pd.DataFrame(candidate_rows),out/'04_不同L2参数验证对比.csv')
    savecsv(allval,out/'05_验证期逐OD条件选择指标.csv.gz')
    savecsv(valsummary,out/'05B_六模型验证均值指标.csv')
    saved_params={
        'version':'chapter4_module4_six_models_time_isolated_v1.0',
        'created_at':datetime.now().isoformat(),
        'pilot':a.pilot,'use_test_observed_routes':False,
        'behavioral_profile_source':'module2 frozen historical 6+6 UDPPMpp soft profile',
        'model_features':SCALE_COLS,'base_features':BASE10,
        'economic_candidate_interaction':ECON_ROUTE,
        'comfort_candidate_interaction':COMF_ROUTE,
        'scaler_params':sfinal.params,
        'models':saved,
        'training_scope':{'calibration':'2024-12-01..14 for original 6+6 UDPPM++',
            'conditional_train':'2024-12-15..23','hyperparameter_validation':'2024-12-24..28',
            'final_fit':'2024-12-15..28 only',
            'test':'2024-12-29..31 never fit'},
        'n_train_od':len(ntrain),'n_validation_od':len(nval),'n_final_fit_od':len(nf),
        'n_train_candidates':len(train),'n_validation_candidates':len(val),
        'history_conditional_candidate_source':str(a.work/'02_训练验证条件候选与静态属性.csv.gz'),
        'history_candidate_sha256':sha(a.work/'02_训练验证条件候选与静态属性.csv.gz'),
        'profile_source_sha256':sha(a.m2/'03_训练验证逐OD严格历史画像.csv.gz'),
        'major_limitation':'包含真实路径的训练和验证条件集合；指标不是新OD真实导航Hit1',
        'convergence_policy':'require scipy success unless explicitly --allow-nonconverged'}
    writejson(model_path,saved_params)
    info('六模型条件选择训练与验证完成；验证Hit1/MRR/NLL不可解释为后三天自然路线推荐准确率')
    return saved_params


def blind_predict(a,tm):
    out=a.work
    model_file=out/'06_正式模型参数与训练来源.json'
    model=readjson(model_file)
    if model['pilot']!=a.pilot:raise RuntimeError('正式/试运行模型来源与当前模式不一致，停止')
    if model['model_features']!=SCALE_COLS:raise ValueError('加载的模型特征表不一致')
    featfile=a.m3/'02_未知OD候选静态属性与PathSize_模型输入.csv.gz'
    test=pd.read_csv(featfile,encoding='utf-8-sig',low_memory=False,dtype={'trip_id':'string'})
    blocked={'chosen','VehicleID','edge_seq_uvk_json','real_added_TRACE_NOT_FEATURE','is_real_added'}
    if blocked.intersection(test.columns):raise RuntimeError('后三天盲测候选混入真实行驶、标签或补入字段')
    if len(test)!=int(tm['all_generated_natural_candidates']):
        raise RuntimeError('后三天候选行数与模块三审计不一致')
    raw_profile=pd.read_csv(a.m2/'05_最后3天盲推理8维画像_只按trip_id.csv',
                            encoding='utf-8-sig',dtype={'trip_id':'string'})
    if len(raw_profile)!=int(tm['selected_od']):raise ValueError('后三天画像OD数量异常')
    if raw_profile.trip_id.duplicated().any():raise ValueError('测试画像trip_id重复')
    if not set(PROFILE+['history_n_prior'])<=set(raw_profile.columns):
        raise ValueError('模块二后三天画像缺少必需字段')
    test=test.merge(raw_profile[['trip_id','history_n_prior']+PROFILE],
                    on='trip_id',how='left',validate='many_to_one')
    if test[PROFILE].isna().any().any():raise ValueError('后三天候选历史画像存在缺失')
    if a.pilot:
        all_ids=sorted(test.trip_id.unique().tolist())
        rng=np.random.default_rng(a.seed)
        ids=set(rng.choice(all_ids,min(a.pilot_od,len(all_ids)),replace=False).tolist())
        test=test[test.trip_id.isin(ids)].reset_index(drop=True)
    else:
        if test.trip_id.nunique()!=int(tm['blind_test_1plus_od']):
            raise ValueError('实际可预测OD数量与模块三成功OD数量不一致')
    test=test.sort_values(['trip_id','candidate_id_TRACE_NOT_FEATURE'],kind='mergesort').reset_index(drop=True)
    if test.candidate_id_TRACE_NOT_FEATURE.duplicated().any():raise ValueError('后三天候选ID重复')
    sizes,ids=by_groups(test,False)
    scal=CandidateScaler.from_params(model['scaler_params'])
    blocks=features_for_models(test,scal)
    predictions=[];top3=[];times=[]
    for name,mm in model['models'].items():
        if name not in MODELS or mm['blocks']!=MODELS[name]:raise ValueError('模型配置与既定六模型不一致')
        X=np.hstack([blocks[z] for z in mm['blocks']])
        if len(mm['beta'])!=X.shape[1]:raise ValueError(f'{name}模型系数和输入维数不同')
        t0=time.perf_counter()
        util,prob=prob_softmax(X,mm['beta'],sizes)
        times.append({'model':name,'n_od':len(ids),'n_candidate_rows':len(test),
                      'score_and_rank_seconds':None})
        cur=test[['trip_id','candidate_id_TRACE_NOT_FEATURE','total_len_km',
                  'ratio_to_shortest','turns_per_km','segment_density_seg_per_km']].copy()
        cur['model']=name;cur['utility']=util;cur['choice_probability']=prob
        cur['rank_in_od']=cur.groupby('trip_id',sort=False).choice_probability.rank(
            method='first',ascending=False).astype(int)
        if not (cur.groupby('trip_id').choice_probability.sum().sub(1).abs()<1e-6).all():
            raise AssertionError('后三天条件概率分组未归一')
        predictions.append(cur)
        ranked=cur.loc[cur.rank_in_od<=3].copy()
        top3.append(ranked)
        times[-1]['score_and_rank_seconds']=round(time.perf_counter()-t0,5)
        info(f'后三天盲排序 {name}: {len(ids):,}个OD, Top3实际行={len(ranked):,},'
             f' 计算耗时={times[-1]["score_and_rank_seconds"]:.3f}s')
    pred=pd.concat(predictions,ignore_index=True)
    out3=pd.concat(top3,ignore_index=True)
    # 推荐结果严格不含chosen、真实路径长度、真实VehicleID等字段。
    savecsv(pred,out/'07_最后3天六模型逐路线盲排序.csv.gz')
    savecsv(out3,out/'08_最后3天六模型Top3盲推荐.csv.gz')
    ct=pd.read_csv(a.m3/'03_全部OD搜索与Top3审计.csv',encoding='utf-8-sig',dtype={'trip_id':'string'})
    if len(ct)!=int(tm['selected_od']):raise ValueError('模块三OD审计缺少失败OD')
    if not a.pilot:
        if set(ct.loc[ct.natural_count.astype(int)>0,'trip_id'].astype(str))!=set(ids):
            raise ValueError('测试预测OD与模块三所有自然候选成功OD不一致')
        # 预期全量所有失败OD依然被计入分母。
        odtot=int(tm['selected_od']);top3od=int(tm['blind_test_3plus_od']);
    else:
        odtot=len(ids);top3od=int(np.sum(sizes>=3))
    audit={
        'version':'chapter4_module4_six_models_blind_inference_v1.0',
        'mode':'PILOT_NOT_FOR_PAPER' if a.pilot else 'FULL_REAL',
        'created_at':datetime.now().isoformat(),
        'source_model_file':str(model_file),'source_model_sha256':sha(model_file),
        'source_test_candidates':str(featfile),'source_test_candidates_sha256':sha(featfile),
        'n_all_test_od':odtot,'n_scored_od':len(ids),'n_top3_available_od':top3od,
        'n_test_candidate_rows':len(test),'n_models':len(MODELS),
        'n_saved_all_model_prediction_rows':len(pred),
        'n_saved_top3_rows':len(out3),
        'per_model_scoring_time':times,
        'actual_observed_routes_read':'NEVER','chosen_test_labels_read':'NEVER',
        'future_profiles_reestimated':'NEVER',
        'graphml_for_test':'NOT_LOADED (Module3 natural features only)',
        'warning':[
            '这里的概率是各OD自然候选集合内部的条件softmax，不是真实舒适性评分或事故安全概率。',
            '本模块只提供冻结模型的盲排序；无真实标签，因此不生成未来Hit@1/NLL。',
            '下一模块评价Top-3推荐集合的多目标质量与多样性。',
            '训练含观察真实路线的条件选择集合，推荐的是纯自然候选，二者分布可能不同。'
        ]}
    writejson(out/'09_最后3天盲排序与时间隔离审计.json',audit)
    info(f'完成：{len(ids):,}个OD可排序；至少3条的OD={top3od:,}；最后3天未读取真值')
    return audit


def get_args():
    p=argparse.ArgumentParser(description='模块四：真实训练条件候选、六模型Logit与后三天自然候选盲排序')
    p.add_argument('--phase',choices=['check','diagnose','prepare','fit','predict','all'],default='check')
    p.add_argument('--m1',type=Path,default=M1)
    p.add_argument('--m2',type=Path,default=M2)
    p.add_argument('--m3',type=Path,default=M3)
    p.add_argument('--graphml',type=Path,default=GRAPH)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--pilot',action='store_true',help='真实小样本功能测试；不可作为论文指标')
    p.add_argument('--pilot-od',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--k',type=int,default=10)
    p.add_argument('--max-attempts',type=int,default=50)
    p.add_argument('--max-route-ratio',type=float,default=2.)
    p.add_argument('--penalty-start',type=float,default=.20)
    p.add_argument('--penalty-step',type=float,default=.07)
    p.add_argument('--max-expanded',type=int,default=800000)
    p.add_argument('--turn-angle',type=float,default=45.)
    p.add_argument('--vehicle-filter',choices=['exclude_nonmotor','legacy_all_length_edges'],default='exclude_nonmotor')
    p.add_argument('--max-node-coordinate-error-m',type=float,default=50.)
    p.add_argument('--min-straight-km',type=float,default=0.)
    p.add_argument('--ridge-grid',default='0.003,0.01,0.03',
                   help='候选L2系数，验证期按NLL为每模型选择；如只运行0.01可设 --ridge-grid 0.01')
    p.add_argument('--max-iter',type=int,default=350)
    p.add_argument('--allow-nonconverged',action='store_true',help='仅诊断使用，不建议正式论文')
    p.add_argument('--reset-db',action='store_true',help='明确删除并重建历史候选SQLite；不可恢复')
    p.add_argument('--overwrite-results',action='store_true',help='明确覆盖之前拟合的模型及结果')
    p.add_argument('--no-graph-hash',dest='verify_graph_sha',action='store_false')
    p.set_defaults(verify_graph_sha=True)
    return p.parse_args()


def main():
    a=get_args()
    if a.max_attempts<1 or a.k<3 or a.max_route_ratio<1 or a.max_iter<10 or a.pilot_od<20:
        raise ValueError('搜索/模型参数不合法；k>=3、max_attempts>=1、max_iter>=10、pilot-od>=20')
    if a.min_straight_km!=0:
        raise ValueError('正式候选必须与模块三完全一样 min_straight_km=0')
    a.work=a.out/('pilot_od'+str(a.pilot_od)+'_seed'+str(a.seed) if a.pilot else 'formal_K10')
    # diagnose模式不依赖其他原始文件，用于读取旧版已保存的逐OD失败原因。
    if a.phase=='diagnose':
        diagnose_existing_db(a)
        return
    program_self_check()
    sources,man,pm,tm=check_inputs(a)
    info(f'前两模块核对通过，测试OD={man["counts"]["future_3d"]:,}，'
         f'模块三已自然生成候选{tm["all_generated_natural_candidates"]:,}条')
    if a.phase=='check':
        info('检查完成：未读取后三天真实路径，未开始大规模候选生成或训练')
        return
    if a.phase in ('prepare','all'):prepare(a,sources,man,tm)
    if a.phase in ('fit','all'):fit(a)
    if a.phase in ('predict','all'):blind_predict(a,tm)
    info(f'第四章模块四 {a.phase} 执行完成，输出：{a.work}')

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        info('用户中断：prepare阶段已完成的OD保存在SQLite中，重新运行相同参数可续跑')
        raise SystemExit(130)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(f'模块四停止：{type(e).__name__}: {e}')
