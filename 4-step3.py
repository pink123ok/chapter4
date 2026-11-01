# -*- coding: utf-8 -*-
"""
第四章 · 模块三 v1.0：未知OD自然候选路径、静态道路属性、Path Size（严格盲测）

真实数据：
  模块一/05_最后3天_纯OD推理输入_绝无真实路线.csv
  模块一/11_完整性防泄漏审计.json
  正式有向路网 C:/study/轨迹数据预处理/luwangdata/曼谷_路网_兼容旧版.graphml

严格限制：
  - 不读取真实HMM行驶边序列、chosen、司机身份、模块二预测画像。
  - 只使用OD坐标/起终点、GraphML。不会补入真实路线。
  - 沿用原Step42的有向(u,v,key) A*/Dijkstra及边惩罚式候选搜索。
  - 沿用原Step43的道路分组、几何转弯、长度加权和计数Path Size计算口径。
  - GraphML的完整道路边长度不等同于HMM估计的首末部分边行程长度。
  - 在生成所有候选路径后，按当次自然候选集合计算Path Size，绝不引用真值。
  - 仅预先剔除明确非机动车通行道路及OSM明确禁止机动车的边（可关闭）。
    本程序仍不能验证临时交通管制、所有转向限制、收费及实时拥堵。
  - 不生成或猜测任何实验结果数值；缺失真实输入必须报错。

依赖：python -m pip install -U pandas networkx numpy shapely matplotlib

Windows：
  # 仅核对真实输入，不读取大型GraphML：
  python "第四章_模块三_未知OD自然候选与静态属性PathSize_纯盲测.py" --check-input-only

  # 先试跑100条真实OD（独立目录，不可作为正式论文指标）：
  python "第四章_模块三_未知OD自然候选与静态属性PathSize_纯盲测.py" --mode pilot --pilot-size 100

  # 最后3天全量4469条OD：
  python "第四章_模块三_未知OD自然候选与静态属性PathSize_纯盲测.py" --mode full

  # 中断后原命令重跑即可自动续跑；参数或真实文件变更会被阻断。
  # 如需从头重新做当前输出目录，可显式 --reset。

注意：输出含三份独立文件：
  01_真实自然候选路径_含有向边序列.csv.gz（只含生成路径）
  02_未知OD候选静态属性与PathSize_模型输入.csv.gz（无chosen、无VehicleID）
  03_全部OD搜索与Top3审计.csv（包括失败OD）
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import random
import re
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath

try:
    import networkx as nx
    import pandas as pd
    from shapely import wkt
except ImportError as exc:
    raise SystemExit('依赖缺失：python -m pip install -U pandas networkx numpy shapely matplotlib\n'+str(exc))

BASE = Path(r'C:\study\轨迹数据预处理')
FULL = BASE/'曼谷GPS_正式清洗_202412'/'FULL_全量'
ROOT = FULL/'第四章_28天训练_3天独立测试'
MOD1 = ROOT/'模块一_时间划分与隔离'
DEFAULT_OD = MOD1/'05_最后3天_纯OD推理输入_绝无真实路线.csv'
DEFAULT_MANIFEST = MOD1/'11_完整性防泄漏审计.json'
DEFAULT_GRAPH = BASE/'luwangdata'/'曼谷_路网_兼容旧版.graphml'
DEFAULT_OUT = ROOT/'模块三_未知OD自然候选与静态道路属性'

OD_FIELDS = ['trip_id','start_time','origin_node','destination_node','od_straight_km',
             'origin_lon','origin_lat','destination_lon','destination_lat']
# 与Step43(2).py MODEL_ATTRS对应；禁止任何实际轨迹及标签字段。
MODEL_ATTRS = [
    'total_len_km','n_segments','segment_density_seg_per_km',
    'ratio_motorway','ratio_major','ratio_local','ratio_other',
    'oneway_ratio','oneway_attribute_coverage',
    'lanes_mean_w','lanes_attribute_coverage',
    'maxspeed_mean_known_kmh','maxspeed_attribute_coverage',
    'turn_count_45deg','u_turn_count_150deg','turns_per_km',
    'wkt_geometry_length_fraction','node_line_fallback_length_fraction',
    'ps_count','log_ps_count','ps_length','log_ps_length',
    'shared_edge_length_ratio','unique_edge_length_ratio',
    'ratio_to_shortest','detour_ratio_vs_straight'
]
# _TRACE_NOT_FEATURE为元信息标识，不能作为Logit特征。
FEATURE_FIELDS = ['trip_id','start_time','candidate_id_TRACE_NOT_FEATURE']+MODEL_ATTRS
RAW_FIELDS = ['trip_id','start_time','origin_node','destination_node','candidate_id_TRACE_NOT_FEATURE',
              'candidate_rank_TRACE_NOT_FEATURE','edge_seq_uvk_json','road_edge_count',
              'route_length_km','ratio_to_shortest','route_hash_TRACE_NOT_FEATURE']
OD_AUDIT_FIELDS = [
    'trip_id','start_time','origin_node','destination_node','od_straight_km',
    'status','reason','natural_count','top3_available','shortest_length_km',
    'candidate_min_km','candidate_max_km','attempts','expanded_total',
    'search_budget_hit','elapsed_seconds','search_states_json'
]
ROAD_PRIORITY = ['motorway','motorway_link','trunk','trunk_link','primary','primary_link',
                 'secondary','secondary_link','tertiary','tertiary_link','residential',
                 'living_street','service','unclassified','road','track','busway',
                 'footway','pedestrian','path','cycleway']
NONMOTOR = {'footway','pedestrian','path','cycleway','steps','bridleway',
            'platform','corridor','construction','proposed','busway'}

try: csv.field_size_limit(sys.maxsize)
except OverflowError: csv.field_size_limit(2**31-1)


def info(s):
    print(f'[{time.strftime("%H:%M:%S")}] {s}',flush=True)


def finite_num(v):
    try:
        z=float(v)
        return z if math.isfinite(z) else math.nan
    except (ValueError,TypeError):
        return math.nan


def nonempty_file(p,description):
    if not p.is_file() or p.stat().st_size == 0:
        raise FileNotFoundError(f'找不到{description}：{p}\n请检查本机真实文件路径，程序不产生模拟数据。')


def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def jsonsafe(o):
    return json.dumps(o,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=str)


def save_json(p,o):
    tmp=p.with_suffix(p.suffix+'.writing')
    tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8-sig')
    tmp.replace(p)


def read_blind_od(p,manifest):
    nonempty_file(p,'模块一纯OD文件')
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        read=csv.DictReader(f)
        fields=read.fieldnames or []
        if fields != OD_FIELDS:
            raise ValueError(f'纯OD列名或列序不符合白名单！\n应该是：{OD_FIELDS}\n实际是：{fields}\n'
                             '不得混入VehicleID、实际行程、chosen、真实边序列或未知预测字段。')
        raw=list(read)
    if not raw:raise RuntimeError('纯OD数据为空')
    if len(raw)!=len({r['trip_id'] for r in raw}):raise ValueError('纯OD出现重复trip_id')
    if len(raw)!=int(manifest['counts']['future_3d']):
        raise ValueError('纯OD数量与模块一审计future_3d不一致')
    for row in raw:
        trip=row['trip_id']
        if not trip:raise ValueError('空trip_id')
        try:date=datetime.fromisoformat(row['start_time']).date()
        except (TypeError,ValueError) as exc:raise ValueError(f'{trip}出发时间无效') from exc
        if not ('2024-12-29' <= date.isoformat() <= '2024-12-31'):
            raise ValueError(f'{trip}不是2024-12-29~31的测试OD，禁止与其他阶段混用')
        for col in OD_FIELDS[4:]:
            if not math.isfinite(finite_num(row[col])):
                raise ValueError(f'{trip}有非有限OD值：{col}={row[col]}')
        if float(row['od_straight_km'])<=0:raise ValueError(f'{trip}的OD直线距离不是正数')
        for stem in ('origin','destination'):
            lon=float(row[stem+'_lon']);lat=float(row[stem+'_lat'])
            if not (-180<=lon<=180 and -90<=lat<=90):raise ValueError(f'{trip}坐标越界：{stem}')
        if not row['origin_node'].strip() or not row['destination_node'].strip():
            raise ValueError(f'{trip}起终节点为空')
    return raw


def filemeta(p,hash_on=False):
    a={'path':str(p),'bytes':p.stat().st_size}
    if hash_on:a['sha256']=filehash(p)
    return a


def load_manifest(p,blind):
    nonempty_file(p,'模块一完整性防泄漏审计JSON')
    m=json.loads(p.read_text(encoding='utf-8-sig'))
    if m.get('version')!='chapter4_module1_chrono_split_v1.0':
        raise ValueError('模块一审计版本与预期不符，必须先检查来源')
    checks=m.get('checks',{})
    for k in ('dev_test_trip_id_disjoint','blind_od_columns_exact_whitelist','test_truth_stored_separately'):
        if checks.get(k) is not True:raise RuntimeError(f'模块一来源审计未通过：{k}')
    declared=str(m.get('outputs',{}).get('blind_od',''))
    if declared and PureWindowsPath(declared).name != blind.name:
        raise RuntimeError('本次纯OD文件名与模块一审计不一致，禁止混用实验')
    return m


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
        # OD级失败要完整审计；路径或特征阶段失败不能留下半套候选。
        base.update(status='od_error',reason=f'{type(exc).__name__}: {exc}',
                    natural_count=0,top3_available=0,
                    candidate_min_km=None,candidate_max_km=None)
        return base,[],[]
    finally:
        base['elapsed_seconds']=round(time.perf_counter()-t0,6)


def sqlite_prepare(path,config,reset):
    if reset and path.exists():
        path.unlink()
        for ext in ('-wal','-shm'):
            sub=Path(str(path)+ext)
            if sub.exists():sub.unlink()
    db=sqlite3.connect(str(path))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, config_json TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS od (trip_id TEXT PRIMARY KEY, data_json TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS route (candidate_id TEXT PRIMARY KEY, trip_id TEXT NOT NULL, raw_json TEXT NOT NULL, safe_json TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_route_trip ON route(trip_id)')
    old=db.execute('SELECT config_json FROM meta WHERE id=1').fetchone()
    encoded=jsonsafe(config)
    if old is None:
        db.execute('INSERT INTO meta(id,config_json) VALUES (1,?)',(encoded,))
        db.commit()
    elif old[0]!=encoded:
        db.close()
        raise RuntimeError('存在不同参数或输入文件的SQLite旧进度！请为新实验指定--out，或明确使用--reset。')
    return db


def sqlite_commit_od(db,audit,raw,safe):
    if len(raw)!=len(safe):raise RuntimeError('原始路径数与模型特征行数不一致')
    with db:
        db.execute('INSERT INTO od(trip_id,data_json) VALUES (?,?)',
                   (audit['trip_id'],jsonsafe(audit)))
        for rr,feat in zip(raw,safe):
            if rr['candidate_id_TRACE_NOT_FEATURE']!=feat['candidate_id_TRACE_NOT_FEATURE']:
                raise ValueError('候选ID不一致')
            db.execute('INSERT INTO route(candidate_id,trip_id,raw_json,safe_json) VALUES (?,?,?,?)',
                       (rr['candidate_id_TRACE_NOT_FEATURE'],audit['trip_id'],jsonsafe(rr),jsonsafe(feat)))


def open_write_csv(p,fieldnames):
    tmp=Path(str(p)+'.writing')
    if p.suffix=='.gz':f=gzip.open(tmp,'wt',encoding='utf-8-sig',newline='',compresslevel=5)
    else:f=open(tmp,'w',encoding='utf-8-sig',newline='')
    w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='raise')
    w.writeheader()
    return f,w,tmp


def export_results(db,out,selected,config,roadstats,full_n,elapsed):
    audits={r[0]:json.loads(r[1]) for r in db.execute('SELECT trip_id,data_json FROM od')}
    if len(audits)!=len(selected):
        raise RuntimeError(f'数据库已完成OD={len(audits)}，本次选择OD={len(selected)}；拒绝输出不完整正式结果')
    expected={r['trip_id'] for r in selected}
    if set(audits)!=expected:raise RuntimeError('数据库OD与当前输入集合不相同')
    p_raw=out/'01_真实自然候选路径_含有向边序列.csv.gz'
    p_safe=out/'02_未知OD候选静态属性与PathSize_模型输入.csv.gz'
    p_od=out/'03_全部OD搜索与Top3审计.csv'
    writers=[]
    for p,header in ((p_raw,RAW_FIELDS),(p_safe,FEATURE_FIELDS),(p_od,OD_AUDIT_FIELDS)):
        f,w,tmp=open_write_csv(p,header);writers.append((p,f,w,tmp))
    counts=Counter();sizes=Counter();day=defaultdict(lambda:Counter())
    n_route=0;missing_feature=Counter()
    try:
        for row in selected:
            trip=row['trip_id'];d=audits[trip]
            writers[2][2].writerow(d)
            counts[d['status']]+=1
            sizes[d['natural_count']]+=1
            day[row['start_time'][:10]]['od']+=1
            if d['natural_count']>0:day[row['start_time'][:10]]['at_least_1']+=1
            if d['natural_count']>=3:day[row['start_time'][:10]]['at_least_3']+=1
            n=0
            for v in db.execute('SELECT raw_json,safe_json FROM route WHERE trip_id=? ORDER BY candidate_id',(trip,)):
                rr=json.loads(v[0]);safe=json.loads(v[1]);n+=1;n_route+=1
                writers[0][2].writerow(rr);writers[1][2].writerow(safe)
                for k in MODEL_ATTRS:
                    if safe[k] is None:missing_feature[k]+=1
            if n!=d['natural_count']:
                raise RuntimeError(f'{trip}数据库自然候选数不一致：{n} vs {d["natural_count"]}')
        for p,f,w,tmp in writers:
            f.close();tmp.replace(p)
    finally:
        for p,f,w,tmp in writers:
            if not f.closed:f.close()
            if tmp.exists():tmp.unlink()
    stat=[]
    for date in sorted(day):
        x=day[date]
        stat.append({'date':date,'od_n':x['od'],'at_least_1':x['at_least_1'],
                     'at_least_3':x['at_least_3'],
                     'success_rate_1':x['at_least_1']/x['od'] if x['od'] else None,
                     'success_rate_3':x['at_least_3']/x['od'] if x['od'] else None})
    pd.DataFrame(stat).to_csv(out/'04_逐日自然候选成功率.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'natural_count':k,'od_n':v} for k,v in sorted(sizes.items())]).to_csv(
        out/'05_每OD候选数量分布.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'status':k,'od_n':v} for k,v in sorted(counts.items())]).to_csv(
        out/'06_候选生成状态统计.csv',index=False,encoding='utf-8-sig')
    elapsed_values=[float(x.get('elapsed_seconds') or 0) for x in audits.values()]
    sorted_elapsed=sorted(elapsed_values)
    p95=sorted_elapsed[max(0,math.ceil(len(sorted_elapsed)*.95)-1)] if sorted_elapsed else None
    nn=sum(v for k,v in sizes.items() if k>=1)
    nt=sum(v for k,v in sizes.items() if k>=3)
    summary={
        'version':'chapter4_module3_blind_natural_routes_v1.0',
        'creation_time':datetime.now().isoformat(),
        'source_config':config,'original_full_test_od':full_n,
        'selected_od':len(selected),'is_formal_full':config['mode']=='full',
        'blind_test_1plus_od':nn,'blind_test_3plus_od':nt,
        'success_1plus_rate':nn/len(selected),
        'success_3plus_rate':nt/len(selected),
        'all_generated_natural_candidates':n_route,
        'avg_candidates_per_od':n_route/len(selected),
        'candidate_count_distribution':dict(sorted(sizes.items())),
        'status_counts':dict(counts),'missing_candidate_feature_counts':dict(missing_feature),
        'p95_per_od_search_and_feature_seconds':p95,
        'elapsed_python_session_seconds':elapsed,
        'graph_stats':roadstats,
        'feature_fields':MODEL_ATTRS,
        'input_roles':{
            'pure_od':'start_time+origin_node+destination_node+coordinates+OD straight distance',
            'observed_real_route':'NOT_READ', 'chosen':'NOT_PRESENT',
            'VehicleID':'NOT_READ','module2_preference_profiles':'NOT_READ'},
        'limitations':[
            '所有评价只涉及固定GraphML上自然生成的候选；不等于全路网最优Top3。',
            '潜在机动车通行限制已作有限静态过滤，OSM缺失的法规/时间禁行及转向限制仍无法验证。',
            'ratio_to_shortest使用当前可搜有向路网首次无惩罚最短路径。',
            'Path Size仅在当OD实际生成的自然候选集合中计算。',
            '缺失的maxspeed/lanes/oneway均保留为缺失值，不假造数值。',
            '没有读取最后3天实际GPS路线，无法从本模块得出Hit@1、NLL或司机主观舒适度。',
            '试运行(pilot)不能用于第四章正式指标报告。',
        ],
        'outputs':{
            'natural_paths':str(p_raw),'blind_candidate_features':str(p_safe),
            'od_audit':str(p_od),'daily':str(out/'04_逐日自然候选成功率.csv')
        }}
    save_json(out/'07_模块三完整性_路网及数据隔离审计.json',summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        fonts={x.name for x in font_manager.fontManager.ttflist}
        for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','WenQuanYi Micro Hei'):
            if f in fonts:plt.rcParams['font.sans-serif']=[f];break
        plt.rcParams['axes.unicode_minus']=False
        fig,ax=plt.subplots(figsize=(7.3,4.5))
        ks=sorted(sizes)
        ax.bar(ks,[sizes[k] for k in ks],color='#377EA4',width=.75)
        ax.set_xlabel('每OD自然候选数量');ax.set_ylabel('OD数量')
        ax.set_title(f'最后3天自然候选数量分布（{config["mode"]}）')
        ax.set_xticks(ks);ax.grid(axis='y',alpha=.15)
        fig.tight_layout()
        fig.savefig(out/'图4-X_最后三天自然候选数量分布.png',dpi=360,bbox_inches='tight')
        fig.savefig(out/'图4-X_最后三天自然候选数量分布.svg',bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        info(f'绘图失败但数据与审计已安全输出：{type(exc).__name__}: {exc}')
        summary['plot_warning']=str(exc)
        save_json(out/'07_模块三完整性_路网及数据隔离审计.json',summary)
    info(f'输出完成：{len(selected):,}个测试OD，自然候选{n_route:,}条；至少1条={nn:,}，至少3条={nt:,}；'
         f'单OD P95计算时间={p95:.3f}s')
    return summary


def parser():
    p=argparse.ArgumentParser(description='第四章模块三：真实未知OD自然候选及PathSize（无真值无chosen）')
    p.add_argument('--od',type=Path,default=DEFAULT_OD,help='模块一纯OD白名单CSV')
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST,help='模块一来源审计JSON')
    p.add_argument('--graphml',type=Path,default=DEFAULT_GRAPH,help='曼谷正式有向GraphML')
    p.add_argument('--out',type=Path,default=None,help='输出目录；不同实验参数需不同目录')
    p.add_argument('--mode',choices=['pilot','full'],default='full',help='默认全部独立测试OD；pilot仅试运行')
    p.add_argument('--pilot-size',type=int,default=100,help='pilot选取的真实OD数量')
    p.add_argument('--pilot-seed',type=int,default=42)
    p.add_argument('--k',type=int,default=10,help='每个OD最多自然候选路径数量')
    p.add_argument('--max-attempts',type=int,default=0,help='搜索尝试次数，0=K*5，与原Step42默认一致')
    p.add_argument('--max-route-ratio',type=float,default=2.,help='候选全边长度不得超过OD最短路的倍数')
    p.add_argument('--penalty-start',type=float,default=.20)
    p.add_argument('--penalty-step',type=float,default=.07)
    p.add_argument('--max-expanded',type=int,default=800000,help='单次搜索最大扩展节点数，0不限制')
    p.add_argument('--min-straight-km',type=float,default=0.,help='纯OD筛选，仅依据出发前OD几何；默认全部保留')
    p.add_argument('--max-node-coordinate-error-m',type=float,default=50.,help='拒绝起终点节点不匹配的图版本')
    p.add_argument('--turn-angle',type=float,default=45.,help='和原Step43相同的默认转角阈值')
    p.add_argument('--vehicle-filter',choices=['exclude_nonmotor','legacy_all_length_edges'],
                   default='exclude_nonmotor',help='只禁止OSM明确非机动车及禁行；legacy按原Step42全部有效长度边')
    p.add_argument('--check-input-only',action='store_true',help='只检查模块一真实输入，不读取GraphML不产生结果')
    p.add_argument('--reset',action='store_true',help='明确重建当前SQLite结果；旧结果不可直接继续')
    return p


def main():
    args=parser().parse_args()
    if args.k<3:raise ValueError('本章需要评价Top-3，设置 --k 至少3')
    if args.max_attempts<0 or args.max_expanded<0:raise ValueError('搜索预算不能为负')
    if args.max_attempts==0:args.max_attempts=args.k*5
    if args.max_attempts<args.k:
        info('提醒：max-attempts小于K，最多只能产生max-attempts条自然候选')
    if args.pilot_size<1:raise ValueError('pilot-size必须>=1')
    if args.max_route_ratio<1:raise ValueError('最大绕行倍数不能小于1')
    if args.penalty_start<0 or args.penalty_step<0:raise ValueError('路径边惩罚不能为负')
    if not (0<=args.min_straight_km<1000):raise ValueError('OD几何阈值不合理')
    if args.max_node_coordinate_error_m<=0:raise ValueError('坐标容差必须为正')
    if not (0<args.turn_angle<=180):raise ValueError('转角阈值必须在(0,180]')
    nonempty_file(args.od,'最后3天纯OD文件')
    manifest=load_manifest(args.manifest,args.od)
    blind=read_blind_od(args.od,manifest)
    info(f'模块一隔离输入检查完成：{len(blind):,}条纯OD，9列严格白名单，时间均属于最后3天。')
    if args.check_input_only:
        info('本次仅执行输入检查，未读取路网、未搜索候选、未生成任何结果文件。')
        return
    nonempty_file(args.graphml,'正式曼谷GraphML')
    if args.mode=='pilot':
        pick=set(random.Random(args.pilot_seed).sample(range(len(blind)),min(args.pilot_size,len(blind))))
        chosen=[x for i,x in enumerate(blind) if i in pick]
    else:chosen=blind
    out=args.out or DEFAULT_OUT / (f'pilot_K{args.k}_N{len(chosen)}_seed{args.pilot_seed}' if args.mode=='pilot' else f'full_K{args.k}')
    out.mkdir(parents=True,exist_ok=True)
    info('核验输入与GraphML文件SHA256，避免断点续跑混用不同路网或源文件...')
    source={'od':filemeta(args.od,True),'manifest':filemeta(args.manifest,True),
            'graphml':filemeta(args.graphml,True)}
    config={'version':'chapter4_module3_blind_natural_routes_v1.0',
            'mode':args.mode,'k':args.k,'pilot_size':len(chosen) if args.mode=='pilot' else None,
            'pilot_seed':args.pilot_seed if args.mode=='pilot' else None,
            'max_attempts':args.max_attempts or args.k*5,'max_route_ratio':args.max_route_ratio,
            'penalty_start':args.penalty_start,'penalty_step':args.penalty_step,
            'max_expanded':args.max_expanded,'min_straight_km':args.min_straight_km,
            'turn_angle':args.turn_angle,'vehicle_filter':args.vehicle_filter,
            'max_node_coordinate_error_m':args.max_node_coordinate_error_m,
            'input_source':source,'selected_count':len(chosen)}
    db=sqlite_prepare(out/'00_逐OD自然候选断点续跑.sqlite3',config,args.reset)
    done={x[0] for x in db.execute('SELECT trip_id FROM od')}
    selected_ids={x['trip_id'] for x in chosen}
    if not done.issubset(selected_ids):raise RuntimeError('断点数据库中有当前批次以外的OD，停止')
    info(f'实验{args.mode}：计划{len(chosen):,}个OD，已完成{len(done):,}个；'
         f'每OD最多{args.k}条自然候选，最多{config["max_attempts"]}次有向搜索。')
    start=time.perf_counter()
    road=None
    if len(done)<len(chosen):
        road=RoadNetwork(args.graphml,args.vehicle_filter)
        save_json(out/'00_本次固定GraphML路网概况.json',road.stats)
        new=0;success=0
        for row in chosen:
            if row['trip_id'] in done:continue
            audit,raw,feats=one_od(road,row,args)
            sqlite_commit_od(db,audit,raw,feats)
            new+=1
            success+=int(bool(raw))
            if new%10==0 or len(done)+new==len(chosen):
                info(f'进度{len(done)+new:,}/{len(chosen):,}；本轮候选成功OD={success:,}；'
                     f'最近{row["trip_id"]}：{audit["status"]}，路线={len(raw)}，'
                     f'耗时={audit["elapsed_seconds"]:.3f}s')
    if road is None:
        p=out/'00_本次固定GraphML路网概况.json'
        nonempty_file(p,'正式路网概况JSON')
        graphstats=json.loads(p.read_text(encoding='utf-8-sig'))
    else:graphstats=road.stats
    final=export_results(db,out,chosen,config,graphstats,len(blind),time.perf_counter()-start)
    db.close()
    if args.mode=='pilot':
        info('注意：当前是PILOT真实样本试跑，不能当作4469条全量论文结论！正式运行 --mode full。')
    if final['blind_test_3plus_od']==0:
        info('警告：本批次没有OD生成>=3条不同路线，请检查路网版本/筛选及候选搜索规则。')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        info('操作被用户中断：已经完成的OD保存在SQLite，重复原命令会自动续跑。')
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
