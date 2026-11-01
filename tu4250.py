# -*- coding: utf-8 -*-
"""图4-25：从真实五折逐折结果生成两种科研图（无模拟数据）。

默认读取论文最新正式运行的：
  模块五至七_车辆五折与泛化消融/full_safe_K10/formal_refit/
  05_02_核心模型五折逐折指标.csv

使用示例（Windows PowerShell 或 CMD）：
  python "图4-25_高级科研绘图.py"
  python "图4-25_高级科研绘图.py" --input "D:\\...\\05_02_核心模型五折逐折指标.csv"

也支持已完成运行的“06_01_消融逐折性能.csv”提取A00-A05六个等价基线，
仅用于已核实的正式真实数据，不产生模拟/替代数值。

依赖：pip install pandas numpy matplotlib

说明：四位小数仅用于显示，平均值与标准差用原始完整精度计算；
标准差采用 pandas 默认 ddof=1，与 step457.py fold_summary 一致。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager
from matplotlib.patches import Rectangle

DEFAULT_IN = Path(
    r'C:\study\轨迹数据预处理\曼谷GPS_正式清洗_202412\FULL_全量'
    r'\第四章_无泄漏路径选择实验\模块五至七_车辆五折与泛化消融'
    r'\full_safe_K10\formal_refit\05_02_核心模型五折逐折指标.csv'
)
MODEL = [
    ('MNL_L2', 'MNL', 'A00_MNL_NoPS_NoHistory'),
    ('PSL_Count_L2', 'PSL-Count', 'A01_PSL_Count'),
    ('PSL_LengthWeighted_L2', 'PSL-Length', 'A02_PSL_Length'),
    ('UDPPM_Econ_PSL_L2', 'Econ + PSL', 'A03_Econ_WithPS'),
    ('UDPPM_Comf_PSL_L2', 'Comf + PSL', 'A04_Comf_WithPS'),
    ('UDPPM_Dual_PSL_L2', 'Dual + PSL', 'A05_Dual_WithPS'),
]
NAME_MAP = {short: full for full, _, short in MODEL}
NAME_LABEL = {full: lab for full, lab, _ in MODEL}
ORDER = [m[0] for m in MODEL]
COLOR = '#2C6584'
ACCENT = '#D47C43'
INK = '#1D2B38'
MUTED = '#687784'
GRID = '#E5EBEF'


def configure_fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ('Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'Source Han Sans SC', 'Noto Sans CJK JP'):
        if n in names:
            plt.rcParams['font.sans-serif'] = [n, 'DejaVu Sans']
            break
    plt.rcParams.update({
        'axes.unicode_minus': False, 'font.size': 11,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.edgecolor': '#C2CBD2', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
        'savefig.facecolor': 'white'
    })


def load_real_results(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            '没有找到真实实验文件：' + str(path) + '\n'
            '请先确定正式运行目录 formal_refit 或 formal_fixed，'
            '再通过 --input 指定正确的 05_02 CSV。'
        )
    d = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    need = {'split', 'model', 'fold', 'n_od', 'hit1'}
    missing = need - set(d.columns)
    if missing:
        raise ValueError('CSV缺少必要字段：' + ', '.join(sorted(missing)))
    d = d[d['split'].astype(str).str.strip().str.lower().eq('oof')].copy()
    d['model'] = d['model'].replace(NAME_MAP)
    d = d[d['model'].isin(ORDER)].copy()
    d['fold'] = pd.to_numeric(d['fold'], errors='raise').astype(int)
    d['hit1'] = pd.to_numeric(d['hit1'], errors='raise')
    d['n_od'] = pd.to_numeric(d['n_od'], errors='raise').astype(int)
    if len(d) != 30:
        raise ValueError(f'期望六模型 × 五折 = 30行，实际{len(d)}行；请确认完整正式实验文件。')
    if d.duplicated(['model','fold']).any():
        raise ValueError('model+fold组合存在重复记录。')
    for model, g in d.groupby('model'):
        if set(g['fold']) != {1, 2, 3, 4, 5}:
            raise ValueError(model + ' 不包含完整Fold 1~5。')
        if not np.isfinite(g['hit1']).all() or not g['hit1'].between(0,1).all():
            raise ValueError(model + ' 存在异常Hit@1值。')
        if g['n_od'].min() < 1:
            raise ValueError(model + ' 存在无效验证OD数量。')
    tot = d.groupby('model')['n_od'].sum()
    if tot.nunique() != 1:
        raise ValueError('不同模型的验证OD总数不一致；不能直接比较。')
    tab = d.pivot(index='model', columns='fold', values='hit1').reindex(ORDER)
    summ = d.groupby('model', sort=False)['hit1'].agg(['mean','std']).reindex(ORDER)
    n_od = int(tot.iloc[0])
    print('使用真实实验文件：', path)
    print('每个模型的验证OD总数：', n_od)
    print('均值与标准差（原始精度计算，显示4位）：')
    for model in ORDER:
        print(f'  {model:27s}  {summ.loc[model,"mean"]:.4f} ± {summ.loc[model,"std"]:.4f}')
    return tab, summ, n_od


def save_fig(fig, out:Path, stem):
    out.mkdir(parents=True,exist_ok=True)
    p1=out/(stem+'.png')
    p2=out/(stem+'.svg')
    fig.savefig(p1, dpi=500, bbox_inches='tight', facecolor='white')
    fig.savefig(p2, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('已输出：', p1, '\n         ', p2)


def heatmap_report(tab, summ, n_od, out):
    """方案A：5折矩阵 + 均值/SD，适合单张放入学位论文。"""
    data = tab.to_numpy(float)
    # 全部六模型统一色阶；使用有意义的绝对Hit@1，不按行单独归一化。
    cmap = LinearSegmentedColormap.from_list('academic',
          ['#EFF3F5','#D9E5E9','#A8CCD0','#5A9EA8','#23637A'])
    norm = Normalize(vmin=0.35, vmax=0.75)
    fig=plt.figure(figsize=(12.9,5.8), facecolor='white')
    ax=fig.add_axes([0.17, 0.22, 0.56, 0.62])
    im=ax.imshow(data,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(5),[f'Fold {i}' for i in range(1,6)],fontsize=11)
    ax.set_yticks(range(6),[NAME_LABEL[k] for k in ORDER],fontsize=11)
    ax.tick_params(axis='both',length=0,pad=9)
    ax.set_xticks(np.arange(-.5,5,.999999),minor=True)
    ax.set_yticks(np.arange(-.5,6,.999999),minor=True)
    ax.grid(which='minor',color='white',linewidth=3)
    ax.tick_params(which='minor',bottom=False,left=False)
    for s in ax.spines.values(): s.set_visible(False)
    for i in range(6):
        for j in range(5):
            val=data[i,j]
            c='white' if norm(val)>0.72 else INK
            ax.text(j,i,f'{val:.4f}',ha='center',va='center',weight='medium',fontsize=10.4,color=c)

    # 汇总区：精准呈现均值和样本标准差，不在拥挤位置强行叠加折线标签。
    right=fig.add_axes([0.775,0.22,0.21,0.62],sharey=ax)
    right.set_xlim(0,1);right.set_ylim(5.5,-0.5)
    right.axis('off')
    right.text(0.10,-0.83,'Mean',fontsize=11,weight='bold',color=INK,ha='center')
    right.text(0.72,-0.83,'SD',fontsize=11,weight='bold',color=INK,ha='center')
    for i,k in enumerate(ORDER):
        bold = k == 'UDPPM_Dual_PSL_L2'
        if bold: right.add_patch(Rectangle((-.05,i-.4),1.02,.8,facecolor='#F0F7F9',edgecolor='none',zorder=0))
        right.text(.10,i,f'{summ.loc[k,"mean"]:.4f}',ha='center',va='center',fontsize=12,
                   weight='bold' if bold else 'normal',color=COLOR if bold else INK)
        right.text(.72,i,f'{summ.loc[k,"std"]:.4f}',ha='center',va='center',fontsize=11,color=MUTED)

    fig.text(.17,.955,'车辆级五折交叉验证 · 六种核心模型 Hit@1',fontsize=18,weight='bold',ha='left')
    fig.text(.17,.905,'逐折结果及折间波动  |  数值保留四位小数',fontsize=11,color=MUTED,ha='left')
    cbax=fig.add_axes([.17,.115,.56,.018])
    cbar=fig.colorbar(im,cax=cbax,orientation='horizontal')
    cbar.set_ticks([.40,.50,.60,.70])
    cbar.ax.tick_params(labelsize=9,color=MUTED,length=2)
    cbar.outline.set_visible(False)
    fig.text(.17,.055,f'注：各模型验证OD总数均为 {n_od:,}；SD为5折样本标准差（ddof=1）。色阶对六模型保持一致。',
             fontsize=9.3,color=MUTED,ha='left')
    save_fig(fig,out,'图4-25_方案A_五折热力矩阵_均值标准差')


def forest_rank(tab,summ,n_od,out):
    """方案B：完整模型排名 + 高分段放大；每模型灰点为原始5折。"""
    fig=plt.figure(figsize=(12.8,6.0),facecolor='white')
    ax=fig.add_axes([.23,.21,.28,.61]); side=fig.add_axes([.535,.21,.15,.61]); zoom=fig.add_axes([.73,.21,.22,.61])
    side.set_xlim(0,1); side.set_ylim(5.5,-.5);side.axis('off')
    ys=np.arange(6)
    cols=['#929FA8','#607D8E','#3C8091','#4F8494','#4A8796','#C87645']
    jitter=np.array([-.12,-.06,0,.06,.12])
    for i,k in enumerate(ORDER):
        m,sd=summ.loc[k,['mean','std']]
        vals=tab.loc[k].to_numpy(float)
        ax.scatter(vals,i+jitter,s=21,alpha=.43,color=cols[i],zorder=2,linewidths=0)
        ax.errorbar([m],[i],xerr=[sd],fmt='o',markersize=8,capsize=4,
                    color=cols[i],elinewidth=2.4,capthick=1.7,zorder=4)
        side.text(.48,i,f'{m:.4f} ± {sd:.4f}',fontsize=10.6,ha='center',va='center',color=INK,
                  weight='bold' if i==5 else 'normal')
        if i>=2:
            zoom.scatter(vals,i+jitter,s=24,alpha=.42,color=cols[i],zorder=2,linewidths=0)
            zoom.errorbar([m],[i],xerr=[sd],fmt='o',markersize=8,capsize=4,
                          color=cols[i],elinewidth=2.4,capthick=1.7,zorder=4)
    ax.set_xlim(.36,.75);ax.set_ylim(5.5,-.5)
    ax.set_yticks(ys,[NAME_LABEL[m] for m in ORDER],fontsize=11)
    ax.set_xticks([.4,.5,.6,.7]);ax.set_xlabel('Hit@1',fontsize=11)
    ax.grid(axis='x',color=GRID,linewidth=.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0,pad=9)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ax.set_title('全部六模型',fontsize=12,weight='bold',pad=15,loc='left')
    side.text(.48,-.90,'Mean ± SD',fontsize=12,weight='bold',ha='center',color=INK)
    zoom.set_xlim(.685,.733);zoom.set_ylim(5.5,1.5)
    zoom.set_yticks([2,3,4,5],['PSL-Length','Econ + PSL','Comf + PSL','Dual + PSL'],fontsize=10)
    zoom.yaxis.tick_right();zoom.tick_params(axis='y',length=0,pad=8)
    zoom.set_xticks([.69,.70,.71,.72,.73]);zoom.tick_params(axis='x',labelsize=9)
    zoom.grid(axis='x',color=GRID,linewidth=.9);zoom.set_axisbelow(True)
    zoom.set_title('高性能模型局部放大',fontsize=12,weight='bold',pad=15,loc='left')
    for s in ['top','left','right']:zoom.spines[s].set_visible(False)
    fig.text(.06,.96,'车辆级五折交叉验证 · Hit@1模型性能分布',fontsize=18,weight='bold',ha='left')
    fig.text(.06,.90,'实心点：五折均值     水平误差棒：±1 SD     小点：单折结果',fontsize=11,ha='left',color=MUTED)
    fig.text(.06,.07,f'注：来源于五折真实逐折CSV；每模型验证OD总数 {n_od:,}；均值/SD基于未舍入的逐折结果计算。',
             fontsize=9.4,color=MUTED,ha='left')
    save_fig(fig,out,'图4-25_方案B_森林图_高分局部放大')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=DEFAULT_IN,help='真实05_02.csv的路径')
    parser.add_argument('--output',type=Path,default=None,help='输出文件夹；默认CSV同级下的论文图片目录')
    args=parser.parse_args()
    configure_fonts()
    tab, summ, n_od=load_real_results(args.input)
    out = args.output or args.input.parent/'论文图片_PNG600DPI_SVG'
    heatmap_report(tab,summ,n_od,out)
    forest_rank(tab,summ,n_od,out)

if __name__=='__main__': main()
