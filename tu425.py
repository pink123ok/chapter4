# -*- coding: utf-8 -*-
"""
图4-25：车辆级五折核心模型 Hit@1 对比
数据源：05_02_核心模型五折逐折指标.csv

说明：
1. 本图使用“逐折指标”文件，因此是“按 fold 展示的折线图”。
2. hit1 按要求四舍五入到小数点后4位。
3. 默认画6个核心模型；如果只想画两条线，可修改 ONLY_TWO_MODELS = True。
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =========================
# 1) 路径设置：按你的正式实验目录修改
# =========================
csv_path = Path(
    r"C:\study\轨迹数据预处理\曼谷GPS_正式清洗_202412\FULL_全量"
    r"\第四章_无泄漏路径选择实验\模块五至七_车辆五折与泛化消融"
    r"\full_safe_K10\formal_refit\05_02_核心模型五折逐折指标.csv"
)

out_dir = csv_path.parent / "论文图片_PNG600DPI_SVG"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2) 是否只画两个模型
# True  : 只画 MNL_L2 和 PSL_LengthWeighted_L2
# False : 画全部6个核心模型
# =========================
ONLY_TWO_MODELS = False

# =========================
# 3) 中文字体设置
# =========================
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]:
    if font_name in available_fonts:
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["svg.fonttype"] = "none"

# =========================
# 4) 读取数据
# =========================
if not csv_path.is_file():
    raise FileNotFoundError(f"未找到文件：{csv_path}")

df = pd.read_csv(csv_path, encoding="utf-8-sig")

required_cols = {"split", "model", "fold", "n_od", "hit1", "hit3", "mrr", "nll"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"文件缺少必要列：{sorted(missing)}")

# 只保留五折 OOF 结果（通常就是 oof）
df = df[df["split"].astype(str).str.lower() == "oof"].copy()

# hit1 四舍五入到小数点后4位
df["hit1"] = pd.to_numeric(df["hit1"], errors="coerce").round(4)
df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype(int)

# =========================
# 5) 模型顺序与显示名称
# =========================
model_display = {
    "MNL_L2": "MNL_L2",
    "PSL_Count_L2": "PSL_Count_L2",
    "PSL_LengthWeighted_L2": "PSL_LengthWeighted_L2",
    "UDPPM_Econ_PSL_L2": "UDPPM_Econ_PSL_L2",
    "UDPPM_Comf_PSL_L2": "UDPPM_Comf_PSL_L2",
    "UDPPM_Dual_PSL_L2": "UDPPM_Dual_PSL_L2",
}

if ONLY_TWO_MODELS:
    model_order = ["PSL_LengthWeighted_L2", "MNL_L2"]
else:
    model_order = [
        "MNL_L2",
        "PSL_Count_L2",
        "PSL_LengthWeighted_L2",
        "UDPPM_Econ_PSL_L2",
        "UDPPM_Comf_PSL_L2",
        "UDPPM_Dual_PSL_L2",
    ]

df = df[df["model"].isin(model_order)].copy()
if df.empty:
    raise ValueError("筛选后没有可用数据，请检查 model 名称或 split 列内容。")

# 校验每个模型是否都有5折
check = df.groupby("model")["fold"].nunique()
bad_models = check[check != 5]
if not bad_models.empty:
    print("警告：以下模型不是完整5折：")
    print(bad_models)

# =========================
# 6) 绘图
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

markers = ["o", "s", "^", "D", "v", "P"]
linestyles = ["-", "--", "-.", ":", "-", "--"]

all_y = []

for i, model in enumerate(model_order):
    sub = df[df["model"] == model].sort_values("fold").copy()
    if sub.empty:
        continue

    x = sub["fold"].to_numpy()
    y = sub["hit1"].to_numpy()
    all_y.extend(y.tolist())

    mean_val = round(sub["hit1"].mean(), 4)
    label = f"{model_display[model]} (Mean={mean_val:.4f})"

    ax.plot(
        x, y,
        marker=markers[i % len(markers)],
        linestyle=linestyles[i % len(linestyles)],
        linewidth=2,
        markersize=6,
        label=label
    )

    # 数值标注：贴近点，避免飘到上方
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.4f}",
            xy=(xi, yi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

# 坐标轴范围自动留边
ymin = min(all_y)
ymax = max(all_y)
pad = max((ymax - ymin) * 0.25, 0.002)

ax.set_xlim(0.8, 5.2)
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("Fold")
ax.set_ylabel("Hit@1")
ax.set_title("图4-25 车辆级五折核心模型 Hit@1 对比")

ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="best", fontsize=9, frameon=True)

fig.tight_layout()

png_path = out_dir / "图4-25_车辆级五折核心模型Hit1对比_逐折.png"
svg_path = out_dir / "图4-25_车辆级五折核心模型Hit1对比_逐折.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("绘图完成：")
print(png_path)
print(svg_path)