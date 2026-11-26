# report/figures：报告用图

`main.tex` 用 `\figslot{文件名}{宽}{高}{说明}` 引用这里的图：文件存在就直接插入，不存在就画一个同样大小的占位框，所以草稿的页数接近最终版本。

## 计划中的图

| 文件名 | 在报告里的位置 | 宽度 | 内容 |
|---|---|---|---|
| `fig_nmse_test.pdf` | 图 1（通栏，`figure*`） | `PAGE_W` = 7.16 in，高约 1.9 in | 测试集 run 0057 上 NMSE 随导频 SNR 的变化，16 / 8 / 4 个导频各一栏；线性、样条、LMMSE、MLP 各变体；95% 分块 bootstrap 置信区间 |
| `fig_gain_test.pdf` | 图 2（单栏） | `COL_W` = 3.5 in，高约 2.0 in | 各学习方法相对 LMMSE 的配对增益（dB，正数更好），三个种子的均值和范围 |
| `fig_mismatch_test.pdf` | 图 3（单栏） | `COL_W` = 3.5 in，高约 1.8 in | 噪声功率失配：NMSE 损失随"假设 SNR − 真实 SNR"的变化 |

表 II（复杂度）和表 III（天线 × 频率联合设置）直接写在 `main.tex` 里，不需要图片文件。

## 规则

- 用 `src/style.py` 画图：`apply_style("paper")`，颜色、标记和图例名用 `COLORS` / `MARKERS` / `LABELS`，宽度用 `COL_W` / `PAGE_W`。图里的文字用英文。
- 存成 PDF（矢量图，`pdf.fonttype` 已设为 42）。
- 测试集的图只能来自冻结后那一次测试集运行的 `run.json`，不要用验证集的数字代替。
- 图里只放坐标轴、图例和必要的标注；数据来源、置信区间的算法等较长的说明写在 `main.tex` 的图注里。
