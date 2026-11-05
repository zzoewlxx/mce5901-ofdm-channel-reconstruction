# MCE5901 小组项目：基于实测 CSI 的 OFDM 信道重建

**题目：** OFDM Channel Reconstruction from Sparse Noisy Pilots Using Measured CSI

**一句话：** 用 DICHASUS 的真实测量信道（dichasus-005x），研究如何从 64 个子载波中少量带噪声的导频（16、8 或 4 个）恢复其余子载波上的信道。比较四种方法：线性插值、三次样条插值、用训练数据估计协方差的 LMMSE，以及一个小型神经网络（MLP）。

> 原始数据和模型权重不进仓库，见下文"数据"。

## 进度

| 状态 | 任务 | 位置 |
|---|---|---|
| ✅ | 选题与可行性试跑（2026-11-06） | `experiments/00_pilot_2026-11-06/` |
| ✅ | 提案 v2（11 月 8 日截止） | `docs/proposal/proposal_v2.docx` |
| ⬜ | 全量数据下载、抽取子载波块、筛掉坏快照 | `tools/`、`src/` |
| ⬜ | 全量数据上的基线：线性、样条、LMMSE | |
| ⬜ | MLP 及残差变体（只学 LMMSE 的残差） | |
| ⬜ | 评估：SNR × 导频密度、bootstrap 置信区间、噪声功率失配，可选误码率 | `results/` |
| ⬜ | 汇报 PPT（12 月 18 日）与最终报告 | `report/` |

具体任务和负责人在 GitHub Issues 里登记，用 Projects 看板跟踪进度。

## 试跑得到的关键结论

详见 `experiments/00_pilot_2026-11-06/README.md`。

- **无噪声导频的任务太简单。** 信道在频率上非常平滑，64 个子载波两端的相关系数约 0.97。线性插值就能到约 −43 dB，和测量误差在同一水平，因此研究对象改成**含噪导频**。
- **加噪声后各方法明显拉开。** 用 8 或 16 个导频时，LMMSE 比线性插值好约 4–10 dB（导频 SNR 0–30 dB）。
- **评估中两个容易出假象的地方：**
  1. 少数快照的 CSI 完全是噪声，必须直接从 CSI 估计误差来筛掉，数据里的 `snr` 字段识别不出来。
  2. 不能每个快照用自己的真实功率归一化。这会把信道功率泄露给模型，让 MLP 显得好 1–2 dB。
- **公平设置下 LMMSE 很强。**
  - MLP 在 0–20 dB 与 LMMSE 相差不到 1 dB，但 30 dB 时差约 4 dB，有误差底。
  - "LMMSE + 残差 MLP"与 LMMSE 持平。

## 目录结构

```
src/                  可复用代码：dichasus_reader.py（不依赖 TensorFlow 的读取器）、runlog.py（保存结果）
tools/                download_dichasus.py（下载数据，支持断点续传和 MD5 校验）
experiments/          每个阶段一个文件夹，脚本可以独立运行
  00_pilot_2026-11-06/  可行性试跑及复核
results/              每次正式运行的结果（小文件），规则见 results/README.md
docs/proposal/        提案（v1 是组员原稿，v2 是提交版）
report/               最终报告（IEEE 会议模板，main.tex + reference.bib）
data/                 本地数据（不进 git）
external/             第三方参考代码（不进 git）
```

## 环境

- Python 3.12：`pip install -r requirements.txt`
- 只用 CPU，不需要独立显卡。后续如果用 PyTorch，装 CPU 版即可。

## 数据

```bash
python tools/download_dichasus.py                    # 全部 5 个 run，6.87 GB，存到 data/raw/dichasus-005x/
python tools/download_dichasus.py --runs 0053 0056   # 只下载部分 run
python tools/download_dichasus.py --dest D:/data     # 存到其他位置
```

中断后重新运行会接着下载，每个文件下载完都会核对 DaRUS 提供的 MD5。读取方式：

```python
import sys; sys.path.insert(0, "src")
from dichasus_reader import iter_file
for rec in iter_file("data/raw/dichasus-005x/dichasus-0053.tfrecords"):
    csi = rec["csi"]   # complex64 (32, 1024)，已经 fftshift：DC 在 512，频带边缘在 0 和 1023
```

数据划分固定为：
- run 0053–0055：训练
- run 0056：验证（调参、早停）
- run 0057：测试（只在最终评估用）

所有预处理参数和协方差都只用训练数据拟合。

## 结果怎么存

- 每次正式实验用 `src/runlog.py` 的 `save_run(...)` 保存到 `results/<时间>_<成员>_<实验名>/`。
- 自动记录实验配置、指标、git commit 和运行命令，所以**先提交代码再跑正式实验**。
- 只提交 JSON、CSV、PNG 等小文件，单个不超过 5 MB。

详细规则见 `results/README.md`。

## 协作流程

1. `main` 分支始终保持能运行。每个人在自己的分支上开发，例如 `feat/lmmse-full-data`，完成后提 Pull Request，至少一位组员看过再合并。
2. 开工前 `git pull`，提交信息写清楚做了什么，例如 "Add spline baseline and NMSE-vs-SNR plot"。
3. 任务、问题、讨论结论都记在 Issues 里，每周看一次 Projects 看板。
4. 任何结论都要能从 `results/` 里的 `run.json` 追溯到对应的代码版本。

## 分工（待小组确认）

| 角色 | 负责内容 | 汇报时讲的部分 |
|---|---|---|
| 数据与信道分析 | 下载和读取数据；fftshift 与偏移补偿；选择子载波块；筛坏快照；分析信道平滑度和相干带宽 | 数据集与问题建模 |
| 传统估计器 | 线性插值、样条插值、LMMSE（协方差估计与正则化）、计算复杂度 | 基线方法，以及 LMMSE 与岭回归的关系 |
| 学习模型 | MLP 和残差变体的设计、训练、多随机种子 | 学习方法 |
| 评估与鲁棒性 | SNR × 导频密度扫描、置信区间、噪声功率失配、可选误码率；整合报告 | 实验结果与结论 |

## 时间线

| 时间 | 目标 |
|---|---|
| 11 月 8 日 | 提交提案 |
| 11 月 15 日前 | 数据管线完成；全量数据上的线性、样条、LMMSE 结果 |
| 11 月 29 日前 | MLP 和残差变体；主实验跑完 |
| 12 月 6 日前 | 鲁棒性实验；图表定稿 |
| 12 月 13 日前 | PPT 初稿，组内彩排 |
| 12 月 18 日 | 课堂汇报 |
| 待确认 | 最终报告截止 |

## 数据集与引用

- F. Euchner, M. Gauger, *CSI Dataset dichasus-005x: Indoor Non-Line of Sight, Single Room*, DaRUS, 2021, CC BY 4.0, doi:10.18419/darus-2204。官网：https://dichasus.inue.uni-stuttgart.de/
- F. Euchner, M. Gauger, S. Dörner, S. ten Brink, "A Distributed Massive MIMO Channel Sounder for 'Big CSI Data'-driven Machine Learning," WSA 2021.
- 官方教程（偏移补偿、由上行 CSI 预测下行 CSI 等）：https://dichasus.inue.uni-stuttgart.de/tutorials/

## 外部参考代码（不进 git）

`external/CeBed`：一个基于 TensorFlow 的 OFDM 信道估计测试平台，实现了 ReEsNet 等深度学习估计器，可以作为文献综述和"考虑过的其他方法"的参考。注意它的 "LMMSE" 用的是每个样本的真实信道，不是实际可用的基线；许可证是 CC BY-NC 4.0。获取方式：

```bash
git clone --depth 1 https://github.com/SAIC-MONTREAL/CeBed.git external/CeBed
```
