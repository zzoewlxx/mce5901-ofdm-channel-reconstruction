# MCE5901 小组项目：基于实测 CSI 的 OFDM 信道重建

**题目：** OFDM Channel Reconstruction from Sparse Noisy Pilots Using Measured CSI

**一句话：** 用 DICHASUS 的真实测量信道（dichasus-005x），研究如何从 64 个子载波中少量带噪声的导频（16、8 或 4 个）恢复其余子载波上的信道。比较四种方法：线性插值、三次样条插值、用训练数据估计协方差的 LMMSE，以及一个小型神经网络（MLP）。

> 原始数据和模型权重不进仓库，见下文"数据"。

## 进度

| 状态 | 任务 | 位置 |
|---|---|---|
| ✅ | 选题与可行性试跑（2026-11-06） | `experiments/00_pilot_2026-11-06/` |
| ✅ | 提案 v4（提交版）：v3 加上复核后的措辞修正 | `docs/proposal/proposal_v4.docx`（生成脚本在 `docs/proposal/build/`） |
| ✅ | 全量数据下载、抽取子载波块、筛掉坏快照 | `tools/`、`src/`，Release `data-v1` |
| ✅ | 全量数据基线：线性、样条、LMMSE（验证集） | `experiments/01_baselines_fulldata/` |
| ✅ | MLP 四种变体（验证集，随机种子 0） | `experiments/02_mlp_fulldata/` |
| ⬜ | MLP 多随机种子、噪声功率失配、可选误码率；最后在测试集上跑一次 | 见 `docs/next_steps.md` |
| ⬜ | 扩展：天线和频率联合重建（学习方法最有可能真正有用的地方） | 见 `docs/next_steps.md` 第 3 节 |
| ⬜ | 汇报 PPT（12 月 18 日）与最终报告 | `report/` |

具体任务和负责人在 GitHub Issues 里登记，用 Projects 看板跟踪进度。

## 全量数据基线结果（验证集 run 0056）

结果文件夹：`results/20261106-0713_auto_baselines-val/`，包括图和 `run.json`。

- **数据**：拟合用约 51 万条向量（run 0053–0055 × 32 根天线），筛掉了 0.42% 的坏向量。
- **LMMSE 比线性插值好多少**（导频 SNR 分别为 0 / 10 / 20 / 30 dB）：

  | 导频数（共 64 个子载波） | 0 dB | 10 dB | 20 dB | 30 dB |
  |---|---|---|---|---|
  | 16 | 9.7 dB | 8.3 dB | 7.0 dB | 5.9 dB |
  | 8 | 7.3 dB | 6.1 dB | 4.7 dB | 3.9 dB |
  | 4 | 5.0 dB | 3.7 dB | 2.4 dB | 1.7 dB |

  配对 bootstrap 的 95% 置信区间约为 ±0.1 dB。
- **三次样条**比线性插值差约 1–1.5 dB：样条在含噪导频之间振荡，放大了噪声。

## 全量数据 MLP 结果（验证集 run 0056）

结果文件夹：`results/20261110-1734_auto_mlp-val/`。这是一次验证实验：16 和 8 个导频、随机种子 0、各训练 15 轮；4 个导频和多随机种子还没跑。表中数字是"相对 LMMSE 好多少"（dB，正数表示更好）：

| 模型 | 0 dB | 10 dB | 20 dB | 30 dB |
|---|---|---|---|---|
| MLP 2×128（提案中的模型），16 / 8 个导频 | −0.01 / +0.01 | −0.03 / −0.02 | −0.15 / −0.10 | −1.48 / −1.00 |
| 按期望误差加权损失 | −0.22 / −0.11 | −0.13 / −0.07 | −0.15 / −0.09 | −0.73 / −0.41 |
| LMMSE + 残差 MLP | ≈0 | ≈0 | ≈0 | −0.002 |
| MLP 3×256（参数量 5 倍） | ≈ 2×128 | ≈ 2×128 | ≈ 2×128 | −1.92 / −1.14 |

**结论的准确说法：** 本次验证中，所测试的模型都没有显著超过 LMMSE，残差变体与它几乎持平。

**可能的解释：** 在 64 个子载波的块里，信道协方差的前 2 个特征值占了 99.99% 的能量，LMMSE 能充分利用这种频域相关性，所以基线很强。但 LMMSE 只是最优的线性估计器，这个观察不能证明非线性方法一定无法更好。

**还不能下的结论：** 加权损失改善了高 SNR、同时牺牲了一点低 SNR；大网络在相同训练轮数下没有改善。这还不足以排除网络容量、优化或训练轮数的影响。

**可选扩展（待验证的假设）：** 32 根天线 × 64 个子载波联合重建时，99% 能量对应的线性维数约 28，而主要的物理变量是发射机位置，非线性模型也许有空间。详见 `docs/next_steps.md`。

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
src/                  可复用代码：dichasus_reader.py（读取器）、estimators.py（导频布局、插值、LMMSE、NMSE）、runlog.py（保存结果）
tools/                download_dichasus.py（下载）、extract_block.py（抽取子载波块）、run_pipeline.py（一键全流程）
experiments/          每个阶段一个文件夹，脚本可以独立运行
  00_pilot_2026-11-06/  可行性试跑及复核
  01_baselines_fulldata/ 全量数据基线（验证集）
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

中断后重新运行会接着下载，每个文件下载完都会核对 DaRUS 提供的 MD5。

**不想下载 6.9 GB 的话**：仓库的 Releases 里有 `data-v1`，是处理好的 `dichasus005x_sc256-319.npz`（约 440 MB）：
- 内容：64 个子载波的块、全部 32 根天线、全部 5 个 run，已做偏移补偿，并带每个向量的质量指标；
- 用法：放到 `data/processed/` 下即可直接跑基线；
- 要换其他子载波块，仍需原始数据，用 `python tools/extract_block.py --start ... --width ...` 重新抽取。

一键全流程（下载、抽取、跑基线、提交结果）：`python tools/run_pipeline.py --member <你的名字>`

读取原始文件的方式：

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
