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
| ✅ | MLP 变体：3 个随机种子 × 16/8/4 个导频（验证集） | `experiments/02_mlp_fulldata/` |
| ✅ | 噪声功率（SNR）失配：真实 × 假设 SNR 网格，3 种导频数（验证集，2026-11-26） | `experiments/04_snr_mismatch/` |
| ✅ | 扩展：GMM 3 个种子、加宽网格（验证集，2026-11-26） | `experiments/06_gmm_seeds/` |
| ✅ | 扩展：天线 × 子载波联合 MLP，3 个种子（验证集，2026-11-26） | `experiments/07_joint_mlp/` |
| ✅ | 计算成本：精确乘加次数 + 空闲机器上固定核心的计时（2026-11-26） | `experiments/05_complexity/` |
| ✅ | 测试集 run 0057 上的唯一一次评估（2026-11-26）：先冻结方法清单，验证集演练完全复现后只运行一次 | `experiments/08_test_run/`，记录在 `results/20261126-2240_auto_test-0057/` |
| ✅ | 最终报告（5 页，IEEE 格式）与汇报幻灯片（10 页，12 月 18 日） | `report/`、`slides/` |

具体任务和负责人在 GitHub Issues 里登记，用 Projects 看板跟踪进度。

## 测试集结果（run 0057，只评估一次）

记录：`results/20261126-2240_auto_test-0057/`（开始标记在 `results/20261126-2203_auto_test-0057-started/`）。测试集有 5,001 个快照、159,400 条 64 子载波向量、4,807 个联合样本。判定规则在测试前冻结（`experiments/08_test_run/frozen.json`）：配对 95% 置信区间不含 0，并且提升至少 0.1 dB，才算"更好"。

| 事先定好的结论 | 测试集结果 |
|---|---|
| 64 子载波：没有任何 MLP 变体优于 LMMSE | **成立**。84 个组合（4 个变体 × 3 种导频数 × 7 个 SNR）里没有一个"更好"；35 个更差，其余 49 个的整个置信区间都在 ±0.1 dB 以内 |
| 联合设置：GMM 优于联合 LMMSE | **8/8 行成立**。32 根天线都有导频时 +0.17 到 +2.08 dB，只有 8 根天线有导频时 +2.95 到 +5.85 dB |
| 联合设置：GMM 优于联合 MLP | **8/8 行成立** |
| 联合设置：GMM 优于联合残差 MLP | **7/8 行成立**；8 天线 × 4 子载波、10 dB 那一行是 +0.05 [−0.03, +0.13] dB，按规则判为"不确定" |

其他要点（都是描述性的，没有事先设为结论）：

- LMMSE 比线性插值好 1.7–9.7 dB；导频数每减半，LMMSE 约损失 2.5 dB；三次样条在所有 SNR 下都比线性插值差 0.8–1.5 dB。
- 普通 MLP 在 15 dB 以下与 LMMSE 相差不到 0.07 dB，30 dB 时差 0.93–1.56 dB。
- SNR 假设错误：假设的 SNR 偏低 10 dB，LMMSE 损失 2.1–2.4 dB；偏高 10 dB 只损失 0.7 dB。不用 SNR 输入的 MLP 在 20 dB 以下与知道真实 SNR 的 LMMSE 相差不超过 0.29 dB（三个种子的平均）。
- 测试集上的数字与验证集奇数块的数字相差都不到 0.3 dB。

报告 `report/main.tex` 第七节的所有数字都来自这份记录，并且由独立的核对流程逐个对照过。

## 全量数据基线结果（验证集 run 0056）

结果文件夹：`results/20261106-0713_auto_baselines-val/`，包括图和 `run.json`。

- **数据**：拟合用 507,853 条向量（run 0053–0055 × 32 根天线），训练 run 中筛掉了 0.44% 的坏向量（0053 / 0054 / 0055 分别为 0.48% / 0.49% / 0.33%，验证 run 0056 为 0.40%）。以前写的 0.42% 把测试 run 也算进去了，不再使用。
- **LMMSE 比线性插值好多少**（导频 SNR 分别为 0 / 10 / 20 / 30 dB）：

  | 导频数（共 64 个子载波） | 0 dB | 10 dB | 20 dB | 30 dB |
  |---|---|---|---|---|
  | 16 | 9.7 dB | 8.3 dB | 7.0 dB | 5.9 dB |
  | 8 | 7.3 dB | 6.1 dB | 4.7 dB | 3.9 dB |
  | 4 | 5.0 dB | 3.7 dB | 2.4 dB | 1.7 dB |

  配对 bootstrap 的 95% 置信区间约为 ±0.1 dB。
- **三次样条**比线性插值差约 1–1.5 dB：样条在含噪导频之间振荡，放大了噪声。

## 全量数据 MLP 结果（验证集 run 0056）

三个随机种子（0、1、2），导频数 16 / 8 / 4，各训练 15 轮。结果文件夹：`results/20261126-1204_auto-seed{0,1,2}_mlp-val/`，同时保存了每个模型的权重（`data/runs/<同名文件夹>/`，不进 git，SHA-256 记在 run.json 里）。共有的三个变体与 11 月 11 日的运行（`results/20261111-0053_auto-seed*`）逐位相同。

表中数字是"相对 LMMSE 好多少"（dB，正数表示更好），为三个种子的均值，括号里是三个种子中的最差值到最好值：

| 导频数 | 模型 | 0 dB | 10 dB | 20 dB | 30 dB |
|---|---|---|---|---|---|
| 16 | MLP 2×128（提案中的模型） | −0.01 | −0.03 | −0.17 | −1.55（−1.65 到 −1.48） |
| 16 | 按期望误差加权损失 | −0.21 | −0.13 | −0.12 | −0.56（−0.73 到 −0.45） |
| 8 | MLP 2×128 | +0.01 | −0.02 | −0.11 | −0.92（−1.00 到 −0.84） |
| 8 | 按期望误差加权损失 | −0.13 | −0.07 | −0.09 | −0.40（−0.46 到 −0.34） |
| 4 | MLP 2×128 | +0.02 | −0.01 | −0.12 | −0.95（−1.13 到 −0.81） |
| 4 | 按期望误差加权损失 | −0.06 | −0.03 | −0.06 | −0.32（−0.38 到 −0.26） |
| 16 | 不输入 SNR 的 MLP（盲） | −0.02 | −0.07 | −0.21 | −1.56（−1.66 到 −1.49） |
| 8 | 不输入 SNR 的 MLP（盲） | −0.03 | −0.07 | −0.18 | −1.09（−1.11 到 −1.07） |
| 4 | 不输入 SNR 的 MLP（盲） | −0.11 | −0.12 | −0.28 | −1.64（−1.86 到 −1.51） |

- **LMMSE + 残差 MLP**：所有配置下与 LMMSE 相差不超过 0.007 dB。
- **种子间波动很小**：20 dB 及以下不超过 0.05 dB，25 dB 时约 0.1 dB，30 dB 时约 0.3 dB，结论不依赖随机种子。
- **没有任何一个配置显著优于 LMMSE**：252 个"种子 × 导频数 × 模型 × SNR"组合里，配对置信区间整体高于 0 的有 0 个，整体低于 0 的有 193 个。记录里的区间端点只保留三位小数，有 15 个上界被存成了 -0.0（实际略小于 0），按符号计入后是 193 个；以前写的 178 漏掉了这 15 个。
- **盲 MLP 只比带 SNR 输入的 MLP 差一点**：16 个导频时差 0.01–0.04 dB；4 个导频、30 dB 时差得最多，约 0.7 dB。说明 MLP 几乎没有利用 SNR 输入。
- 更早的一次运行（`results/20261110-1734_auto_mlp-val/`，种子 0）还测了 3×256 的大网络：30 dB 时比 2×128 更差（16 / 8 个导频分别为 −1.92 / −1.14 dB）。

**结论的准确说法：** 本次验证中，所测试的模型都没有显著超过 LMMSE，残差变体与它几乎持平。

**可能的解释：** 在 64 个子载波的块里，信道协方差的前 2 个特征值占了 99.99% 的能量，LMMSE 能充分利用这种频域相关性，所以基线很强。但 LMMSE 只是最优的线性估计器，这个观察不能证明非线性方法一定无法更好。

**还不能下的结论：** 加权损失改善了高 SNR、同时牺牲了一点低 SNR；大网络在相同训练轮数下没有改善。这还不足以排除网络容量、优化或训练轮数的影响。

**扩展的出发点（当时的假设，结果见下文"扩展结果"）：** 32 根天线 × 64 个子载波联合重建时，99% 能量对应的线性维数约 28，而主要的物理变量是发射机位置，非线性模型也许有空间。详见 `docs/next_steps.md`。

**扩展的可行性试跑（2026-11-19，验证集）：** 用文献中在实测数据上赢过 LMMSE 的 GMM 估计器做了检验。
- 在原题目（单天线、64 个子载波）上，GMM 与 LMMSE 只差 0.00–0.02 dB，和 MLP 的结论一致。
- 在 32 天线去噪和天线 × 频率联合重建中，GMM 在 0 dB 时分别好 2.8 dB 和 1.5 dB，置信区间不含 0。

详见 `experiments/03_gmm_feasibility/README.md`。

## SNR 失配结果（验证集 run 0056）

结果文件夹：`results/20261126-1625_auto_snr-mismatch-val/`。接收机假设的 SNR 和真实 SNR 各取 0–30 dB，LMMSE 和带 SNR 输入的 MLP 拿到同一个假设值。对角线（假设 = 真实）与上面实验 02 的 378 个数完全相同。

- **SNR 估低比估高危险得多。** LMMSE 假设的 SNR 偏低 30 dB 时平均损失 16.7–19.9 dB，偏高 30 dB 时只损失约 3 dB。偏差为 ±5 dB 时，所有方法的平均损失不超过 0.5 dB。
- **带 SNR 输入的 MLP 比 LMMSE 稍稳**（最坏损失少 1–4 dB），但同样依赖 SNR 输入。残差 MLP 和 LMMSE 一样（差别 < 0.005 dB）。
- **不输入 SNR 的盲 MLP 最稳**：与知道真实 SNR 的 LMMSE 相比，25 dB 及以下只差 0.02–0.63 dB，30 dB 时差 1.1–1.6 dB。与假设了错误 SNR 的 LMMSE 相比，126 个失配格点里有 113 个显著更好。

详见 `experiments/04_snr_mismatch/README.md`。

## 扩展结果：天线 × 子载波联合重建（验证集 run 0056）

512 维（32 根天线 × 16 个子载波）。两种导频图样：32 天线 × 4 子载波（128 个导频）和 8 天线 × 4 子载波（32 个导频）。表中是相对联合 LMMSE 的增益（dB），GMM 是 3 个 k-means 种子的均值，MLP 是 3 个种子的集成：

| 图样 | 方法 | 0 dB | 10 dB | 20 dB | 30 dB |
|---|---|---|---|---|---|
| 32 天线 × 4 子载波 | GMM（K=64 或 32） | +2.10 | +0.80 | +0.38 | +0.18 |
| 32 天线 × 4 子载波 | 联合 MLP | +0.36 | −0.18 | −3.56 | −11.06 |
| 32 天线 × 4 子载波 | 联合 LMMSE + 残差 MLP | +0.91 | +0.17 | −0.01 | −0.01 |
| 8 天线 × 4 子载波 | GMM（K=64） | +2.98 | +4.66 | +5.47 | +5.86 |
| 8 天线 × 4 子载波 | 联合 MLP | +2.25 | +4.37 | +4.66 | +4.59 |
| 8 天线 × 4 子载波 | 联合 LMMSE + 残差 MLP | +2.29 | +4.50 | +4.85 | +4.73 |

- 和单天线的主线相反，加入天线维度后**非线性方法明显优于 LMMSE**，导频越稀疏增益越大。GMM 在所有格点上的点估计都最高，它的每个种子的置信区间都不含 0。GMM 和 MLP 之差还没有配对置信区间，要在测试集评估时给出。
- 直接回归的联合 MLP 在导频充足、高 SNR 时有误差底（约 −21 到 −23 dB），加上 LMMSE 残差结构后这个问题消失。
- GMM 选中的 K 在网格边上，增益应看作下界。512 维时 EM 不会移动 k-means 的划分，GMM 实际上是"k-means 划分 + 每类一个收缩协方差"。
- 只有空间维（32 天线去噪，`results/20261126-1811_auto_gmm-seeds-space32/`）时，GMM 也好 +0.05 到 +6.5 dB。

详见 `experiments/06_gmm_seeds/README.md`、`experiments/07_joint_mlp/README.md`。

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
src/                  可复用代码：dichasus_reader.py（读取器）、dataset.py（数据划分、测试集保护、bootstrap）、estimators.py（导频布局、插值、LMMSE、NMSE）、
                      mlp_numpy.py（NumPy MLP，可保存权重）、neural.py（MLP 变体和读取权重）、runlog.py（保存结果和权重）、style.py（统一配色和图尺寸）
tools/                download_dichasus.py（下载）、extract_block.py（抽取子载波块）、run_pipeline.py（一键全流程）
experiments/          每个阶段一个文件夹，脚本可以独立运行
  00_pilot_2026-11-06/  可行性试跑及复核
  01_baselines_fulldata/ 全量数据基线（验证集）
  02_mlp_fulldata/       全量数据 MLP 变体（验证集）
  03_gmm_feasibility/    GMM 可行性试跑：单天线 / 多天线 / 天线 × 频率三种设置（验证集）
  04_snr_mismatch/       SNR 失配（验证集）
  05_complexity/         计算成本：乘加次数和实测耗时
  06_gmm_seeds/          GMM：3 个种子、加宽网格（验证集）
  07_joint_mlp/          天线 × 子载波联合 MLP（验证集）
  08_test_run/           测试集 run 0057 上的唯一一次评估（frozen.json + 演练 + 测试）
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
