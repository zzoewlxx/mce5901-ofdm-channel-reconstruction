# 04 噪声功率（SNR）失配（验证集）

**要回答的问题：** 接收机假设的导频 SNR 和真实 SNR 不一致时，LMMSE 和实验 02 训练好的 MLP 各损失多少？不输入 SNR 的盲 MLP 是否更稳健？

**公平规则（小组约定）：**
- LMMSE 和带 SNR 输入的 MLP（`mlp`、`mlp_relweight`、`residual`）拿到**同一个**错误的 SNR。残差变体里的 LMMSE 基线也用这个假设 SNR。
- 每个假设 SNR 对应一套固定的设计（LMMSE 的矩阵和对角加载系数、网络权重）。真实噪声只改变输入，不改变设计。
- 盲 MLP（`mlp_blind`）不用 SNR，单独算作一种方法，不是对照组。
- 线性插值也不用 SNR，作为参考。

## 实验协议

与实验 01、02 相同：
- **数据**：拟合用 run 0053–0055，LMMSE 统计量用 `channel_stats(d.Hf)`，与实验 02 完全一致；全局只缩放一次，导频 SNR 相对平均信道功率定义。
- **验证**：run 0056 按每 50 个快照切块。偶数块只用来选 LMMSE 的对角加载系数，奇数块用来报告所有数字。
- **测试**：run 0057 不使用（`load_split` 不允许把它当作拟合或验证 run）。
- **网格**：每种导频数（16 / 8 / 4）都扫描真实 SNR {0, 5, …, 30} × 假设 SNR {0, 5, …, 30}，共 49 个格点。
- **噪声**：`Y = add_noise(Hv[:, p], 10**(-true/10), seed=1000*n_obs+true)`，与实验 01/02 是同一组噪声，所以"假设 = 真实"的对角线会复现实验 02 报告的数字（报告值保留三位小数，应完全一致）。脚本会自动核对（只在用 `--runs` 时），结果记在 `metrics.check_vs_experiment02`；最大差超过 0.0015 dB 时打印 WARNING。
- **LMMSE 的设计**：假设 SNR 为 a 时，用 `G = lm.matrix(10**(-a/10), rho_a)`。其中 `rho_a` 在偶数块上、噪声与 a 匹配的条件下选取，和 `run_baselines.py` 的做法完全相同。
- **模型**：实验 02 保存的权重。每个种子一个结果文件夹，读取时用 `runlog.load_artifacts` 核对 SHA-256。如果权重的 `scale`、子载波块或筛选阈值和当前数据不一致，脚本会直接报错（smoke 模式下只给提示）。
- **种子平均**：每个块的平方误差先在种子间取平均，再换算成 NMSE。MLP 相对 LMMSE（同一真实 SNR、同一假设 SNR）的增益用配对分块 bootstrap 给出 95% 置信区间（B = 1000，两种方法用同一组重采样块），同时给出每个种子增益的最小值和最大值。

## 汇总指标（`run.json` 的 `metrics`）

| 键 | 内容 |
|---|---|
| `grid` | 每种导频数、每种方法的 7×7 NMSE 网格，按 `[真实][假设]` 排列，包括每个种子的值、相对 LMMSE 的增益、置信区间和种子范围 |
| `matched` | 对角线（假设 SNR = 真实 SNR），可以和实验 02 对照 |
| `offset_loss_db` | 按偏移量 o = 假设 − 真实（−30 到 +30 dB）统计：在 t+o 仍落在网格内的真实 SNR t 上，对 NMSE(t, t+o) − NMSE(t, t) 取平均。参考点有两种：方法自己 SNR 匹配时的结果（`vs_own_matched`），以及 SNR 匹配时的 LMMSE（`vs_matched_lmmse`） |
| `worst_case_loss_db` | 整个网格上的最坏损失和出现的位置；另给只看偏差不超过 10 dB 时的最坏损失 |
| `fixed_design` | 按 10 dB 或 20 dB 设计的接收机，在各个真实 SNR 下的 NMSE；同时给出知道真实 SNR 的 LMMSE 作参考 |

图（保存到结果文件夹）：
- `offset_loss_val.png`：每种导频数一个子图。横轴是假设 SNR − 真实 SNR，纵轴是相对 SNR 匹配时的 NMSE 损失。曲线为 LMMSE、MLP、残差 MLP 和加权损失 MLP。盲 MLP 不随偏移量变化，放在第二张图里。
- `fixed_design_20db_val.png`：接收机都按 20 dB 设计，横轴是真实 SNR，纵轴是 NMSE。曲线为 LMMSE、MLP、残差 MLP 和盲 MLP；灰色虚线是知道真实 SNR 的 LMMSE，只作参考。

## 运行

先用 `experiments/02_mlp_fulldata/run_mlp.py` 训练并保存权重。`results/20261111-0053_auto-seed*` 这几次运行没有保存权重，不能直接用（脚本会报错"no saved weights"）。

数据文件不在本工作树的默认位置 `data/processed/` 时，用 `--data <npz 路径>` 指定。

```bash
# 代码测试：只用一个训练 run（前 70% 拟合，后 30% 验证），SNR 网格只取 0/10/20/30 dB，不写入 results/
python experiments/04_snr_mismatch/run_mismatch.py --member <名字> --smoke --data <npz 路径> \
    --runs <结果文件夹 seed0> <结果文件夹 seed1> <结果文件夹 seed2>

# 正式运行：每个种子一个实验 02 的结果文件夹
python experiments/04_snr_mismatch/run_mismatch.py --member <名字> --data <npz 路径> \
    --runs results/<时间>_<成员>-seed0_mlp-val results/<时间>_<成员>-seed1_mlp-val results/<时间>_<成员>-seed2_mlp-val
```

- 默认评估这些文件夹里的全部导频数和变体。`--densities 16 8`、`--variants mlp residual mlp_blind` 可以只评估其中一部分。注意 `run_mlp.py` 默认只训练 16 和 8 个导频，训练时加了 `--densities 16 8 4` 才会有 4 导频的结果。
- 如果实验 02 也训练了 `mlp_wide`（`run_mlp.py` 默认会训练），它会算进指标、但不会画进图里。它的单次评估约是 2×128 网络的 3 倍，三种导频数时约多 10 分钟。不需要时用 `--variants mlp mlp_relweight residual mlp_blind` 排除。
- `--runs` 也可以指向另一个工作树里的结果文件夹，权重会到那个工作树的 `data/runs/` 下去找，同样核对 SHA-256。
- 也可以用 `--weights-dir <文件夹> ...`（每个种子一个装着 `.npz` 权重的普通文件夹）代替 `--runs`，主要用于代码测试；这时不核对 SHA-256，也不和实验 02 的数字对照。
- smoke 模式下权重的 `scale` 必然和数据对不上，这是预期行为，脚本只打印提示；和实验 02 对照的差值也会很大，同样是预期的。

**运行时间（Intel Core Ultra 5 125H，`OPENBLAS_NUM_THREADS=4`）：**
- 正式运行约 15–20 分钟（三种导频数、四个变体、三个种子）：约 1400 次 MLP 评估、每次约 8.3 万个奇数块向量，每个向量约 7–8.5 µs（含误差统计）。这是按合成数据上测得的单向量耗时推算的。只有 16、8 两种导频数时约 11 分钟；加上 `mlp_wide` 时见上文。
- 内存峰值约 1.5 GB，出现在读数据时，与实验 02 相同。
- smoke 约 2–3 分钟。

## 结果

正式运行完成后填写（结果文件夹 `results/<时间>_<成员>_snr-mismatch-val/`）。
