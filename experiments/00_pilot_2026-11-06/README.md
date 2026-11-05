# OFDM 信道重建选题：可行性试跑（2026-11-06）

用 dichasus-005x 的一小部分真实数据，检查组员提案的设定是否有足够的研究空间。

- 拟合样本：run 0053 中均匀抽取的 150 个快照；评估样本：run 0056 中均匀抽取的 90 个快照；32 根天线合并作为样本。
- 测试 run 0057 没有使用，正式实验时可以作为未接触过的测试集。
- 结果只是试跑数字，不是最终结论。

## 文件

| 文件 | 作用 |
|---|---|
| `../../src/dichasus_reader.py` | 只依赖 numpy 的 TFRecord 读取器，不需要安装 TensorFlow；支持本地文件，也支持用 HTTP Range 只抽取部分记录 |
| `pilot_check.py` | 试跑实验：频域相关性、CSI 噪声水平，以及线性插值、三次样条和 LMMSE 的 NMSE |
| `make_figure.py` | 根据 `pilot_results.json` 画 `pilot_findings.png` |
| `pilot_results.json` / `pilot_findings.png` | 第一轮试跑的结果 |
| `rigor_check.py` → `rigor_results.json` | 复核：两种 SNR 定义、6 个子载波块、16/8/4 个导频、bootstrap 置信区间、32 根天线间的差异、训练量、空间迁移，以及 numpy 实现的 MLP |
| `screen_check.py` → `screen_results.json` | 坏快照筛选：用四阶差分估计每个向量的 CSI 误差 |
| `mlp_variants.py` → `mlp_variants_results.json` | 固定噪声功率与逐样本归一化的对比；普通、残差、按单个 SNR 训练三种 MLP |
| `mismatch_check.py` → `mismatch_results.json` | 噪声功率未知时：按名义 SNR 设计的 LMMSE 与盲 MLP 的对比 |

## 复核后的结论（已写进 `docs/proposal/proposal_v2.docx`）

- **无噪声导频的任务太简单。** 线性插值就能达到约 −43 dB，和测量误差在同一水平，所以必须给导频加噪声。
- **加噪声后 LMMSE 明显更好。** 用 8 或 16 个导频时，LMMSE 比线性插值好约 4–10 dB（0–30 dB 导频 SNR）。
- **必须筛掉坏快照。** 少数快照的 CSI 完全是噪声，`snr` 字段识别不出来，要直接从 CSI 估计误差来筛选。不筛的话，LMMSE 会出现"训练数据越多越差"的假象。
- **不要逐样本用真实功率归一化。** 这样会把信道范数泄露给模型，让 MLP 显得比 LMMSE 好 1–2 dB。改用一个全局常数缩放后，这个优势消失。
- **MLP 的表现。** 在固定噪声功率下，MLP 在 0–20 dB 与 LMMSE 相差不到 1 dB，但 30 dB 时差约 4 dB，存在误差底。"LMMSE + 残差 MLP"与 LMMSE 持平。
- **不必研究的方向。** 空间迁移和跨 run 都几乎没有分布差异；LMMSE 用约 100 个样本就已收敛。

## 运行

在本文件夹内运行。所有脚本共用同一个样本缓存目录，首次运行会通过 HTTP Range 下载约 63 MB 的样本：

```bash
python -I pilot_check.py    --cache <样本缓存目录> --out .
python -I make_figure.py    pilot_results.json pilot_findings.png
python -I rigor_check.py    --cache <样本缓存目录> --out .
python -I screen_check.py   --cache <样本缓存目录> --out .
python -I mlp_variants.py   --cache <样本缓存目录> --out .
python -I mismatch_check.py --cache <样本缓存目录> --out .
```

## 数据注意事项

- 原始 CSI 按 FFT 顺序存储，需要先 `fftshift`；频带两端有 −10~−15 dB 的滚降，512 处 DC 有轻微凹陷。选择子载波块时应避开这些位置，例如只在 128–896 范围内取块，并跳过 512。
- 记录中的实际字段是 `csi`、`snr`、`pos-lidar`、`time`、`rot-lidar` 和 `cfo`。spec.json 写的是 `rot`，与实际字段名不一致。
- `snr` 字段的中位数约为 19.5 dB，但 CSI 本身的误差中位数约为 −43 dB（相对信道功率，四阶差分估计）。所以不能用 `snr` 字段代表 CSI 误差，也不能用它来筛坏快照。
- 每根天线有固定的相位和定时偏移，官方提供了 `reftx-offsets-dichasus-005x.json`（`tools/download_dichasus.py` 会一并下载）。补偿方法见官方教程 "Compensating for the Reference Transmitter Channel"：https://dichasus.inue.uni-stuttgart.de/tutorials/
