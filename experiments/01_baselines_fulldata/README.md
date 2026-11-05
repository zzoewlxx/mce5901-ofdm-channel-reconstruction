# 01 全量数据基线（验证集）

```bash
python tools/extract_block.py                                              # 先生成 data/processed/dichasus005x_sc256-319.npz
python experiments/01_baselines_fulldata/run_baselines.py --member <名字>  # 结果保存到 results/
```

实验协议与提案一致：
- **筛选**：剔除 CSI 误差高于 −30 dB 的向量（误差在 `extract_block.py` 中用四阶差分估计）。
- **拟合**：LMMSE 的均值和协方差只用训练 run 0053–0055，32 根天线都作为样本。
- **验证**：run 0056 按每 50 个连续快照切块。偶数块用来选 LMMSE 的对角加载系数，奇数块用来报告结果，并给出 bootstrap 95% 置信区间。
- **测试**：run 0057 不在这里使用。
- **归一化**：所有数据只用一个全局常数缩放，导频 SNR 相对平均信道功率定义。
- **扫描范围**：SNR 取 0–30 dB，每 5 dB 一档；导频数取 16、8、4（共 64 个子载波）。

每次运行的图和完整记录（`run.json`）都保存在 `results/<时间>_<成员>_baselines-val/` 中，本文件夹不再单独保留图片。

在约 51 万条训练向量下，LMMSE 的对角加载系数（1e-6 到 1e-1）对结果几乎没有影响。
