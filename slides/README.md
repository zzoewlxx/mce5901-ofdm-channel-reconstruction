# slides：12 月 18 日汇报用的幻灯片

10 页，10 分钟汇报 + 2 分钟问答。在线版本（可演示、可导出 PPTX / PDF）是一个私有的 Claude Slides 文档；这里保存它的源文件，方便以后修改或重建。

| 内容 | 位置 |
|---|---|
| 每页幻灯片的 HTML（含讲稿，在每页末尾的 `<aside>` 里） | `deck/slides/*.html`，顺序见 `deck/deck.json` 的 `order` |
| 共用样式（颜色、字号、版式） | `deck/deck.css` |
| 幻灯片里的四张图 | `fig/*.png` |
| 画图脚本 | `make_slide_figures.py` |

## 重新画图

```
python -I slides/make_slide_figures.py . results/20261126-2240_auto_test-0057 slides/fig
```

图里的所有数字都来自测试集 run 0057 的那一次评估记录（脚本会检查记录确实是 run 57 的完整评估），配色和图例名沿用 `src/style.py`。

## 页面顺序

1. 封面
2. 研究问题
3. 数据：DICHASUS dichasus-005x 和数据划分
4. 评估协议（全局缩放、分块选择与报告、配对 bootstrap、先冻结再测试）
5. 比较的方法
6. 结果 1：LMMSE 比线性插值好 1.7–9.7 dB
7. 结果 2：84 个设置里没有一个 MLP 变体优于 LMMSE
8. 结果 3：SNR 假设错误的代价；不需要 SNR 的 MLP
9. 结果 4：天线 × 子载波联合重建，GMM 好 2.9–5.9 dB（8 根天线有导频）
10. 结论与局限

幻灯片里的数字和报告 `report/main.tex` 第七节一致；改动任何一处时请两边一起改。
