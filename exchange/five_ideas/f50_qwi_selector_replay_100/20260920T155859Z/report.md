# F50-QWI 100题 selector replay

该运行是固定 Top-50 -> Top-5 的 L0 诊断。F50-QWI 是六量子比特寄存器上的单粒子边缘 Born 排序，并由经典矩阵特征分解精确计算；它不是联合五段量子状态，也不构成量子优势或 L1/L2 结论。

- 输入 SHA-256：`669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731`
- Top-50 identity：`True`
- 数值/泄漏机制门：`True`
- 四臂总选择耗时：`2.098s` / 600s

## Silver 后验诊断

| 方法 | Silver overlap 均值 | broad 选中均值 | selector miss | 归一化 embedding 冗余 | 中位选择 ms |
|---|---:|---:|---:|---:|---:|
| qore_as | 2.590 | 1.770 | 15 | 0.778 | 0.000 |
| topk_as | 3.550 | 2.030 | 14 | 0.831 | 0.000 |
| born_exact | 1.350 | 1.200 | 23 | 0.745 | 5.026 |
| real_diffusion_control | 2.670 | 1.960 | 13 | 0.840 | 5.026 |
| dephased_control | 2.560 | 2.010 | 10 | 0.844 | 5.026 |
| phase_scramble_control | 1.970 | 1.710 | 18 | 0.819 | 5.026 |

## 判定

- Born 严格超过 Top-k 3.55/5：`False`。
- 下一层完整 selector replay 是否获准：`False`。
- Silver labels 只在所有选择完成后读取；13 个或更多 retrieval miss 不能归因于 selector。
- 即使门通过，结果仍是 L0 诊断，不能声明真实数据集 utility、量子优势、L1 或 L2。
