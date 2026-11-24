# Qwen embedding 替换实验：执行结果

已将 MiniLM 物品编码器替换为 OpenJev 同版本 Qwen3.5-2B，完整执行 embedding → RQ-Kmeans → 数据构建 → 两阶段 LoRA → 校准/重载 → 自回归推荐。

## 模型和配置

- 基础模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`，与 OpenJev 的本地基础权重一致。
- 仅使用冻结文本骨干，共 1,881,825,088 个参数；不加载推荐 LoRA，推理 dtype 为 bf16。用户此次指定 2B 编码器，取代此前 ≤1B 的约束。
- 文本保持 title + genres；attention-masked mean + L2；batch=128、max_length=128。池化以 float32 累加。
- embedding 从 3,883×384 变为 3,883×2048；这是更换模型的输出维度变化。
- RQ 仍为三层、每层255中心，KMeans n_init=5、seed=42。
- 原用户序列与物品文本逐字节一致；训练/校准/验证/测试数量分别为2,928,147 / 18,120 / 18,120 / 18,120个语义token决策。
- LoRA rank=8、候选分块=1、梯度累积=1、lr=2e-5、head_lr=1e-4、Brier权重=0.1、历史长度=20。
- 与上次实际实验相同：stage1为3步/seed42/参考内核；stage2追加3步/seed43/FLA，重置优化器；每阶段校准3条。训练及校准采样位置与MiniLM版本一致。
- 第一阶段从原始已发布OpenJev权重开始，第二阶段从本次Qwen语义ID checkpoint继续，没有混用旧MiniLM推荐适配器。

## 结果

| 指标 | MiniLM | Qwen3.5-2B |
|---|---:|---:|
| embedding维度 | 384 | 2048 |
| 唯一三元组 | 3786 | 3712 |
| 碰撞组数 | 93 | 161 |
| 涉及碰撞的物品数 | 190 | 332 |
| 单码最多物品数 | 4 | 3 |

Qwen本次碰撞更多；不能仅凭模型参数量判断推荐效果。不同维度的逐元素残差MSE不宜直接比较。

| 阶段 | 三步loss | checkpoint重载最大误差 |
|---|---|---:|
| stage1 | 6.223874, 6.273469, 5.121642 | 0.000000 |
| stage2 | 4.999646, 5.307935, 5.306763 | 0.000000 |

两阶段校准温度分别为 0.869938 和 20.000000。第二阶段温度达到官方拟合搜索上限20；只有3条校准样本，不能据此证明校准质量，配置按要求保持原样。

同一测试用户 `783` 的自回归推荐已完成：beam=3、top-k=3，Recall@3=0.0、NDCG@3=0.0。使用模型生成前缀，排除历史物品，输出均可映射到目录物品。

这仍是累计6步的流程验证，未运行1,000步或全量epoch；单用户评估不能说明推荐质量改善或退化。详见 `runs/qwen-embedding/verification.json`、各阶段 `training.jsonl` 和 `evaluation.json`。

## 文件与复现

- 当时的运行入口为 `bash run_qwen.sh`。该次实验保留了已有的训练输出。
- embedding：`data/ml1m_qwen_embeddings/embeddings.npy`。
- 同一份语义 ID 映射现存于 [`data/ml1m_qwen_sequences/item_semantic_ids.jsonl`](../data/ml1m_qwen_sequences/item_semantic_ids.jsonl)。
- 推荐训练数据：`data/ml1m_qwen_sequences`。
- 最终短程checkpoint：`runs/qwen-embedding/stage2/checkpoint`。
- 本次同条件评估使用stage1 checkpoint，以匹配原MiniLM实验。
- 检查：5项单元测试通过、官方1,200条schema样本校验通过、完整对照检查通过；物品embedding及码本均有限、归一化与取值范围正确。
