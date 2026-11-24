# 第二阶段：GSPO + 正样本 NLL

正式对比从 SFT 到全量验证、选模和测试，使用 [OpenJev 单卡运行步骤](../../comparison/OPENJEV_MINIONEREC.md)。三层文本 SID 的旧实验可以通过现行 Python 入口复现，前提是具备原始 SFT checkpoint、SID 和奖励向量。

```bash
# 在 OpenJev 目录运行。需要调整步数时使用新的输出目录。
HF_HUB_OFFLINE=1 python -B -u -m pipeline.train_infer.train_lora \
  --objective gspo --sid-encoding text --data data/ml1m_qwen_sequences \
  --checkpoint runs/qwen-embedding/stage1/checkpoint --base-model models/qwen3.5-2b \
  --reward-embeddings data/ml1m_qwen_embeddings --output runs/qwen-embedding/stage2-gspo \
  --steps 3 --seed 43 --candidate-batch 1 --calibration-rows 3
```

上述文本 SID 实验只更新 LoRA（rank=8）和 decision head。`--steps` 控制优化器更新次数，`--objective gspo` 选择第二阶段目标。

## 与论文的关系

[GenRec §2.3，式 6–8](https://arxiv.org/html/2604.14878v1) 提出的是 **GRPO-SR**：token 级策略比值、单步更新，以及正样本 NLL 正则项。它没有使用 GSPO。这里按本次需求，使用 [GSPO §4.1，式 5–7](https://arxiv.org/html/2507.18071v2) 的序列级比值与裁剪，并借鉴 GenRec 的 NLL 稳定项、奖励门控和组内正样本最大值校准。这是面向当前工程的组合实现，不是 GenRec 工业实验复现。

同一训练历史采样 G 条完整轨迹，每条轨迹是一个物品的三层 SID。令 q 为目录约束后重新归一化的策略：

\[
A_i=\frac{r_i-\mathrm{mean}(r)}{\max(\mathrm{std}_{population}(r),10^{-8})},\qquad
s_i=\exp\left[\frac{1}{3}\sum_{t=1}^{3}(\log q_\theta(y_{i,t}|h,y_{i,<t})-\log q_{old}(y_{i,t}|h,y_{i,<t}))\right].
\]

\[
L=-\frac{1}{G}\sum_i\min(s_i A_i,\mathrm{clip}(s_i,1-\epsilon_l,1+\epsilon_h)A_i)
\; -\alpha\sum_{t=1}^{3}\log q_\theta(y_t^+|h,y_{<t}^+).
\]

裁剪发生在整条序列上，不是逐 token PPO。旧策略概率、奖励、优势全部 detach；同一 rollout 复用更新时不重新生成分母。正样本仅用于 NLL 和已采样命中的奖励校准，不强行加入采样组。组内奖励相同则 GSPO 梯度为零，NLL 仍可提供学习信号。第一步的 GSPO 标量损失通常接近零，但其梯度可以非零；日志同时记录 `policy_logit_grad_norm`。

GSPO 模式不叠加旧 SFT 的 Brier 项或 KL 项；`--brier-weight` 只用于 SFT。NLL 按三个 token **求和**，不是取平均。

## 本地奖励的定义和限制

当前工程没有 GenRec 的 SIM 偏好模型或点击/订单页面数据，默认用已有的冻结 Qwen 文本 embedding 构造可运行的替代奖励：

1. 只取当前训练位置之前、最多 20 个历史物品的 embedding，平均后归一化为兴趣向量。
2. 物品与兴趣向量的余弦相似度为 c，偏好分数为 `(1+c)/2`；只在 `c > reward_threshold` 且物品不在该历史窗口中时计分，否则为 0。
3. 对发生 SID 碰撞的物品取成员奖励平均值。采样码与当前训练目标的码相同，则将奖励提升到该组最大值，参照 GenRec 式 7。这是**语义码命中**，不等价于精确 item 命中。

所有评分仍按原数据设置视为交互，未额外把低评分变成负反馈。奖励只读取训练历史和当前训练目标；校准集仅在训练结束后拟合温度，验证/测试标签不进入 RL。物品文本 embedding 和目录按已有 transductive 设置可见，未引入留出交互。`EmbeddingReward.__call__(codes, history, target)` 是替换成真正偏好奖励模型的接口。

相似度只能提供内容相关性信号，不能证明用户偏好或推荐指标提升；正式实验需训练更合适的奖励模型，并单独评估 Recall/NDCG。

## 采样、推理和显存

255³ 有约 1658 万种组合，现有目录只有 3712 个不同三元组。直接在全组合空间采样再对非法码给零奖励，会产生大量无信息组。因此本实现用目录前缀掩码，在每个前缀上精确归一化所有合法动作，不进行 top-k 或负采样。动作标签仍为 `sid_000`…`sid_254`，没有修改 RQ 码本或 tokenizer。

每组独立有放回 multinomial 采样，温度为 1。目录策略包括历史物品；奖励对历史窗口内物品计零。推理继续排除全部已交互物品，但分数归一化使用完整目录支持集，与训练策略保持一致。`recommendation.json` 中的 `policy_normalization=catalog` 自动启用此行为；旧 SFT checkpoint 继续使用原来的 full-255 归一化。GSPO checkpoint 的校准也在对应目录支持集上拟合。

OpenJev 对每个候选独立打分。代码只计算合法支持集的 logits，等价于先计算全部 255 个再对非法动作置 `-inf`。共享前缀先合并 `dL/dlogits`，再分块重算并反传到 LoRA，避免保留 G×3×255 个模型计算图。两遍计算期间关闭 dropout，参数更新在整组反传完成之后进行。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--group-size` | 4 | 每个历史采样的完整三层轨迹数，至少 2 |
| `--rollout-updates` | 2 | 复用固定 rollout 的优化器更新次数 |
| `--clip-low / --clip-high` | 0.0003 / 0.0004 | 取自 GSPO §5.1 的实验设置，需要在推荐任务上调参 |
| `--nll-weight` | 0.1 | 本地工程默认值；设 0 为纯 GSPO |
| `--reward-threshold` | 0.0 | 本地余弦相关性门槛 |
| `--steps` | 1000 | 本页短程示例显式设置为 3 |
| `--lr / --head-lr` | 2e-5 / 1e-4 | 沿用上一阶段 |

组大小、更新次数、NLL 权重和奖励门槛为本工程选择，不声称是 GenRec 论文超参数。历史长度、基础模型、数据、SID、LoRA 配置保持原设置。

```bash
# 独立评估第二阶段 GSPO checkpoint；此命令不参与训练
HF_HUB_OFFLINE=1 python -B -u -m pipeline.train_infer.recommend --data data/ml1m_qwen_sequences \
  --checkpoint runs/qwen-embedding/stage2-gspo/checkpoint \
  --test-users 100 --beam-size 10 --top-k 10 \
  --output runs/qwen-embedding/gspo-evaluation.json
```

实际短程验证结果见 `GSPO_RUN_REPORT.md`。
