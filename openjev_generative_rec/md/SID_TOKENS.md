# 分层 SID token 与可训练 embedding

新版在 OpenJev 输入 embedding 层加入专用 SID 表示，继续使用原来的候选独立评分结构、LoRA 和标量 decision head。没有改成 255 类线性分类头，也没有把 SID 注入中间网络层。

当前默认还包含[用户历史文字记忆](HISTORY_MEMORY.md)：`prepare_record()` 先按真实 item ID 加入与 SID 历史对齐的标题和类型，再转换分层 SID。描述通过原词表 embedding 输入；可训练参数数量不变。

## 表示和参数

- 第一层 `<sid1_000>`…`<sid1_254>`，第二层 `<sid2_000>`…`<sid2_254>`，第三层 `<sid3_000>`…`<sid3_254>`：共 765 个独立 token。
- 完整历史物品：`<|sid_item|><sid1_021><sid2_061><sid3_038><|sid_end|>`。
- 当前目标前缀：`<|sid_item|>` 后接已生成的分层 token，不加入尚未知的后续 SID。
- 候选只包含当前层 token，数量最多 255，原有 one-hot 标签和候选顺序保持一致。

底层数据和 RQ 映射仍保存原来的 `sid_XXX`，在 `LayeredSIDMixin.forward()` 中统一转换。这样 SFT、GSPO 的 rollout/重算、校准和 recommend 都使用同一种模型输入，数据划分、码本、碰撞处理及奖励映射无需重建。JSONL 的 schema 样本仍为原始数据格式，模型看到的是转换后的格式。

训练参数现在是 **LoRA + decision head + SID 小表**。新增 767 行（含两个物品边界），当前隐藏维度 2048，因此小表为 1,570,816 个 FP32 参数。加上原有 2,508,801 个 LoRA/head 参数，总计 4,079,617 个可训练参数。小表初值来自原词表中 `layer N sid_XXX` 的文本 embedding 均值，包含层号，不直接使用 RQ 残差中心。

`SIDEmbedding` 把普通 token 路由到冻结的原词表，把 SID 和边界 token 路由到独立小表。不为整个大词表创建梯度或 AdamW 状态；原词表不会被 weight decay 修改。小表保留 FP32 参数，进入骨干时转为其计算 dtype。原基础模型文件不改写；Qwen 原词表可能存在 padding 行，新增 token ID 与这些行重叠的情况已显式处理。

## 运行

正式对比使用 [OpenJev 单卡运行步骤](../../comparison/OPENJEV_MINIONEREC.md)。下面的分阶段命令在 OpenJev 目录运行，使用已配置的 Python 3.11 环境。 [服务器安装步骤](../../comparison/README.md#运行环境)

```bash
HF_HUB_OFFLINE=1 python -B -u -m pipeline.train_infer.train_lora \
  --objective sft --sid-encoding layered --data data/ml1m_qwen_sequences \
  --checkpoint models/openjev-2b/package/checkpoint --base-model models/qwen3.5-2b \
  --output runs/sid-sft --steps 1000 --sid-lr 1e-4
```

第二阶段从已经训练过 SID embedding 的 SFT checkpoint 出发。

```bash
HF_HUB_OFFLINE=1 python -B -u -m pipeline.train_infer.train_lora \
  --objective gspo --data data/ml1m_qwen_sequences \
  --checkpoint runs/sid-sft/checkpoint --base-model models/qwen3.5-2b \
  --reward-embeddings data/ml1m_qwen_embeddings --output runs/sid-gspo --steps 500

HF_HUB_OFFLINE=1 python -B -u -m pipeline.train_infer.recommend --data data/ml1m_qwen_sequences \
  --checkpoint runs/sid-gspo/checkpoint --base-model models/qwen3.5-2b \
  --beam-size 4 --top-k 4 --test-users 10 --output runs/sid-evaluation.json
```

训练入口支持 `--sid-lr`（默认 1e-4）和 `--sid-encoding auto|layered|text`。训练默认启用分层 SID，加载已有分层 checkpoint 时恢复其小表。推理按 checkpoint 恢复 SID 格式。复现旧文本 SID 训练时显式传入 `--sid-encoding text`。

## 保存与迁移

除 adapter/head/model 配置外，新 checkpoint 还必须包含：

- `tokenizer/`：扩展后的 tokenizer，包括已有 chat 模板。
- `sid_embeddings.pt`：767 行可训练向量。
- `sid_tokens.json`：格式版本、token 顺序与 ID、基础词表大小、权重和 tokenizer 的 SHA-256。

使用 `pipeline.models.openjev_runtime.load_model()` 加载。直接调用上游 `DecisionModel.load()` 不支持这些扩展文件。迁移机器时传入 `--base-model`；缺失文件、格式不一致或校验失败会报错，不会静默随机初始化。转换旧 checkpoint 到新输入表示后，应重新做 SFT 和概率校准，不能把旧温度直接当作新模型的温度。

当前只支持驻留内存的输入 embedding，不支持上游 meta/offload 特殊路径。上游 `score_cached()` 尚未验证专用小表兼容性，分层模型会明确拒绝调用；当前 pipeline 的 `score()` 可用。专用 SID 减少输入长度，但不会自动消除逐候选评分开销。

## 历史运行结果

历史 `GSPO_RUN_REPORT.md` 和 `QWEN_RUN_REPORT.md` 记录的是文本 SID 版本，不是新版训练结果。

2026-11-04 已在 WSL Anaconda 的 `openjev_sid` 环境、RTX 4080 上完成真实 2B 模型的 3 步 SFT → 2 步 GSPO → beam=4 推荐验收。两阶段保存重载的 logits 最大误差均为 0，LoRA/head/SID embedding 均有更新。checkpoint 位于 `runs/layered-sid-smoke`，详细记录见 `ENVIRONMENT_TEST_REPORT.md`。这些短程结果仅验证功能，不代表模型已收敛。
