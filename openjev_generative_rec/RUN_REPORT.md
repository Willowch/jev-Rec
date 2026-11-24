# 本机执行记录

已完成数据下载、Amazon Leave-one-out/DS、ML-1M embedding、三层 RQ-Kmeans、已发布 OpenJev 的 LoRA 短程微调，以及真实自回归推荐验证。

## 数据与环境

- 原环境：Ubuntu-20.04，`/home/accelerator/anaconda3/envs/pytorchgpu`，Python 3.8.16 / PyTorch 1.13.1 / CUDA 11.7。
- 训练隔离环境：`../../work/openjev-env`，Python 3.11 / PyTorch 2.8.0+cu128 / Transformers 5.10.2 / PEFT 0.19.1，RTX 4080 16GB。
- Amazon：1,429 用户；训练 1,429 → DS 后 7,403 条，测试 1,429 条。全部时间切分已检查。
- ML-1M：6,040 用户，1,000,209 交互；训练 2,928,147 个可按需构造的 token 样本。
- Embedding：3,883 × 384；本地 MiniLM 22,713,216 参数。
- RQ：3 × 255 个中心；3,786 个唯一三元组，93 个碰撞组涉及 190 个物品。
- 最长配置输入检查：422 tokens，低于 checkpoint 4,096 上限。

## 实际微调与推理

这是 **3 步流程验证 checkpoint**，仅消费 3 个训练决策，分别覆盖三个语义层；不是完成一个 epoch 或已经收敛的推荐模型。完整训练脚本默认 1,000 步，可调整预算后运行。

- 初始化：已发布 Open-Jev-2B 的 LoRA 与 head；基础权重固定版本并验证 SHA-256。
- 只训练 2,508,801 个 LoRA/head 参数，候选分块大小 1。
- 每个样本仍计算完整 255 类 softmax 与 cross-entropy + Brier 损失。
- 三步 loss：6.558540, 6.084576, 5.456218。
- 训练日志记录的峰值张量显存：4.12 GiB；不等于整卡占用。
- 训练、校准及重载总时间（不含初次加载）：973.1 秒。
- 校准样本数：3，仅用于验证校准过程；温度为 0.973496，不可据此判断校准质量。
- 保存后重载的 logits 最大绝对误差：0.000000。
- 保存的适配器中 120 个 LoRA 张量均相对发布版本发生更新（详见 `adapter_change.json`）。
- 自回归评估测试用户数：1，beam=3。完整物品目录检索，排除历史已交互物品，不读取目标语义前缀。
- 测试输出：`runs/smoke-evaluation.json`；极小 smoke 样本不提供推荐质量结论。

## 检查

- 四项关键测试通过：分块梯度等价、全部用户时间边界、无真实目标前缀的目录约束解码、码本维度与范围。
- 官方 OpenJev 校验器通过 1,200 条 schema 样本（每个有效 split 300 条）；这不是对全部训练记录的全量 schema 审计。
- 训练环境 `pip check` 无冲突。

当前共享数据的来源哈希见各数据集的 `manifest.json`。部署和运行命令见[五模型部署总指南](../comparison/README.md)。

## 可选 FLA 端到端验证

在上述 3 步 checkpoint 基础上，用 FLA 0.5.2 再训练 3 步（累计 6 步），并完成 3 样本校准和重载。仍然是 smoke checkpoint。

- 线性注意力实际使用 `chunk_gated_delta_rule`；LoRA/head 参数数量不变。
- 三步 loss：6.204323, 5.673757, 5.873182。
- 后两步耗时：51.0、50.4 秒；与兼容实现使用不同样本，且有并行验证，不能视为严格速度基准。
- 重载误差：0.000000；温度 0.538267。
- 输出目录：`runs/ml1m-fast-smoke/checkpoint`。
