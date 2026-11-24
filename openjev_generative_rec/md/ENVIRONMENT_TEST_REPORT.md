# WSL 训练环境安装与测试报告

2026-11-04 的历史验收记录。该次验收使用分层 SID token、可训练 SID embedding、OpenJev 候选评分，以及 SFT 后继续做 GSPO 与正样本 NLL 训练。现行训练和评估见 [OpenJev 单卡运行步骤](../../comparison/OPENJEV_MINIONEREC.md)。

## 环境处理

原环境 `/home/accelerator/anaconda3/envs/pytorchgpu` 已安装 `peft==0.10.0`、`accelerate==0.29.3`。安装前后的 pip freeze 对比仅增加这两个包，原 Python 3.8.16、PyTorch 1.13.1、Transformers 4.40.2 保留。

该环境的 21 项单元测试通过，但真实 `jev.model` 导入仍因 Transformers 缺少 `AutoModelForImageTextToText` 而失败。根据用户授权，在 WSL Anaconda 下新建了不同名环境 `openjev_sid`。

| 项目 | 新训练环境 |
|---|---|
| 路径 | `/home/accelerator/anaconda3/envs/openjev_sid` |
| Python | 3.11.18 |
| PyTorch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| Transformers | 5.10.2 |
| PEFT / Accelerate | 0.19.1 / 1.13.0 |
| FLA / fla-core | 0.5.2 / 0.5.2 |
| 实测 GPU | NVIDIA GeForce RTX 4080 16GB |
| 环境占用 | 约 6.9 GiB |

依赖从官方 PyPI 下载，全部 wheel 按解析计划中的 SHA-256 校验后安装。安装完成后已删除本次临时 wheel 目录，释放 **3.688 GiB**。此前卸载的项目内 `work/openjev-env` 没有重新创建。

`run.sh`、`run_sid.sh`、`run_qwen.sh`、`run_gspo.sh` 默认训练解释器已改为 `openjev_sid`；`TRAIN_PY` 可覆盖。`run.sh test` 使用该训练环境。数据预处理仍由原 `pytorchgpu` 执行。

## 依赖和基础测试

- 新环境 `pip check`：通过，无依赖冲突。
- 新环境 **21 项单元测试全部通过**，耗时 8.728 秒。覆盖数据时间边界、分块梯度、GSPO 数学、分层 token 原子性、SID 梯度、原词表冻结、保存重载等。
- Qwen 数据的官方 schema 样本校验：**1,200 条 Choice 记录通过**，train/calibration/validation/test 各 300 条。
- 四个运行脚本的 Bash 语法检查通过。
- CUDA 前向/反向、bf16 输出与 FP32 SID 小表更新通过；原词表权重逐元素不变。
- `jev.model`、`jev.data`、`pipeline.train_infer.train_lora`、`pipeline.train_infer.recommend` 均可导入。

原环境 `pip check` 另发现既有的 TensorFlow Probability 0.21.0 与 typing-extensions 4.11.0 版本约束冲突。安装日志和前后 freeze 确认此次未改变这两个包；本项目的新训练环境无此冲突。

## 真实 OpenJev-2B GPU 测试

复用现有 Qwen embedding、3×255 RQ 语义 ID 和 ML-1M 时间划分。本次没有重新生成 embedding 或重训码本。新模型输入使用 765 个分层 SID token 和 2 个边界 token，SID 小表为 1,570,816 个参数，总可训练参数 **4,079,617**。实际线性注意力内核为 FLA `chunk_gated_delta_rule`。

产物目录：`runs/layered-sid-smoke`。

| 检查 | SFT | GSPO + NLL |
|---|---:|---:|
| 优化步数 | 3，分别覆盖三个 SID 层 | 2，同一组 4 条轨迹复用两次 |
| 末步损失 | 5.630592 | 0.680050 |
| LoRA 最大参数变化 | 0.000060102 | 0.000040043 |
| decision head 最大变化 | 0.000300340 | 0.000200216 |
| SID embedding 最大变化 | 0.000300497 | 0.000200249 |
| 保存重载 logits 最大误差 | **0** | **0** |
| PyTorch 峰值已分配显存 | 4.123 GiB | 4.123 GiB |

每步损失、梯度均为有限值，SID 小表每步都有非零梯度。显存数值是 PyTorch 进程内已分配峰值，不等于 nvidia-smi 的整卡总占用。

GSPO 第一次更新的策略 logit 梯度范数为 0.21888，证明策略项实际参与反传。第二次更新保留相同 old log-prob，四条轨迹全部触发裁剪（clip_fraction=1），此时策略项梯度为零，正样本 NLL 继续产生更新。这是当前极窄裁剪范围（0.0003/0.0004）下该样本组的观测，不代表长期训练表现。

推荐推理验收通过：1 个测试用户、beam=4、top-k=4，返回 4 个有效且未交互的目录物品；逐项检查其 SID 与目录映射一致，分数有限，物品无重复。用户 783 的输出物品为 260、2123、2142、1196。本次单用户 Recall@4、NDCG@4 均为 0；这一短程验收样本不用于估计正式推荐效果。

`work/audit_sid_smoke.py` 已通过：核对两阶段参数更新、767 个 token、权重/tokenizer 校验和、重载误差、GSPO 旧策略复用与裁剪，以及推荐输出有效性。完整验收结果保存于 `runs/layered-sid-smoke/verification.json`，测试脚本正常退出。

本次为短程功能验收，不能据此判断模型收敛或推荐质量。完整实验仍需增加训练步数并评估全量测试用户。

## 当时的命令与产物

以下命令记录当时在 WSL 项目目录使用的旧入口。

```text
source /home/accelerator/anaconda3/etc/profile.d/conda.sh
conda activate openjev_sid
bash run.sh test

# 新目录重跑两阶段短程测试，避免覆盖本次产物
SID_RUN_DIR=runs/layered-sid-next SFT_STEPS=3 GSPO_STEPS=2 bash run.sh sid
```

- `work/openjev_sid_environment.json`：新环境版本、CUDA 和导入检查。
- `work/openjev-sid-freeze.txt`：本次实际安装的完整依赖版本。
- `work/openjev-sid-tests.log`、`work/openjev-sid-schema.log`：单元测试和 schema 校验。
- `work/layered-sid-smoke.log`、`work/layered-sid-evaluation.log`：真实模型训练、重载和推荐日志。
- `runs/layered-sid-smoke/stage1/summary.json`、`stage2-gspo/summary.json`：训练验收结果。
- `runs/layered-sid-smoke/stage2-gspo/checkpoint`：本次可加载的分层 SID + GSPO checkpoint。
- `work/pytorchgpu-before-install.txt`、`work/pytorchgpu-after-install.txt`：旧环境前后依赖记录。
- `work/wheel-cleanup.json`：临时安装包清理记录。
