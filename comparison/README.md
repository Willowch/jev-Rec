# OpenJev 五模型对照实验

支持 `ml1m`、`Industrial_and_Scientific`、`Office_Products`。以下命令从服务器上解压后的工作区根目录执行。

## 运行环境

```bash
root=$PWD
openjev_py="$HOME/anaconda3/envs/openjev_sid/bin/python"
baseline_env="$HOME/venvs/openjev_baselines"
baseline_py="$baseline_env/bin/python"
tiger_py="$baseline_py"
recbole_py="$baseline_py"
letter_py="$baseline_py"
```

OpenJev 复用服务器上的 `openjev_sid`。安装或修复该环境时使用 [requirements-openjev.txt](../requirements-openjev.txt)。下面的命令检查现有环境。

```bash
"$openjev_py" --version
"$openjev_py" -c 'import torch, transformers, peft; print(torch.__version__, torch.version.cuda, transformers.__version__)'
"$openjev_py" -m pip check
```

SASRec、BERT4Rec、TIGER 和 LETTER‑TIGER 共用 [requirements-baselines.txt](../requirements-baselines.txt)。用 `openjev_sid` 的 Python 3.11 创建 venv，以复用已安装的 CUDA 版 PyTorch。

```bash
if [ ! -x "$baseline_py" ]; then
  "$openjev_py" -m venv --system-site-packages "$baseline_env"
fi
"$baseline_py" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --prefer-binary --timeout 120 --retries 5 -r "$root/requirements-baselines.txt"
"$baseline_py" -m pip check
```

## 共享数据和测试名单

### 使用已处理的数据

使用下文的压缩包部署后，在解压目录运行上面的环境设置，再运行这一段。`dataset` 可设为 `ml1m`、`Industrial_and_Scientific` 或 `Office_Products`。重新登录后，先设回原来的 `dataset` 和 `run_tag`，再重跑变量设置。

```bash
dataset="${dataset:-ml1m}"
run_tag="${run_tag:-identity-sid-$(date +%Y%m%d-%H%M%S)}"
shared="$root/comparison/data/$dataset"
cohort="$root/comparison/results/cohorts/$dataset-full.json"

(cd "$root" && "$openjev_py" -B - "$dataset" "$shared" "$cohort" <<'PY'
import sys
from pathlib import Path
from comparison.prepare import load_metadata, load_prepared
from comparison.protocol import MAX_HISTORY, load_cohort

dataset, shared, cohort = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
users, items, manifest = load_prepared(shared, dataset)
load_metadata(shared, items, manifest)
selected, details = load_cohort(cohort, dataset, users, items, MAX_HISTORY)
if details["scope"] != "formal":
    raise SystemExit("A formal cohort is required.")
print(f"{dataset}: {len(selected)} test users, {len(items)} items")
PY
)
```

检查通过后进入训练。`features/sentence-t5` 下的缓存供 TIGER 和 LETTER‑TIGER 复用。两个历史 1,000 人名单仍存于 `comparison/cohorts/`，用于复现以往的抽样指标。

### 从原始数据重建

需要在服务器重建时，先上传 `comparison/data/raw/<数据集>/`，再使用显式的原始数据路径生成共享序列和全量测试名单。

```bash
run_tag="identity-sid-$(date +%Y%m%d-%H%M%S)"
experiment_data="$root/comparison/data/$run_tag"
dataset=ml1m
shared="$experiment_data/$dataset"
cohort="$experiment_data/cohorts/$dataset-full.json"
"$openjev_py" -B -m comparison.prepare data --dataset "$dataset" \
  --raw-root "$root/comparison/data/raw" --output "$shared" &&
"$openjev_py" -B -m comparison.prepare cohort --dataset "$dataset" \
  --data-dir "$shared" --output "$cohort"
```

ML‑1M 验证完成后，将 `dataset` 改为 `Industrial_and_Scientific` 或 `Office_Products` 并重新设置 `shared` 和 `cohort`。继续同一实验时可以复用已生成的共享数据及完整名单。

每个模型使用同一个数据集的全量测试名单。ML‑1M 覆盖 6,040 人，Industrial_and_Scientific 覆盖 9,331 人，Office_Products 覆盖 88,383 人。ML‑1M 也可用 `--openjev-evaluation` 从完整历史评估导入同一批用户的顺序，生成时使用新的输出文件名。

Amazon 对同一用户同一 ASIN 保留首次交互，按时间和 ASIN 排序，保留至少 5 件不同物品的用户。五个模型均把每位用户倒数第 4 件物品作为唯一训练目标，后三件依次用于校准、验证和测试。模型输入最多使用最近 20 件物品。验证和测试时屏蔽完整历史中的已见商品。BERT4Rec 只掩码窗口末尾的训练目标。

四个 baseline 的正式训练每 5 轮和末轮验证全量用户，用倒数第二条交互的 NDCG@10 选 checkpoint。NDCG@10 相同的轮次保留较早的权重。TIGER 和 LETTER‑TIGER 的验证 beam 为 10，测试 beam 为 20。训练摘要保存验证人数、目标指纹与最佳轮次。

## 训练和独立评估

以下命令可按阶段执行。每次正式运行选择新的输出目录。

```bash
mkdir -p "$root/comparison/logs" "$root/comparison/results"
gpu="$root/comparison/gpu.sbatch"
log="$root/comparison/logs/%x-%j.log"
eval_step=5
recbole_eval_batch=64
generative_eval_batch=16
oj="$root/openjev_generative_rec"
qwen="$oj/models/qwen3.5-2b"
sentence=$(find "$root/TIGER/02_data/3_models/huggingface" -path '*/snapshots/*/modules.json' -print -quit)
sentence=${sentence%/modules.json}
```

### OpenJev

按 [OpenJev 单卡命令](OPENJEV_MINIONEREC.md) 准备三个数据集的身份 SID，依次运行 SFT 试跑、正式 SFT、GSPO、全量验证选模和全量测试分片。正式测试使用全量验证中胜出的 SFT 或 GSPO checkpoint。

### TIGER

```bash
tiger_run="$root/TIGER/4_runs/comparison/$dataset/$run_tag"
if [ "$dataset" = ml1m ]; then
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sbatch --job-name=tiger-train-ml1m --output="$log" "$gpu" "$root" "$tiger_py" \
    TIGER/0_repos/model/run_ml1m.py --run-dir "$tiger_run" \
    --comparison-data "$shared" --sentence-model "$sentence" --cohort "$cohort" \
    --eval-step "$eval_step" --eval-batch-size "$generative_eval_batch"
else
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sbatch --job-name="tiger-train-$dataset" --output="$log" "$gpu" "$root" "$tiger_py" \
    TIGER/0_repos/model/run_amazon.py all --dataset "$dataset" --run-dir "$tiger_run" \
    --comparison-data "$shared" --sentence-model "$sentence" --cohort "$cohort" \
    --eval-step "$eval_step" --eval-batch-size "$generative_eval_batch"
fi
```

TIGER 的 ML‑1M 新训练直接读取共享序列和电影元数据，再生成本模型的 SID。Amazon 也可按 `prepare`、`sid`、`train`、`evaluate` 分阶段执行，沿用相同的 `--run-dir`，在 `prepare` 传入 `--comparison-data`。TIGER 和 LETTER‑TIGER 从同一目录复用按内容校验的 Sentence‑T5 向量。

正式配置默认 RQ‑VAE 训练 3,000 轮，TIGER 训练最多 200 轮，训练批量 256。用 `--rqvae-epochs`、`--tiger-epochs` 和 `--batch-size` 调整。物品 SID 由三层 RQ‑VAE 码和一层消歧码构成，每层有 256 个码字。默认用前三层的 30 条校准决策拟合四层解码所用的温度。

ML‑1M 将 SID、摘要、验证结果和权重写到运行目录下的 `item_sid.npy`、`summary.json`、`validation.json` 和 `checkpoint/`。Amazon 的 `data/manifest.json` 记录共享数据引用及指纹，`sid/` 保存量化权重和原生 SID，`tiger/` 保存训练记录和最佳权重，`evaluation/evaluation.json` 保存测试结果。旧权重重评时保持原运行目录，并通过 ML‑1M 的 `--output` 或 Amazon 的 `--evaluation-dir` 指定新路径。

将 `tiger_run` 指向已有全量验证记录的训练目录，使用其最佳 checkpoint 重评。

```bash
if [ "$dataset" = ml1m ]; then
  sbatch --job-name=tiger-eval-ml1m --output="$log" "$gpu" "$root" "$tiger_py" \
    TIGER/0_repos/model/run_ml1m.py --run-dir "$tiger_run" \
    --comparison-data "$shared" --cohort "$cohort" --evaluate-only --eval-batch-size "$generative_eval_batch" \
    --output "$root/comparison/results/TIGER-$dataset.json"
else
  sbatch --job-name="tiger-eval-$dataset" --output="$log" "$gpu" "$root" "$tiger_py" \
    TIGER/0_repos/model/run_amazon.py evaluate --dataset "$dataset" --run-dir "$tiger_run" \
    --cohort "$cohort" --eval-batch-size "$generative_eval_batch" \
    --evaluation-dir "$root/comparison/results/TIGER-$dataset"
fi
```

### SASRec、BERT4Rec 和 LETTER‑TIGER

```bash
for model in SASRec BERT4Rec; do
  sbatch --job-name="$model-$dataset" --output="$log" "$gpu" "$root" "$recbole_py" \
    "$model/run.py" --dataset "$dataset" --data-dir "$shared" --cohort "$cohort" \
    --run-dir "$root/$model/runs/$dataset/$run_tag" --device cuda:0 \
    --eval-step "$eval_step" --eval-batch-size "$recbole_eval_batch"
done

sbatch --job-name="letter-$dataset" --output="$log" "$gpu" "$root" "$letter_py" \
  LETTER-TIGER/run.py --dataset "$dataset" --data-dir "$shared" --cohort "$cohort" \
  --run-dir "$root/LETTER-TIGER/runs/$dataset/$run_tag" --sentence-model "$sentence" --device cuda:0 \
  --eval-step "$eval_step" --eval-batch-size "$generative_eval_batch"
```

这三个入口训练后自动评估。独立重评使用同一入口，传入已有 `--run-dir`，附加 `--evaluate-only --output <新结果路径>`。SASRec 与 LETTER 的协同训练复用同一份 RecBole 下一物品训练输入。BERT4Rec 使用原生末位掩码输入。新训练自动生成独立缓存，原始共享数据和全量测试名单可直接复用。

#### LETTER‑TIGER 的配置和产物

LETTER‑TIGER 复用 TIGER 的物品文本、Sentence‑T5、T5 骨干及受约束解码。训练、物品编码和评估的批量默认分别为 256、32 和 16，可用 `--train-batch-size`、`--embedding-batch-size`、`--eval-batch-size` 调整。 [LETTER 实现版本](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda) · [论文](https://arxiv.org/abs/2405.07314)

- 协同向量使用 32 维 SASRec 物品表示，默认训练 100 轮，学习率 0.001，dropout 0.2。协同损失使用训练前缀中出现的物品。
- tokenizer 使用四层各 256 个 32 维码字。以固定种子抽取至多 4,096 件物品初始化，随后按学习率 0.001、批量 1,024 在全目录训练 3,000 轮。协同与多样性损失权重分别为 0.02 和 0.0001。按全目录碰撞率保存最早达到最低值的权重，导出时处理残余碰撞，确保每件物品的四段 SID 唯一。
- 生成器使用训练温度 0.7，在学习率 0.0005 和 0.001 上各训练至多 100 轮，采用 AdamW 和 1% warmup 的余弦学习率计划，30 轮未改善时早停。用全量验证的 NDCG@10 选模，并在前三层的 30 条独立校准决策上拟合推理温度。
- `summary.json` 记录配置与指纹，`cf/`、`tokenizer/` 和 `trial-*/` 保存权重和日志。`sids.json` 保存物品与 SID 的映射，`evaluation.json` 保存逐用户榜单与指标。

需要快速检查流程时，使用 `comparison.prepare cohort --smoke` 创建独立的 8 人名单，并在新运行目录用 `--smoke` 启动模型。正式汇总要求对应数据集的全量名单。

## 汇总指标

单卡 4090 上，测试会分批推理并逐条校验和写出榜单。显存不足时，将 `recbole_eval_batch` 降到 32，或将 `generative_eval_batch` 降到 8，随后用同一个最佳 checkpoint 重评。

所有正式结果保存 Top‑20，统一候选目录和已见物品过滤。生成式模型测试使用 beam 20。SASRec 和 BERT4Rec 使用全目录打分。正式汇总要求五个模型结果齐全，并校验单目标训练标记、全量选模记录、测试名单、用户目标和推荐列表。默认计算 Hits、Recall 和 NDCG 的 @5 和 @10，传入 `--include-at20` 可加入 @20。

```bash
tiger_result="$tiger_run/evaluation.json"
if [ "$dataset" != ml1m ]; then
  tiger_result="$tiger_run/evaluation/evaluation.json"
fi
baseline_results=(
  "$tiger_result"
  "SASRec/runs/$dataset/$run_tag/evaluation.json"
  "BERT4Rec/runs/$dataset/$run_tag/evaluation.json"
  "LETTER-TIGER/runs/$dataset/$run_tag/evaluation.json"
)
"$openjev_py" -B -m comparison.summarize --dataset "$dataset" \
  --data-dir "$shared" --cohort "$cohort" "$run/evaluation.json" "${baseline_results[@]}"

# 只汇总四个 baseline 时执行这一条。
"$openjev_py" -B -m comparison.summarize --baselines-only --dataset "$dataset" \
  --data-dir "$shared" --cohort "$cohort" "${baseline_results[@]}"
```

若使用独立重评的结果，将对应条目指向新输出文件。旧滑窗训练的 baseline 需在新目录重新训练。

## Windows 打包

在本地工作区根目录打开 PowerShell。下面的包包含三套已处理数据、全量名单、Sentence‑T5 缓存、五个模型入口和离线权重。内嵌仓库的 `.git` 会排除，训练读取的 `UPSTREAM_COMMIT` 会保留。确需走原始数据重建路线时，将清单中的三个共享数据目录和 `comparison/results/cohorts` 换成 `comparison/data/raw`。

```powershell
$workspace = (Get-Location).Path
$archive = Join-Path (Split-Path -Parent $workspace) 'openjev-4090-comparison.tar.gz'
if (Test-Path -LiteralPath $archive) { throw '压缩包已存在，请先更换输出文件名' }
$paths = @(
  'requirements-openjev.txt', 'requirements-baselines.txt',
  'comparison/__init__.py', 'comparison/prepare.py', 'comparison/sources.py',
  'comparison/features.py', 'comparison/protocol.py', 'comparison/recbole_runner.py',
  'comparison/summarize.py', 'comparison/gpu.sbatch', 'comparison/README.md',
  'comparison/OPENJEV_MINIONEREC.md',
  'comparison/data/ml1m', 'comparison/data/Industrial_and_Scientific',
  'comparison/data/Office_Products', 'comparison/results/cohorts',
  'openjev_generative_rec/pipeline', 'openjev_generative_rec/models',
  'openjev_generative_rec/third_party/Open-Jev', 'TIGER/0_repos',
  'TIGER/3_configs/mlflow.json', 'TIGER/02_data/3_models/huggingface',
  'SASRec/run.py', 'SASRec/model.yaml', 'BERT4Rec/run.py', 'BERT4Rec/model.yaml',
  'LETTER-TIGER/run.py', 'LETTER-TIGER/features.py', 'LETTER-TIGER/generative.py', 'LETTER-TIGER/tokenizer.py'
)
foreach ($path in $paths) {
    if (-not (Test-Path -LiteralPath (Join-Path $workspace $path))) { throw "缺少打包输入：$path" }
}
tar.exe -czf $archive '--exclude=.git' '--exclude=__pycache__' `
  '--exclude=openjev_generative_rec/third_party/Open-Jev/.git' -C $workspace @paths
if ($LASTEXITCODE -ne 0) { throw "压缩失败，tar 退出码 $LASTEXITCODE" }
$entries = tar.exe -tzf $archive
if ($LASTEXITCODE -ne 0) { throw '无法读取压缩包目录' }
if ($entries -match '(^|/)\.git(/|$)') { throw '包内仍含 .git' }
if ($entries -notcontains 'openjev_generative_rec/third_party/Open-Jev/UPSTREAM_COMMIT') {
    throw '包内缺少 UPSTREAM_COMMIT'
}
Get-Item -LiteralPath $archive
```

## 同步到已有服务器工作区

本次在上一版已同步的工作区上更新以下文件。全新部署可直接使用上面的归档和已处理数据启动步骤。

| 目录 | 同步文件 |
| --- | --- |
| `comparison/` | `summarize.py`、`README.md`、`OPENJEV_MINIONEREC.md` |
| `openjev_generative_rec/pipeline/data_preparation/` | `prepare_amazon.py`、`prepare_comparison.py`、`embedding_cache.py` |
| `openjev_generative_rec/pipeline/train_infer/` | `train_lora.py`、`recommend.py`、`gspo_selection.py`、`test_shards.py`、`evaluation_benchmark.py` |
