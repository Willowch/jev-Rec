# OpenJev 单卡 4090 正式实验

先按[对比实验说明](README.md#共享数据和测试名单)设置环境，准备或复用共享数据和全量测试名单。五个模型使用相同的 `dataset`、`shared` 和 `cohort`。先对 ML‑1M 跑完整流程，再确定 Amazon 的训练预算。

```bash
# 在解压后的工作区根目录执行，沿用 README 中的变量。
run="$oj/runs/minionerec/$dataset-$run_tag"
sid="$run/model-data/sequences"
gspo_steps=50
candidate_batch=4
shard_size=500
calibration_seed=43
mkdir -p "$root/comparison/logs"

prep_job=$(sbatch --parsable --job-name="jev-sid-$dataset" --output="$log" \
  "$gpu" "$oj" "$openjev_py" -m pipeline.data_preparation.prepare_comparison \
  --dataset "$dataset" --comparison-data "$shared" --base-model "$qwen" \
  --output "$run/model-data" --batch-size 8)
printf 'SID job %s\n' "$prep_job"
```

三个数据集都用三层 RQ‑KMeans。存在三段同码时，统一按物品 ID 为同码组添加身份 token。若要复用向量，给准备入口传 `--embedding-cache <上次准备目录的 embeddings 子目录>`，并保持相同编码批量。程序校验物品目录、文本、编码器和缓存文件。旧 checkpoint 继续使用原训练的 SID 映射。

## 试跑和正式 SFT

试跑和正式训练都从相同的发布权重初始化。每次抽到一个训练目标，就学习该目标所有存在多个合法候选的 SID 层。正式 SFT 保存末步权重，并在独立校准集拟合温度。

```bash
sft_args=(--objective sft --sft-recipe minionerec --sft-all-layers
  --sid-encoding layered --history-memory on --validation-users 0
  --data "$sid" --comparison-data "$shared" --base-model "$qwen"
  --checkpoint "$oj/models/openjev-2b/package/checkpoint"
  --seed 42 --candidate-batch "$candidate_batch" --warmup-steps 20
  --calibration-rows 30 --calibration-seed "$calibration_seed")

pilot_job=$(sbatch --parsable --dependency="afterok:${prep_job%%;*}" \
  --job-name="jev-pilot-$dataset" --output="$log" --time=16:00:00 \
  "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.train_lora \
  "${sft_args[@]}" --steps 30 --output "$run/pilot")
printf 'Pilot job %s\n' "$pilot_job"
```

试跑完成后估算正式步数。以下命令取第 6 至 30 步的 P90 步耗时，用 16 小时预算的 80% 估算最多能训练多少步，首轮至多遍历一次用户。试跑摘要和日志记录实际覆盖、有效 SID 决策数和显存峰值。变更候选批量时，先用新批量重测。

```bash
sft_steps=$("$openjev_py" -B - "$run/pilot" "$shared/manifest.json" <<'PY'
import json
import math
import sys
from pathlib import Path

pilot, manifest = map(Path, sys.argv[1:])
if json.loads((pilot / "summary.json").read_text())["status"] != "complete":
    raise SystemExit("Wait for the pilot to finish first.")
rows = [json.loads(line) for line in (pilot / "training.jsonl").read_text().splitlines()]
if [row["step"] for row in rows] != list(range(1, 31)):
    raise SystemExit("Expected a completed 30-step pilot.")
durations = sorted(end["elapsed_seconds"] - start["elapsed_seconds"] for start, end in zip(rows[4:], rows[5:]))
p90 = durations[math.ceil(0.9 * len(durations)) - 1]
if p90 <= 0:
    raise SystemExit("Invalid pilot timings.")
users = json.loads(manifest.read_text())["users"]
steps = min(users, math.floor(0.8 * 16 * 3600 / p90))
if steps < 1:
    raise SystemExit("The job budget is shorter than one estimated step.")
print(f"P90 {p90:.2f} seconds per step", file=sys.stderr)
print(steps)
PY
)

sft_job=$(test -n "$sft_steps" &&
  sbatch --parsable --job-name="jev-sft-$dataset" --output="$log" --time=16:00:00 \
  "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.train_lora \
  "${sft_args[@]}" --steps "$sft_steps" --output "$run/sft")

gspo_args=(--objective gspo --full-validation --reward minionerec-ranked --checkpoint "$run/sft/checkpoint"
  --data "$sid" --comparison-data "$shared" --base-model "$qwen"
  --steps "$gspo_steps" --seed 43 --candidate-batch "$candidate_batch" --group-size 4
  --nll-weight 0.1 --calibration-rows 30 --calibration-seed "$calibration_seed")
gspo_run="$run/gspo"
gspo_job=$(test -n "$sft_job" &&
  sbatch --parsable --dependency="afterok:${sft_job%%;*}" \
  --job-name="jev-gspo-$dataset" --output="$log" --time=08:00:00 \
  "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.train_lora \
  "${gspo_args[@]}" --rollout-updates 2 --output "$gspo_run")
printf 'SFT %s\nGSPO %s\nRun %s\n' "$sft_job" "$gspo_job" "$run"
```

### GSPO 有效奖励诊断与可选对照

先看 SFT 摘要里的 `position_coverage` 和 `effective_sid_decisions`。GSPO 的 `sampled_contexts` 计算新采样组，`unique_sampled_contexts` 计算其中不同的训练上下文。同一组样本的重复更新只计入更新步数。`hit_groups`、`zero_hit_groups` 和 `nonzero_advantage_groups` 区分命中与有效组内优势。`steps_with_policy_gradient` 计算至少一组产生非零 GSPO logit 梯度的更新步。整组未命中时，正样本 NLL 仍可提供梯度。

```bash
cat "$run/sft/summary.json" "$gspo_run/summary.json"
```

需要对照时，用同一份 SFT 和相同的 50 步预算，每步采样新组，写入独立目录。用下面的全量验证流程分别验证两组 GSPO 权重并比较 NDCG@10 和耗时。先完成验证比较，再对最终选定的 checkpoint 运行测试。

```bash
alt_gspo_run="$run/gspo-fresh-every-step"
alt_gspo_job=$(test -n "$sft_job" &&
  sbatch --parsable --dependency="afterok:${sft_job%%;*}" \
  --job-name="jev-gspo-fresh-$dataset" --output="$log" --time=08:00:00 \
  "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.train_lora \
  "${gspo_args[@]}" --rollout-updates 1 --output "$alt_gspo_run")
printf 'Optional GSPO job %s\n' "$alt_gspo_job"
# 验证这组对照时设置 gspo_run="$alt_gspo_run"、gspo_job="$alt_gspo_job"。
```

### 用 4090 估算全量评估时间

提交正式验证前，尤其是 Office_Products，可用正式 SFT checkpoint 对实际模型测速。验证用 beam 10，测试用 beam 20。每种模式预热 2 人后测 30 人，采用和正式评估相同的候选批量、历史及推理逻辑，报告每人耗时、峰值显存、各阶段 GPU 小时和三轮验证加一轮测试的串行估算。估算计入每片加载时间，调度等待由服务器队列决定。测试样本只计时和显存，正式指标由全量测试计算。

用 `estimated_serial_gpu_hours` 估算单卡总预算。若某阶段的 `estimated_p90_shard_hours` 接近作业时限，在开始生成分片前减小 `shard_size`。

```bash
bench_job=$(test -n "$sft_job" &&
  sbatch --parsable --dependency="afterok:${sft_job%%;*}" \
  --job-name="jev-eval-bench-$dataset" --output="$log" --time=04:00:00 \
  "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.evaluation_benchmark \
  --checkpoint "$run/sft/checkpoint" --base-model "$qwen" --data "$sid" --comparison-data "$shared" \
  --cohort "$cohort" --candidate-batch "$candidate_batch" --shard-size "$shard_size" \
  --output "$run/evaluation-benchmark.json")
printf 'Evaluation benchmark job %s\n' "$bench_job"
# 作业完成后执行。
# cat "$run/evaluation-benchmark.json"
```

## 全量验证并选择 checkpoint

`step=0` 是 GSPO 开始前保存的 SFT 权重。三个候选用同一验证集、beam 10 和 NDCG@10。以下数组依次运行，每个数组最多同时运行一个分片。

```bash
users=$("$openjev_py" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["users"])' "$shared/manifest.json")
last_shard=$(( (users + shard_size - 1) / shard_size - 1 ))
previous=${gspo_job%%;*}
validation_dir="$gspo_run/validation"
validation_args=(--run "$gspo_run" --data "$sid" --comparison-data "$shared" --base-model "$qwen"
  --validation-dir "$validation_dir" --shard-size "$shard_size" --candidate-batch "$candidate_batch")

if [ -n "$previous" ]; then
  for step in 0 "$((gspo_steps / 2))" "$gspo_steps"; do
    array_job=$(sbatch --parsable --dependency="afterok:$previous" --array="0-${last_shard}%1" \
      --job-name="jev-valid-$dataset-$step" --output="$log" \
      "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.gspo_selection validate \
      "${validation_args[@]}" --step "$step") || break
    previous=${array_job%%;*}
    printf 'Candidate %s validation job %s\n' "$step" "$previous"
  done
fi
```

用 `sacct` 确认三个候选的验证数组完成。若数组失败，单独提交失败的编号。重提已被 `afterok` 阻塞的后续数组时，使用补跑成功的作业号。基础模型目录及文件元数据、候选批量、分片大小和候选权重指纹校验通过后才能复用分片。旧分片缺少这些设置，或需要改变参数时，显式把 `validation_dir` 改为例如 `$gspo_run/validation-v2`，重建 `validation_args` 后重新提交三组数组。旧目录保留。

```bash
selected_checkpoint="$gspo_run/checkpoint"
# 若原 checkpoint 已存在，可改用 selected_checkpoint="$gspo_run/checkpoint-v2"。
(cd "$oj" && "$openjev_py" -B -m pipeline.train_infer.gspo_selection select \
  "${validation_args[@]}" --output "$selected_checkpoint") &&
cat "$selected_checkpoint/selection.json"
```

选择记录包含三个候选的分数、相对 SFT 的 NDCG@10 差值和实际获胜的权重。同分取较早者，SFT 也可进入正式测试。

## 全量测试分片与合并

选模成功后提交全量测试。同一张 4090 串行执行切片，每片成功后保存独立榜单。保持原名单、权重、候选批量及分片大小，使用 `--array='3,7%1'` 等方式补跑缺失片。分片设置变动时使用新的分片目录。

```bash
test_args=(--checkpoint "$selected_checkpoint" --base-model "$qwen"
  --data "$sid" --comparison-data "$shared" --cohort "$cohort"
  --candidate-batch "$candidate_batch" --shard-size "$shard_size" --shard-dir "${selected_checkpoint}-test-shards")

test_job=$(test -s "$selected_checkpoint/selection.json" &&
  sbatch --parsable --array="0-${last_shard}%1" --job-name="jev-test-$dataset" --output="$log" \
  "$gpu" "$oj" "$openjev_py" -m pipeline.train_infer.test_shards infer "${test_args[@]}")
printf 'Test job %s\n' "$test_job"
```

所有分片完成后在登录节点合并。合并校验用户、目标、榜单和指纹，写出标准 `evaluation.json`，供五模型汇总使用。

```bash
(cd "$oj" && "$openjev_py" -B -m pipeline.train_infer.test_shards merge \
  "${test_args[@]}" --output "$run/evaluation.json")
```

## 已训练的 ML‑1M 权重加速评估

以下用服务器现有的 `ml1m-jev-full-v1` 运行目录，复用已经训练的 SFT 和 GSPO 权重。每个验证或测试数组最多并行四个分片，每个分片申请一张 4090。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
cd "$root"
oj="$root/openjev_generative_rec"
openjev_py="$HOME/anaconda3/envs/openjev_sid/bin/python"
run="$oj/runs/minionerec/ml1m-jev-full-v1"
gspo="$run/gspo"
audit="$gspo/scoring-audits"
cohort="$root/comparison/results/cohorts/ml1m-full.json"
gpu_run=("$root/comparison/gpu.sbatch" "$oj" "$openjev_py" -m)
eval_input=(--data "$run/model-data/sequences" --comparison-data "$root/comparison/data/ml1m"
  --base-model "$oj/models/qwen3.5-2b" --candidate-batch 4)
mkdir -p "$root/comparison/logs"

submit_eval() {
  env -u SBATCH_GET_USER_ENV sbatch --export=ALL --parsable \
    --ntasks=1 --cpus-per-task=2 \
    --output="$root/comparison/logs/jev-ml1m-%x-%j.log" "$@"
}
```

GSPO 结束后，对三个候选各提交一次缓存一致性检查。每份报告在固定的 32 位验证用户上对比两条评分路径，覆盖 beam 10 和 beam 20。检查会记录榜单、目标名次、同一候选的分数偏差、耗时和显存。

```bash
for step in 0 25 50; do
  submit_eval --job-name="audit-$step" --time=02:00:00 \
    "${gpu_run[@]}" pipeline.train_infer.evaluation_benchmark "${eval_input[@]}" \
    --checkpoint "$gspo/candidates/step-$step" --cohort "$cohort" \
    --compare-backends --output "$audit/step-$step.json" || break
done
```

三份报告都生成后执行下一段。只有三份 `scoring_audit.passed` 都通过才会启用前缀缓存。时间紧时可在下方保留 `backend=batched` 和 `audit_args=()`，直接用原打分路径的四卡分片并行。

```bash
backend=batched
audit_args=()
if "$openjev_py" -B -c 'import json,sys; from pathlib import Path
d=Path(sys.argv[1])
files=[d/f"step-{step}.json" for step in (0,25,50)]
sys.exit(not all(p.is_file() and json.loads(p.read_text())["scoring_audit"]["passed"] for p in files))' "$audit"; then
  backend=prefix-cache
  audit_args=(--backend-audits "$audit")
fi

selected="$gspo/checkpoint-$backend"
val_args=("${eval_input[@]}" --run "$gspo" --shard-size 500 \
  --validation-dir "$gspo/validation-$backend" --scoring-backend "$backend" "${audit_args[@]}")
test_args=("${eval_input[@]}" --checkpoint "$selected" --cohort "$cohort" \
  --shard-size 500 --shard-dir "${selected}-test-shards" --scoring-backend "$backend")
previous=
for step in 0 25 50; do
  dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:$previous" --kill-on-invalid-dep=yes)
  fi
  job=$(submit_eval --job-name="valid-$backend-$step" --array='0-12%4' "${dependency[@]}" \
    "${gpu_run[@]}" pipeline.train_infer.gspo_selection validate "${val_args[@]}" --step "$step") || break
  previous=${job%%;*}
  printf 'step=%s validation=%s\n' "$step" "$previous"
done
```

三个验证数组完成后选模并提交全量测试。

```bash
(cd "$oj" && "$openjev_py" -B -m pipeline.train_infer.gspo_selection select \
  "${val_args[@]}" --output "$selected") &&
submit_eval --job-name="test-$backend" --array='0-12%4' \
  "${gpu_run[@]}" pipeline.train_infer.test_shards infer "${test_args[@]}"
```

测试数组完成后合并，终端显示完整指标，结果可用于五模型汇总。

```bash
(cd "$oj" && "$openjev_py" -B -m pipeline.train_infer.test_shards merge \
  "${test_args[@]}" --output "$run/evaluation-$backend.json")
```

## 后续 SFT 和 GSPO 的候选编码复用

提交新训练时，在原 `sft_args` 和 `gspo_args` 中加入 `--training-scorer prepared`。保留相同 checkpoint、训练样本、种子和损失设置。优先用原候选批量核对 `training.jsonl` 的损失、梯度范数、耗时和峰值显存，再根据 4090 的实测比较 `--candidate-batch 8` 与 `16`。选出合适的批量后将它传给正式训练。常规入口的默认打分路径为 `--training-scorer legacy`。

## ML‑1M 全量 SFT 提速短测

在服务器项目根目录的 Bash 中初始化。以下命令复用已经生成的 `ml1m-jev-full-v1` SID 数据。短测从同一份发布权重开始，使用独立输出目录。

```bash
root="$PWD"
oj="$root/openjev_generative_rec"
py="$HOME/anaconda3/envs/openjev_sid/bin/python"
sid="$oj/runs/minionerec/ml1m-jev-full-v1/model-data/sequences"
bench="$oj/runs/minionerec/ml1m-sft-speed-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$root/comparison/logs"
printf '短测输出 %s\n' "$bench"

sft_common=(--objective sft --sft-recipe minionerec --sft-all-layers
  --training-scorer prepared --sid-encoding layered --history-memory on --validation-users 0
  --data "$sid" --comparison-data "$root/comparison/data/ml1m"
  --base-model "$oj/models/qwen3.5-2b" --checkpoint "$oj/models/openjev-2b/package/checkpoint"
  --seed 42 --accumulation 1 --lr 2e-5 --head-lr 1e-4 --sid-lr 1e-4 --calibration-seed 43)

submit_sft_speed() {
  local name=$1 output=$2 gpus=$3 batch=$4 checkpointing=$5 limit=$6
  shift 6
  local launch=(-m pipeline.train_infer.train_lora)
  if (( gpus > 1 )); then
    launch=(-m torch.distributed.run --standalone --nnodes=1 "--nproc-per-node=$gpus"
      --module pipeline.train_infer.train_lora --candidate-parallel)
  fi
  env -u SBATCH_GET_USER_ENV TOKENIZERS_PARALLELISM=false sbatch --export=ALL --parsable \
    --job-name="$name" --nodes=1 --ntasks=1 --cpus-per-task="$((2 * gpus))" \
    --gres="gpu:4090:$gpus" --time="$limit" --output="$root/comparison/logs/$name-%j.log" \
    "$root/comparison/gpu.sbatch" "$oj" "$py" "${launch[@]}" \
    "${sft_common[@]}" --candidate-batch "$batch" --gradient-checkpointing "$checkpointing" \
    --output "$output" "$@"
}

probe=(--steps 32 --warmup-steps 15 --calibration-rows 3 --audit-sft-steps 4 --profile-steps 4)
submit_sft_speed jev-sft-b16-on "$bench/b16-on" 1 16 on 00:45:00 "${probe[@]}"
submit_sft_speed jev-sft-b16-off "$bench/b16-off" 1 16 off 00:45:00 "${probe[@]}"
```

看 `run.json` 的 `linear_attention_kernels` 和 `causal_conv1d_kernels`。`torch_chunk_gated_delta_rule` 和 `torch_conv1d` 分别表示对应操作使用 PyTorch 实现。旧作业的 `run.json` 可能只有第一项，通用的 fast path 警告也可能只代表缺少其中一种内核。若需要验证依赖，在独立环境运行同一短测。保持现有 PyTorch 和 Transformers 版本，使用匹配 Python 3.11、PyTorch/CUDA 和 C++ ABI 的扩展包。缺少兼容包时可继续使用原环境做检查点和多卡对照。

```bash
conda create --yes --clone "$HOME/anaconda3/envs/openjev_sid" \
  --prefix "$HOME/anaconda3/envs/openjev_sid_kernels"
kernel_py="$HOME/anaconda3/envs/openjev_sid_kernels/bin/python"
"$kernel_py" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.compiled_with_cxx11_abi())'
# 获取兼容的扩展包后，在克隆环境安装并提交同配置短测。
# "$kernel_py" -m pip install --no-deps /path/to/compatible-extension.whl
# py="$kernel_py"
# submit_sft_speed jev-sft-kernels "$bench/kernels" 1 16 on 00:45:00 "${probe[@]}"
```

在单卡 `off` 通过核验且显存还有余量时，可保持相同设置试 `batch=32`。若单卡预计仍超过 24 小时，申请同节点四张 4090。`candidate-batch` 在多卡模式下是每卡批量。资源暂不可用时，可把下列卡数改成 `3`。先以 `on` 验证多卡，只有同批量的单卡 `off` 有足够显存余量且通过核验时才继续尝试关闭。

```bash
submit_sft_speed jev-sft-g4 "$bench/g4" 4 64 on 00:45:00 "${probe[@]}"
```

短测完成后，分别用 `sft_audit` 和训练记录核对每组。以下示例对照四卡，把 `trial` 改为 `b16-off` 可核对单卡关闭梯度检查点。选择配置时，关注第 9 到第 32 步的平均值和 P90。前 4 步的计时和审计有额外开销。多卡 `phase_seconds` 取各阶段在各卡上的最大值，显存字段取每卡峰值的最大值。

```bash
trial=g4
(cd "$oj" && "$py" -B -m pipeline.train_infer.sft_audit \
  "$bench/b16-on/sft_audit.pt" "$bench/$trial/sft_audit.pt") &&
"$py" -B - "$bench/b16-on" "$bench/$trial" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

for location in sys.argv[1:]:
    run = Path(location)
    rows = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    assert len(rows) == 32 and summary["status"] == "complete"
    durations = [right["elapsed_seconds"] - left["elapsed_seconds"]
                 for left, right in zip(rows[7:], rows[8:])]
    mean = statistics.mean(durations)
    p90 = sorted(durations)[math.ceil(0.9 * len(durations)) - 1]
    peak = max(row["peak_gpu_gib"] for row in rows)
    reserved = max(row["peak_gpu_reserved_gib"] for row in rows)
    hours = (6040 * mean + summary["startup_seconds"] + summary["post_train_seconds"]) / 3600
    print(f"{run.name}: mean={mean:.2f}s/step p90={p90:.2f}s/step "
          f"allocated={peak:.2f}GiB reserved={reserved:.2f}GiB estimated={hours:.1f}h")
PY
```

估算使用短测的加载和收尾时长。正式使用 30 条校准记录，另留出校准和排队时间。若四卡档通过且达到目标步速，可用下列命令提交 6,040 步。若单卡档达标，使用实测通过的卡数、批量、检查点模式和 Python 环境。

```bash
run="$oj/runs/minionerec/ml1m-sft-fast-$(date +%Y%m%d-%H%M%S)"
full_job=$(submit_sft_speed jev-sft-ml1m-fast "$run/sft" 4 64 on 7-00:00:00 \
  --steps 6040 --warmup-steps 120 --calibration-rows 30) &&
printf 'SFT 作业 %s，输出 %s\n' "$full_job" "$run/sft"
```

查看正式进度时，使用提交回显中的作业号和输出目录。`training.jsonl` 的最后一行记录已完成步数，`summary.json` 在校准、保存和重载检查通过后生成。

```bash
sacct -j "${full_job%%;*}" --format=JobID,State,Elapsed,ExitCode
tail -n 1 "$run/sft/training.jsonl"
```

把该轮的 `$run/sft/checkpoint` 传给原有单卡 GSPO 提交命令，继续使用既定的全量验证、选模和全量测试流程。

## 单卡直接运行前缀复用 SFT

`--training-scorer prefix-all` 让同一位用户的多个有效 SID 层共享一次前缀计算。各层分别归一化并计算损失，后缀梯度加总后传回前缀。能放入一个后缀批次的层直接反传。当共同前缀太短，或预计处理的 token 数未减少时，该样本沿用原先的 `--training-scorer prefix` 逻辑，仍给第一层复用前缀。以 `--prefix-candidate-batch 32` 单独设置后缀批量，完整候选序列维持 `--candidate-batch 16`。`--gradient-checkpointing partial` 给常规分块评分保留四分之三 decoder 层的梯度检查点。BF16 前缀路径已观察到梯度差异，正式指标以同一全量评估协议核对。

### 加速内核环境

模型自动加载可用的内核。可查看作业 `run.json` 的 `linear_attention_kernels` 和 `causal_conv1d_kernels`。如果记录了 `torch_chunk_gated_delta_rule` 或 `torch_conv1d`，可在继承 `openjev_sid` 的独立 venv 补齐对应依赖。以下官方卷积 wheel 对应服务器的 Linux x86_64、Python 3.11、PyTorch 2.8 和 CUDA 12，命令按实际 PyTorch ABI 拼接文件名。

```bash
base_py="$HOME/anaconda3/envs/openjev_sid/bin/python"
fast_env="$HOME/LJY/venvs/openjev_sid_fast"
fast_py="$fast_env/bin/python"

"$base_py" -m venv --system-site-packages "$fast_env" &&
"$fast_py" -m pip install --no-deps --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'triton==3.4.0' 'fla-core==0.5.2' 'flash-linear-attention==0.5.2' &&
abi=$("$fast_py" -c 'import platform, sys, torch
assert sys.version_info[:2] == (3, 11) and platform.system() == "Linux" and platform.machine() == "x86_64"
assert torch.__version__.split(".")[:2] == ["2", "8"] and torch.version.cuda.split(".")[0] == "12"
print(str(torch.compiled_with_cxx11_abi()).upper())') &&
"$fast_py" -m pip install --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu12torch2.8cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"
```

### 正式提交

以下命令提交一个新的单卡作业，使用独立输出目录。`py` 默认沿用原环境，完成独立环境安装后，可改用 `$HOME/LJY/venvs/openjev_sid_fast/bin/python`。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
oj="$root/openjev_generative_rec"
py="$HOME/anaconda3/envs/openjev_sid/bin/python"
run="$oj/runs/minionerec/ml1m-prefix-all-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$root/comparison/logs"

job=$(env -u SBATCH_GET_USER_ENV -u OPENJEV_REFERENCE_KERNELS TOKENIZERS_PARALLELISM=false \
  sbatch --parsable --export=ALL --job-name=jev-sft-prefix-all --time=7-00:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:4090:1 \
  --output="$root/comparison/logs/jev-sft-prefix-all-%j.log" \
  "$root/comparison/gpu.sbatch" "$oj" "$py" -m pipeline.train_infer.train_lora \
  --objective sft --sft-recipe minionerec --sft-all-layers --validation-users 0 \
  --sid-encoding layered --history-memory on --training-scorer prefix-all \
  --candidate-batch 16 --prefix-candidate-batch 32 --gradient-checkpointing partial \
  --data "$oj/runs/minionerec/ml1m-jev-full-v1/model-data/sequences" \
  --comparison-data "$root/comparison/data/ml1m" --base-model "$oj/models/qwen3.5-2b" \
  --checkpoint "$oj/models/openjev-2b/package/checkpoint" \
  --steps 6040 --seed 42 --accumulation 1 --warmup-steps 120 \
  --lr 2e-5 --head-lr 1e-4 --sid-lr 1e-4 --calibration-rows 30 --calibration-seed 43 \
  --output "$run/sft") && printf '作业 %s\n输出 %s\n' "${job%%;*}" "$run/sft"
```

如短后缀批量 32 遇到显存不足，将 `--prefix-candidate-batch` 改为 16 并使用新的输出目录重新提交。`training.jsonl` 中的 `prefix_usage.choices_by_layer` 记录每层累计使用前缀的候选组数，`prefix_usage.prefills` 记录前缀实际计算次数。

新版运行约 100 步后，可用下面的命令比较新旧作业最近 50 步的速度与剩余时间。重新登录时把 `new_job` 设为新版提交回显的作业号。估算仅包含剩余训练步。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
new_job="${job%%;*}"
sacct -X -j "246908,$new_job" --format=JobID,State,Elapsed,ExitCode
"$HOME/anaconda3/envs/openjev_sid/bin/python" -B - \
  "$root/comparison/logs/jev-sft-prefix-246908.log" \
  "$root/comparison/logs/jev-sft-prefix-all-$new_job.log" <<'PY'
import json
import sys
from collections import deque
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    if not path.is_file():
        print(f"{path.name} 尚无日志")
        continue
    recent = deque(maxlen=51)
    with path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            if line.startswith('{"step":') and line.endswith("\n"):
                recent.append(json.loads(line))
    if len(recent) < 2:
        print(f"{path.name} 尚无足够步数")
        continue
    first, last = recent[0], recent[-1]
    count = last["step"] - first["step"]
    rate = (last["elapsed_seconds"] - first["elapsed_seconds"]) / count
    hours = (6040 - last["step"]) * rate / 3600
    print(f"{path.name} 已完成 {last['step']}/6040 步，最近 {count} 步 {rate:.2f} 秒/步，预计剩余 {hours:.1f} 小时")
    print("累计各层复用组数", last.get("prefix_usage", {}).get("choices_by_layer", {}))
PY
```

训练结束后，使用输出目录里的 `checkpoint` 进入原单卡 GSPO 和评估流程。
