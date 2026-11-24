# OpenJev 全目录评分 · ML‑1M

这轮实验复用已完成的 ML‑1M 纯 SFT 权重，训练一个独立的全目录评分头。选模用 6,040 位验证用户的 NDCG@10，最终只评估一次完整测试名单。

## 同步与提交

把 `openjev-full-catalog-update.tar.gz` 上传至服务器的 `$HOME/LJY/`，在服务器终端执行以下命令。输出目录使用固定名称，重复提交时从已完成的阶段继续。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
archive="$HOME/LJY/openjev-full-catalog-update.tar.gz"
tar -tzf "$archive" &&
tar -xzf "$archive" -C "$root"
```

```bash
(
  set -euo pipefail
  root="$HOME/LJY/project/openjev_comparison_20261122"
  oj="$root/openjev_generative_rec"
  py="$HOME/anaconda3/envs/openjev_sid/bin/python"
  run="$oj/runs/minionerec/ml1m-full-catalog-v1"
  sft="$oj/runs/minionerec/ml1m-prefix-all-20261123-141700/sft/checkpoint"
  sid="$oj/runs/minionerec/ml1m-jev-full-v1/model-data/sequences"
  data="$root/comparison/data/ml1m"
  cohort="$root/comparison/results/cohorts/ml1m-full.json"
  base="$oj/models/qwen3.5-2b"

  for path in "$sft/recommendation.json" "$sft/../run.json" "$sid/item_semantic_ids.jsonl" \
              "$data/manifest.json" "$cohort" "$base/config.json"; do
    [[ -s "$path" ]] || { printf '缺少输入 %s\n' "$path" >&2; exit 1; }
  done
  mkdir -p "$run" "$root/comparison/logs"
  cd "$oj"
  "$py" -B -m pipeline.train_infer.full_catalog --help >/dev/null

  id=$(env -u SBATCH_GET_USER_ENV -u JEV_LOAD_4BIT -u JEV_LOAD_8BIT \
    -u JEV_TORCH_DTYPE -u JEV_DEVICE_MAP -u OPENJEV_REFERENCE_KERNELS \
    -u OPENJEV_COMPACT_DELTA_RULE -u OPENJEV_CPU_TOKEN_EMBEDDINGS \
    sbatch --parsable --export=ALL --job-name=jev-ml1m-catalog --time=20:00:00 \
    --output="$root/comparison/logs/jev-ml1m-catalog-%j.log" \
    "$root/comparison/gpu.sbatch" "$oj" "$py" \
    -m pipeline.train_infer.full_catalog run \
    --dataset ml1m --comparison-data "$data" --cohort "$cohort" \
    --sid-data "$sid" --sft-checkpoint "$sft" --base-model "$base" --output "$run")
  printf '%s\n' "${id%%;*}" > "$run/job-id.txt"
  printf '作业号 %s\n结果目录 %s\n' "${id%%;*}" "$run"
)
```

默认用一张 4090 完成整个流程。特征批量从 32 开始，显存不足时自动减半。前 128 位训练用户的特征抽取写出首条 `feature_block`，其中的 `seconds_per_user` 可用来粗估三段共 18,120 位用户的抽取时间。作业中断后先确认旧作业已退出，再重新执行同一提交段。入口会检查现有特征分块、训练状态、已选权重和测试分片的指纹。

## 查看进度

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
run="$root/openjev_generative_rec/runs/minionerec/ml1m-full-catalog-v1"
id=$(cat "$run/job-id.txt")
sacct -X -j "$id" --format=JobID,State,Elapsed,Timelimit,ExitCode
grep -E '"event": "(feature_block|reduced_feature_batch|train_epoch|validation|checkpoint_selected|test_block|evaluation_complete)"' \
  "$root/comparison/logs/jev-ml1m-catalog-$id.log" | tail -n 12
```

最终的 6,040 人逐用户结果和 @5、@10、@20 指标保存在 `evaluation.json`。验证网格和协同分支消融保存在 `selected/validation.json`，实际测试权重记录在 `selected/selection.json`。

## 与四个 baseline 并表

用之前生成 ML‑1M 四模型表时的同四份结果文件。如果每个模型目录只有一份 `evaluation.json`，以下命令可直接定位。存在多个实验时，按原四模型表使用的运行目录填写四个路径。

```bash
(
  set -euo pipefail
  root="$HOME/LJY/project/openjev_comparison_20261122"
  oj="$root/openjev_generative_rec"
  py="$HOME/anaconda3/envs/openjev_sid/bin/python"
  run="$oj/runs/minionerec/ml1m-full-catalog-v1"
  results=()
  for dir in "$root/SASRec/runs/ml1m" "$root/BERT4Rec/runs/ml1m" \
             "$root/TIGER/4_runs/comparison/ml1m" "$root/LETTER-TIGER/runs/ml1m"; do
    [[ -d "$dir" ]] || { printf '找不到模型结果目录 %s\n' "$dir" >&2; exit 1; }
    mapfile -d '' -t matches < <(find "$dir" -type f -name evaluation.json -print0)
    if (( ${#matches[@]} != 1 )); then
      printf '请为 %s 选定原对照表的 evaluation.json\n' "$dir" >&2
      printf '%s\n' "${matches[@]}" >&2
      exit 1
    fi
    results+=("${matches[0]}")
  done
  cd "$oj"
  env -u JEV_LOAD_4BIT -u JEV_LOAD_8BIT -u JEV_TORCH_DTYPE -u JEV_DEVICE_MAP \
    -u OPENJEV_REFERENCE_KERNELS -u OPENJEV_COMPACT_DELTA_RULE -u OPENJEV_CPU_TOKEN_EMBEDDINGS \
    "$py" -B -m pipeline.train_infer.full_catalog merge --device cpu \
    --comparison-data "$root/comparison/data/ml1m" \
    --cohort "$root/comparison/results/cohorts/ml1m-full.json" \
    --sid-data "$oj/runs/minionerec/ml1m-jev-full-v1/model-data/sequences" \
    --sft-checkpoint "$oj/runs/minionerec/ml1m-prefix-all-20261123-141700/sft/checkpoint" \
    --base-model "$oj/models/qwen3.5-2b" --output "$run" \
    --baseline-results "${results[@]}"
)
```

`comparison.md` 保存五模型表。`comparison.json` 保存最高 baseline 的 NDCG@10、OpenJev 的 NDCG@10 和两者差值。合并可在登录节点执行。
