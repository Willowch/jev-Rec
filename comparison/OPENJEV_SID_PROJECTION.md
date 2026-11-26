# OpenJev SID 投影训练 · 发布版初始化与 ML‑1M 单卡

这版用发布版 OpenJev 的 LoRA 和预训练 Qwen 初始化，复用已有 ML‑1M SID 映射，并重新初始化 SID 向量与分层评分头。Qwen 每批用户执行一次训练前向，输出头同时给各层 SID 候选打分。每轮用全部 6,040 位用户的 NDCG@10 选权重，正式测试仍用受 SID 目录约束的 beam search，覆盖 6,040 人并过滤已见商品。

## 同步和初始化

上传 `openjev-sid-projection-from-start-update.tar.gz` 到服务器的 `$HOME/LJY/`，再执行下面的解压命令。压缩包包含批量 SID 训练的完整代码。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
tar -xzf "$HOME/LJY/openjev-sid-projection-from-start-update.tar.gz" -C "$root"
```

每次重登后，先粘贴下面的初始化段。它使用 `openjev_sid` 环境、发布版 OpenJev checkpoint 和已生成的 SID 文件。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
oj="$root/openjev_generative_rec"
py="$HOME/anaconda3/envs/openjev_sid/bin/python"
bench="$oj/runs/minionerec/ml1m-sid-published-bench-v1"
run="$oj/runs/minionerec/ml1m-sid-published-v1"
logs="$root/comparison/logs"

inputs=(--dataset ml1m --initialization published
  --sid-data "$oj/runs/minionerec/ml1m-jev-full-v1/model-data/sequences"
  --source-checkpoint "$oj/models/openjev-2b/package/checkpoint"
  --comparison-data "$root/comparison/data/ml1m"
  --cohort "$root/comparison/results/cohorts/ml1m-full.json"
  --base-model "$oj/models/qwen3.5-2b")

submit_sid() {
  env -u SBATCH_GET_USER_ENV -u JEV_LOAD_4BIT -u JEV_LOAD_8BIT -u JEV_TORCH_DTYPE \
    -u JEV_DEVICE_MAP -u OPENJEV_REFERENCE_KERNELS -u OPENJEV_COMPACT_DELTA_RULE \
    -u OPENJEV_CPU_TOKEN_EMBEDDINGS -u OPENJEV_RECOMPUTE_PREFIX TOKENIZERS_PARALLELISM=false \
    sbatch --parsable --export=ALL --job-name="$1" --time="$2" \
    --output="$logs/%x-%j.log" "$root/comparison/gpu.sbatch" "$oj" "$py" \
    -m pipeline.train_infer.sid_decoder "${@:3}"
}
```

## 先测一轮的成本

512 位固定种子训练用户依次测试新版的 4、8、16 用户批量，再测老版 `prefix-all` 的单用户批量。两条路径都从发布版权重开始。短测记录每用户耗时、峰值显存和折算的全量轮次耗时，再用 32 条验证历史估算 beam 10 和 beam 20 的耗时。

```bash
mkdir -p "$logs" "$bench" &&
job=$(submit_sid jev-sid-published-bench 02:00:00 bench "${inputs[@]}" --output "$bench") &&
printf '%s\n' "${job%%;*}" | tee "$bench/job-id.txt"
```

```bash
job=$(cat "$bench/job-id.txt")
sacct -X -j "$job" --format=JobID,State,Elapsed,ExitCode
grep -E '"event": "(bench_progress|bench_result|bench_decoding|cache_audit)"' \
  "$logs/jev-sid-published-bench-$job.log" | tail -n 12
test ! -s "$bench/benchmark.json" || cat "$bench/benchmark.json"
```

`estimated_epoch_hours` 只估算训练。安排作业时另外加上每次 beam 10 验证和最终一次 beam 20 测试。显存数据来自 512 人短测，长历史批次的用量还需在正式运行中观察。

## 正式训练、选模和测试

正式训练**最多 10 轮，至少 5 轮**，每轮完整验证一次。验证 NDCG@10 严格提高时才保存新的最佳权重。同分保留较早的一轮，连续 3 次验证无提升后可早停。下面的命令选择短测里最快的批量，并按 10 轮训练、10 次验证和一次测试估计 Slurm 时限，额外留出 25% 时间。若最快批量在短测中已接近显存上限，可用 `batch_override` 指定一个较小且通过的批量。

```bash
batch_override=""
settings=$("$py" -B - "$bench/benchmark.json" "$batch_override" <<'PY'
import json
import math
import sys
from pathlib import Path

bench = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
train = bench["fastest_projection"]
if sys.argv[2]:
    requested = int(sys.argv[2])
    train = next((row for row in bench["results"] if row["scorer"] == "sid-projection"
                  and row["status"] == "complete" and row["batch_size"] == requested), None)
    if train is None:
        raise SystemExit(f"批量 {requested} 未通过短测")
valid = bench["decoding"]["10"]["estimated_full_hours"]
test = bench["decoding"]["20"]["estimated_full_hours"]
minutes = max(60, math.ceil(1.25 * (10 * (train["estimated_epoch_hours"] + valid) + test) * 60))
days, rest = divmod(minutes, 1440)
hours, minutes = divmod(rest, 60)
print(train["batch_size"], f"{days}-{hours:02d}:{minutes:02d}:00")
PY
) &&
read -r batch walltime <<< "$settings" &&
mkdir -p "$logs" "$run" &&
printf '用户批量 %s，作业时限 %s\n' "$batch" "$walltime" &&
job=$(submit_sid jev-sid-published "$walltime" run "${inputs[@]}" \
  --epochs 10 --min-epochs 5 --patience 3 --train-batch "$batch" --output "$run") &&
printf '%s\n' "${job%%;*}" | tee "$run/job-id.txt"
```

```bash
job=$(cat "$run/job-id.txt")
sacct -X -j "$job" --format=JobID,State,Elapsed,ExitCode
if [[ -s "$run/evaluation.json" ]]; then
  tail -n 23 "$run/evaluation.json"
else
  tail -n 15 "$logs/jev-sid-published-$job.log"
fi
```

`training.jsonl` 记录每轮训练及验证耗时。`selection.json` 记录实际训练轮数、停止原因和最佳轮次。`evaluation.json` 保存逐用户榜单和 Hits、Recall、NDCG @5、@10、@20。共享汇总入口将本结果标为 `OpenJev (SID projection)`。

如果 `selection.json` 已生成、作业在测试阶段中断，先确认原作业已退出，再用下面的命令继续同一目录的剩余分片。

```bash
job=$(submit_sid jev-sid-published 04:00:00 test "${inputs[@]}" --output "$run") &&
printf '%s\n' "${job%%;*}" | tee "$run/job-id.txt"
```

如果训练阶段中断，从新的输出目录重新开始。如果缓存分数核验报错，新一轮可显式用 `--decode-backend recompute` 重测。验证和测试要使用相同的解码配置。
