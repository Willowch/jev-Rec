#!/bin/bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
oj="$root/openjev_generative_rec"
py="$HOME/anaconda3/envs/openjev_sid/bin/python"
dataset=Industrial_and_Scientific
data="$root/comparison/data/$dataset"
cohort="$root/comparison/results/cohorts/$dataset-full.json"
runs="$oj/runs/minionerec"
sft="${OPENJEV_INDUSTRIAL_SFT_RUN:-$runs/industrial-sid-published-v1}"
smoke="${OPENJEV_INDUSTRIAL_GSPO_SMOKE:-$runs/industrial-sid-gspo-smoke-v1}"
gspo="${OPENJEV_INDUSTRIAL_GSPO_RUN:-$runs/industrial-sid-gspo512-v1}"
logs="$root/comparison/logs"

submit() {
  local name=$1 walltime=$2 out=$3 job
  shift 3
  if [[ -e "$out/job-id.txt" || -e "$out/run.json" ]]; then
    printf '输出目录已关联一次运行 %s\n' "$out" >&2
    return 1
  fi
  mkdir -p "$out" "$logs"
  job=$(env -u SBATCH_GET_USER_ENV -u JEV_LOAD_4BIT -u JEV_LOAD_8BIT -u JEV_TORCH_DTYPE \
    -u JEV_DEVICE_MAP -u OPENJEV_REFERENCE_KERNELS -u OPENJEV_COMPACT_DELTA_RULE \
    -u OPENJEV_CPU_TOKEN_EMBEDDINGS -u OPENJEV_RECOMPUTE_PREFIX TOKENIZERS_PARALLELISM=false \
    sbatch --parsable --export=ALL --job-name="$name" --time="$walltime" \
    --output="$logs/%x-%j.log" "$root/comparison/gpu.sbatch" "$oj" "$py" -m "$@" --output "$out")
  printf '%s\n' "${job%%;*}" > "$out/job-id.txt"
  printf '作业号 %s\n日志 %s/%s-%s.log\n输出目录 %s\n' "${job%%;*}" "$logs" "$name" "${job%%;*}" "$out"
}

case "${1:-sft}" in
  sft)
    sid=$("$py" -B - "$oj" "$data" "${OPENJEV_INDUSTRIAL_SID_DATA:-}" <<'PY'
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

oj, shared = map(Path, sys.argv[1:3])
override = sys.argv[3]
prepared = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
identity = prepared["identity"]

def sid_fingerprint(path):
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("dataset") != identity["dataset"]
                or manifest.get("comparison_identity", identity) != identity
                or manifest.get("split_protocol") != identity["split_protocol"]
                or manifest.get("ratings_sha256") != prepared["source"]["ratings_sha256"]
                or (manifest.get("users"), manifest.get("items")) != (prepared["users"], prepared["items"])
                or manifest.get("max_history") != identity["max_history"]):
            return None
        digest = hashlib.sha256((path / "item_semantic_ids.jsonl").read_bytes()).hexdigest()
        return digest if digest == manifest.get("semantic_ids_sha256") else None
    except (OSError, ValueError):
        return None

if override:
    selected = Path(override).resolve()
    if not sid_fingerprint(selected):
        raise SystemExit(f"指定的 SID 目录与共享 Industrial 数据不符。{selected}")
    print(selected)
    sys.exit(0)

try:
    submitted = subprocess.run(["sacct", "-X", "-n", "-P", "-j", "246985", "-o", "SubmitLine%4096"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    words = shlex.split(submitted)
    args = words[words.index("pipeline.train_infer.train_lora") + 1:]
    selected = (oj / args[args.index("--data") + 1]).resolve()
    if sid_fingerprint(selected):
        print(selected)
        sys.exit(0)
except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
    pass

matches = {}
for base in (oj / "runs", oj / "data", shared):
    for file in sorted(base.rglob("item_semantic_ids.jsonl")):
        digest = sid_fingerprint(file.parent)
        if digest:
            matches.setdefault(digest, file.parent.resolve())
if len(matches) != 1:
    for match in matches.values():
        print(match, file=sys.stderr)
    raise SystemExit("未能唯一确定已生成的 Industrial SID。请用 OPENJEV_INDUSTRIAL_SID_DATA 指定目录。")
print(next(iter(matches.values())))
PY
)
    source="$oj/models/openjev-2b/package/checkpoint"
    test -s "$source/model.json" && test -s "$cohort" || {
      printf '缺少发布版 checkpoint 或 Industrial 全量测试名单\n' >&2
      exit 1
    }
    printf 'Industrial SID 目录 %s\n' "$sid"
    submit jev-sid-industrial-sft 1-12:00:00 "$sft" pipeline.train_infer.sid_decoder run \
      --dataset "$dataset" --initialization published --source-checkpoint "$source" \
      --sid-data "$sid" --comparison-data "$data" --cohort "$cohort" \
      --base-model "$oj/models/qwen3.5-2b" --epochs 10 --min-epochs 5 --patience 3 \
      --train-batch 4 --decode-backend recompute --decode-batch 1
    ;;
  gspo-smoke)
    test -s "$sft/selection.json" || { printf '请先完成新版 Industrial SFT\n' >&2; exit 1; }
    submit jev-sid-industrial-gspo-smoke 01:00:00 "$smoke" pipeline.train_infer.sid_gspo smoke --sft-run "$sft"
    ;;
  gspo)
    test -s "$sft/selection.json" || { printf '请先完成新版 Industrial SFT\n' >&2; exit 1; }
    "$py" -B - "$smoke/training-summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("请先完成 Industrial GSPO 短测")
result = json.loads(path.read_text(encoding="utf-8"))
if result["groups"] != 16 or result["trajectories_per_group"] != 16 or result["policy_gradient_groups"] <= 0:
    raise SystemExit("Industrial GSPO 短测未通过")
PY
    submit jev-sid-industrial-gspo512 12:00:00 "$gspo" pipeline.train_infer.sid_gspo run --sft-run "$sft"
    ;;
  *)
    printf '用法 bash run_industrial_sid.sh [sft|gspo-smoke|gspo]\n' >&2
    exit 2
    ;;
esac
