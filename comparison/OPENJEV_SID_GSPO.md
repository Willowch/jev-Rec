# OpenJev 分层 SID 的 GSPO 后训练

ML‑1M 从作业 `247591` 的全量验证最佳权重出发。程序从 `selection.json` 确认权重，读取 SFT 运行时的数据、SID 目录和解码配置。初版 GSPO 对应作业 `247772`。每个训练用户采样 15 条合法 SID，再加入一条真实训练目标。16 条共同计算 MiniOneRec 完整命中奖励和组内优势。15 条自然采样轨迹使用 GSPO 策略损失，强制目标使用 `0.1 × NLL`。

## 解压和初始化

把 `openjev-sid-gspo-rollback-v1.tar.gz` 上传到 `$HOME/LJY/`，再执行下面的命令。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
tar -tzf "$HOME/LJY/openjev-sid-gspo-rollback-v1.tar.gz" &&
tar -xzf "$HOME/LJY/openjev-sid-gspo-rollback-v1.tar.gz" -C "$root"
```

每次重登后先执行这一段。沿用 SFT 的 `openjev_sid` 环境和单卡 4090 Slurm 脚本。

```bash
root="$HOME/LJY/project/openjev_comparison_20261122"
oj="$root/openjev_generative_rec"
py="$HOME/anaconda3/envs/openjev_sid/bin/python"
sft="$oj/runs/minionerec/ml1m-sid-published-v1"
smoke="$oj/runs/minionerec/ml1m-sid-gspo-smoke-v1"
run="$oj/runs/minionerec/ml1m-sid-gspo512-v1"
logs="$root/comparison/logs"

submit_gspo() {
  env -u SBATCH_GET_USER_ENV -u JEV_LOAD_4BIT -u JEV_LOAD_8BIT -u JEV_TORCH_DTYPE \
    -u JEV_DEVICE_MAP -u OPENJEV_REFERENCE_KERNELS -u OPENJEV_COMPACT_DELTA_RULE \
    -u OPENJEV_CPU_TOKEN_EMBEDDINGS -u OPENJEV_RECOMPUTE_PREFIX TOKENIZERS_PARALLELISM=false \
    sbatch --parsable --export=ALL --job-name="$1" --time="$2" \
    --output="$logs/%x-%j.log" "$root/comparison/gpu.sbatch" "$oj" "$py" \
    -m pipeline.train_infer.sid_gspo "$3" --sft-run "$sft" --output "$4"
}
```

## 单卡短测 16 组

短测后训练 16 组，用于确认奖励、梯度、显存和耗时。`sampled_hit_rate` 只统计自然采样的轨迹。

```bash
test -s "$sft/selection.json" && mkdir -p "$smoke" "$logs" &&
job=$(submit_gspo jev-sid-gspo-smoke 01:00:00 smoke "$smoke") &&
printf '%s\n' "${job%%;*}" | tee "$smoke/job-id.txt"
```

```bash
job=$(cat "$smoke/job-id.txt")
sacct -X -j "$job" --format=JobID,State,Elapsed,ExitCode
if [[ -s "$smoke/training-summary.json" ]]; then
  cat "$smoke/training-summary.json"
else
  tail -n 10 "$logs/jev-sid-gspo-smoke-$job.log"
fi
```

短测成功后确认 `groups` 为 16 且 `policy_gradient_groups` 大于 0。每组含强制目标，自然命中率仍可能为 0。

## 正式训练 512 组

正式作业从同一份最佳 SFT 权重加载。每次先采集 4 组，再逐组更新。完成第 256 和 512 组后各保存一份权重并进行全量验证，与纯 SFT 比较后只测试胜出模型。`12:00:00` 是预留的作业上限。重新训练时，把 `run` 指向新的目录。已有初版结果可在 `ml1m-sid-gspo512-v1` 查看。

```bash
state=$(sacct -X -j "$(cat "$smoke/job-id.txt")" -n -P --format=State)
test "$state" = COMPLETED && mkdir -p "$run" "$logs" &&
job=$(submit_gspo jev-sid-gspo512 12:00:00 run "$run") &&
printf '%s\n' "${job%%;*}" | tee "$run/job-id.txt"
```

```bash
job=$(cat "$run/job-id.txt")
sacct -X -j "$job" --format=JobID,State,Elapsed,Timelimit,ExitCode
tail -n 12 "$logs/jev-sid-gspo512-$job.log"
if [[ -s "$run/evaluation.json" ]]; then
  grep -E '^  "(objective|selected_gspo_step|source_sft_epoch|users|Hits@|Recall@|NDCG@)' \
    "$run/evaluation.json"
fi
```

`training.jsonl` 记录每组采样的 15 条 SID、强制目标、奖励、优势、策略梯度和裁剪比例。`selection.json` 保存三份权重的验证分数与胜出步骤。`evaluation.json` 的 `selected_gspo_step` 为 0、256 或 512，0 表示纯 SFT 胜出。只有纯 SFT 胜出且权重、协议、解码配置与原结果一致时，程序才复用已有的正式测试。

如果训练已完成并生成 `training-summary.json`，但后续验证或测试中断，等旧作业退出后重新执行正式提交段即可补跑。已发布的验证结果和测试分片会经过核验再复用。如果训练中断且仍缺 `training-summary.json`，使用新的正式输出目录重新训练。

Industrial 可在新版 SFT 完成后运行 `bash "$root/comparison/run_industrial_sid.sh" gspo-smoke`。短测完成后运行 `bash "$root/comparison/run_industrial_sid.sh" gspo`。
