# OpenJev 的 Amazon 2018 实验

`Industrial_and_Scientific` 和 `Office_Products` 共用 [五模型实验的数据准备与评估协议](../../comparison/README.md)。

`pipeline.amazon_experiment` 提供 `prepare`、`train`、`recommend` 和 `evaluate`。在类别参数和阶段子命令之间传入 `--comparison-data` 选择共享数据。模型侧准备只发布 Qwen 向量、SID 和 manifest。评估可用 `--cohort` 固定测试名单。
