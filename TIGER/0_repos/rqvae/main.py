import argparse
import random
import torch
import numpy as np
from time import time
# [旧实验记录，已停用]
# import logging
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer

# [新增] 兼容从任意工作目录直接启动，并复用两阶段的实验记录模块。
import sys  # 访问模块搜索路径，使入口能导入共用记录器。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 0_repos 的绝对路径 str 放到模块搜索路径首位。
from experiment_tracking import PROJECT_ROOT, TrackingRun, add_tracking_arguments  # 复用项目根目录、Run 生命周期管理和命令行记录选项。

# 流程：定义模型/训练参数 → 添加共用 MLflow 选项 → 解析命令行 → 返回参数对象。
# 输入：无显式参数；读取 sys.argv，如 ['main.py', '--dataset', 'Beauty', '--disable_mlflow']。
# 输出：argparse.Namespace，如 args.batch_size=1024、args.num_emb_list=[256, 256, 256]、
#       args.tracking_uri=None；列表参数 layers 为 [层数]，num_emb_list/sk_epsilons 为 [L]。
def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument("--eval_step", type=int, default=50, help="eval step")
    parser.add_argument("--learner", type=str, default="AdamW", help="optimizer")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear", help="scheduler"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=50, help="warmup epochs"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Input data path."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="l2 regularization weight"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.0, help="dropout ratio"
    )
    parser.add_argument(
        "--bn", type=bool, default=False, help="use bn or not"
    )
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument(
        "--kmeans_init", type=bool, default=True, help="use kmeans_init or not"
    )
    parser.add_argument(
        "--kmeans_iters", type=int, default=100, help="max kmeans iters"
    )
    parser.add_argument(
        "--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.003], help="sinkhorn epsilons"
    )
    parser.add_argument(
        "--sk_iters", type=int, default=50, help="max sinkhorn iters"
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="gpu or cpu"
    )

    parser.add_argument(
        "--num_emb_list", type=int, nargs="+", default=[256, 256, 256], help="emb num of every vq"
    )
    parser.add_argument(
        "--e_dim", type=int, default=32, help="vq codebook embedding size"
    )
    parser.add_argument(
        "--quant_loss_weight", type=float, default=1.0, help="vq quantion loss weight"
    )
    parser.add_argument(
        "--beta", type=float, default=0.25, help="Beta for commitment loss"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[512, 256, 128, 64],
        help="hidden sizes of every layer",
    )
    parser.add_argument(
        "--save_limit", type=int, default=5, help="save limit for ckpt"
    )

    # [旧实验记录，已停用]
    # parser.add_argument(
    #     "--ckpt_dir",
    #     type=str,
    #     default="./ckpt/Beauty",
    #     help="please specify output directory for model",
    # )
    # [新增] 可选目录下面继续按 Run ID 隔离权重；种子由命令行记录。
    parser.add_argument("--ckpt_dir", default=None)  # 可选权重根目录 str/None；训练时还会追加 Run ID。
    parser.add_argument("--seed", type=int, default=2024)  # 随机种子 int 标量；随后随配置和 checkpoint 保存。
    add_tracking_arguments(parser)  # 加入 dataset、服务地址、实验组名、Run 名、输出根目录和离线开关。
    # [旧实验记录，已停用]
    # parser.add_argument(
    #     "--log_dir",
    #     type=str,
    #     help="log directory",
    # )

    return parser.parse_args()  # 得到 Namespace，供训练和实验参数记录共用。

# 流程：解析参数并设种子 → 检查配置/设备 → 创建 Run 和输出目录 → 保存复现信息 → 构建并训练
#       RQ-VAE → 取最小目录碰撞率权重 → 上传权重、数值摘要和 summary.json → 退出时结束 Run。
# 输入：无显式参数；命令行由 parse_args 读取，args.data_path 指向 embedding 堆叠为 [N, D] 的 parquet。
# 输出：None；保存 checkpoint、配置/环境/源码快照及摘要；每批模型输入 [B, D]，量化索引 [B, L]。
#       N=物品数，D=向量维度，B=当前批大小，L=len(args.num_emb_list)，默认 L=3。
def main():
    args = parse_args()  # 实际运行参数 Namespace，如 args.dataset='Beauty'。
    # [旧实验记录，已停用]
    # seed = 2024
    # [新增] 种子既可配置，也会写入权重和 MLflow 参数。
    seed = args.seed  # int 种子，作为可复现配置记录。
    random.seed(seed)  # 重置 Python 随机状态；记录器创建后也会再重置一次。
    np.random.seed(seed)  # 重置 NumPy 随机状态。
    torch.manual_seed(seed)  # 重置 PyTorch 随机状态。
    torch.cuda.manual_seed_all(seed)  # 重置所有 CUDA 设备的随机状态。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # [旧实验记录，已停用]
    # log_dir = Path(args.log_dir) if args.log_dir else Path(args.ckpt_dir) / "logs"
    # [旧实验记录，已停用]
    # log_dir.mkdir(parents=True, exist_ok=True)

    # [旧实验记录，已停用]
    # run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # [旧实验记录，已停用]
    # log_file = log_dir / f"rqvae_{run_time}.log"

    # [旧实验记录，已停用]
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s | %(levelname)s | %(message)s",
    #     handlers=[
    #         logging.FileHandler(log_file, encoding="utf-8"),
    #         logging.StreamHandler(),
    #     ],
    #     force=True,
    # )

    # [旧实验记录，已停用]
    # logging.info("日志文件：%s", log_file.resolve())
    # [旧实验记录，已停用]
    # logging.info("训练参数：%s", args)

    # [新增] 新 Run 保存完整配置和数据指纹，权重目录不会覆盖其他实验。
    if args.epochs < 1 or args.eval_step < 1 or args.save_limit < 1:  # 建 Run 前先拒绝无法训练/评估/保留权重的配置。
        raise ValueError("epochs, eval_step and save_limit must be positive")  # 配置无效时停止执行；此时尚未建立 Run。
    if args.device.startswith("cuda") and not torch.cuda.is_available():  # 检查所请求的 GPU 是否可用。
        args.device = "cpu"  # 记录实际使用的设备，而非不可用的 CUDA 请求值。
    with TrackingRun(args, "stage1_rqvae") as tracking:  # 创建第一阶段 Run；离开 with 自动登记完成、失败或中断状态。
        args.tracking_uri = tracking.tracking_uri  # 将最终服务地址 str 写回配置，避免记录 None。
        args.experiment_name = tracking.experiment_name  # 将最终实验组名 str 写回，如 'TIGER/Beauty'。
        args.ckpt_dir = str(Path(args.ckpt_dir) / tracking.run_id if args.ckpt_dir  # 指定权重目录时追加唯一 Run ID，避免覆盖其他实验。
                            else tracking.output_dir / "checkpoints")  # 未指定时使用当前 Run 下的 checkpoints；最终为路径 str。
        tracking.record_setup(vars(args), {"embeddings": args.data_path})  # 记录完整配置与输入文件路径/指纹，保存环境、命令和源码快照。
        tracking.params({"selection_metric": "catalog/collision_rate",  # 仅注明模型选择指标是目录碰撞率；实际比较逻辑位于 Trainer.fit。
                         "loss_reduction": "sum_of_batch_means"})  # 仅注明轮损失为各批平均 loss 之和；不改变反向传播或求和方式。
        # [新增] 在记录器初始化后重置随机状态，使记录开关不影响模型初始化。
        random.seed(seed)  # 重置 Python 随机状态；记录器创建后也会再重置一次。
        np.random.seed(seed)  # 重置 NumPy 随机状态。
        torch.manual_seed(seed)  # 重置 PyTorch 随机状态。
        torch.cuda.manual_seed_all(seed)  # 重置所有 CUDA 设备的随机状态。
        data = EmbDataset(args.data_path)  # 物品向量数据集；embeddings 为 [N, D]，单条为 [D]。
        model = RQVAE(
            in_dim=data.dim,
            num_emb_list=args.num_emb_list,
            e_dim=args.e_dim,
            layers=args.layers,
            dropout_prob=args.dropout_prob,
            bn=args.bn,
            loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            beta=args.beta,
            kmeans_init=args.kmeans_init,
            kmeans_iters=args.kmeans_iters,
            sk_epsilons=args.sk_epsilons,
            sk_iters=args.sk_iters,
        )
        # [旧实验记录，已停用]
        # logging.info("模型结构：\n%s", model)
        data_loader = DataLoader(
            data,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        # 将记录器交给训练器；len(data_loader) 是每轮批次数 int。
        trainer = Trainer(args, model, len(data_loader), tracking=tracking)

        device = next(trainer.model.parameters()).device
        # [旧实验记录，已停用]
        # logging.info("模型所在设备：%s", device)
        # [旧实验记录，已停用]
        # if device.type == "cuda":
        #     logging.info("使用的显卡：%s", torch.cuda.get_device_name(device))
        # else:
        #     logging.info("当前使用 CPU")

        best_loss, best_collision_rate = trainer.fit(data_loader)  # 返回两个 float 标量：评估轮次中的最小训练 loss 和最小目录碰撞率。

        # [旧实验记录，已停用]
        # logging.info("Best Loss: %s", best_loss)
        # [旧实验记录，已停用]
        # logging.info("Best Collision Rate: %s", best_collision_rate)

        # [新增] 只上传实际用于生成 SID 的最佳碰撞率权重。
        selected = Path(trainer.ckpt_dir) / trainer.best_collision_ckpt  # 选中权重 Path，指向 best_collision_model.pth。
        tracking.artifact(selected, "checkpoints")  # 将所选权重上传到当前 Run 的 checkpoints 附件目录。
        summary = {"selected_epoch": trainer.best_collision_epoch,  # 摘要 dict 共 4 项；selected_epoch 是从 1 开始的轮号 int。
                   "selected_collision_rate": best_collision_rate,  # 所选权重的目录碰撞率 float 标量，越小越好。
                   "best_training_loss_sum": best_loss,  # 评估轮次中的最小训练 loss float；不保证来自所选碰撞率权重。
                   "checkpoint_path": str(selected)}  # 所选权重的本地路径 str。
        tracking.metrics({key: summary[key] for key in (  # 只提取摘要的 3 个数值字段记录为指标；默认 step=0。
            "selected_epoch", "selected_collision_rate", "best_training_loss_sum")})  # 指标均为标量；字符串路径留在 JSON 摘要中。
        tracking.json_artifact("summary.json", summary)  # 本地保存全部 4 项摘要，并在线上传为附件。
        print(f"Selected checkpoint: {selected}")  # 打印所选 checkpoint 的位置，供后续 SID 导出使用。


if __name__ == "__main__":  # 直接运行脚本才启动训练；导入模块不会创建 Run。
    main()  # 执行第一阶段训练及其记录流程。
