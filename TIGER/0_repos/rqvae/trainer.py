import logging
import numpy as np
import torch
from time import time
from torch import optim
# [新增] 修正原来将 tqdm 模块当函数调用的问题，使训练入口可实际运行。
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from utils import (
    ensure_dir,
    set_color,
    get_local_time,
    delete_file,
)
import os
import heapq

class Trainer(object):

    # 流程：保存训练配置与记录器 → 准备权重目录/最佳值 → 建优化器和调度器 → 将模型放到设备。
    # 输入：self，当前 Trainer；args，Namespace，如 args.lr=0.001、args.ckpt_dir='.../checkpoints'；
    #       model，RQVAE 对象，输入 [B, D]、输出重建 [B, D]/量化 loss 标量/索引 [B, L]；
    #       data_num，int，每轮批次数（不是物品数），如 len(data_loader)=20；
    #       tracking，TrackingRun 或 None，如第一阶段 Run 的记录器；None 时不调用记录接口。
    # 输出：None；配置 self.model/optimizer/scheduler、最佳值和目录；不执行训练。
    # 形状约定：B=当前批大小，D=物品向量维度，L=量化层数，N=全目录物品数。
    def __init__(self, args, model, data_num, tracking=None):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        # [新增] 记录器只接收已有指标，不参与模型前向和优化器更新。
        self.tracking = tracking  # TrackingRun/None，只负责记录已有数值和提供来源元信息。
        self.best_collision_epoch = 0  # 最佳碰撞率轮号 int；0 表示尚未选中，选中后从 1 开始。

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir  # 权重目录 str，入口已按 Run ID 隔离。
        # [旧实验记录，已停用]
        # saved_model_dir = "{}".format(get_local_time())
        # self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        # [新增] 入口已按 Run ID 分配目录，不再创建可能同名的时间戳目录。
        ensure_dir(self.ckpt_dir)  # 确保本地权重目录存在，供后续 checkpoint 写盘。

        self.best_loss = np.inf  # 最小训练 loss 浮点标量；首次评估前为正无穷。
        self.best_collision_rate = np.inf  # 最小目录碰撞率浮点标量；首次评估前为正无穷。
        self.best_loss_ckpt = "best_loss_model.pth"  # 按训练 loss 保存的最佳权重文件名 str。
        self.best_collision_ckpt = "best_collision_model.pth"  # 最终选用并上传的最佳碰撞率权重文件名 str。
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    # 流程：读取优化器名、学习率和权重衰减 → 选择对应优化器 → 返回；未知名称回退到 Adam。
    # 输入：self，已包含 model/lr/learner/weight_decay 的 Trainer，如 learner='AdamW'、lr=0.001。
    # 输出：torch.optim 优化器对象，如 optim.AdamW(...)；内部参数张量形状与模型各层一致。
    def _build_optimizer(self):

        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    # 流程：读取调度类型和预热步数 → 建线性衰减或恒定学习率调度器 → 返回。
    # 输入：self，含 optimizer、warmup_steps/max_steps 的 Trainer，如 warmup_steps=1000（int 步数）。
    # 输出：学习率调度器对象，每批调用 scheduler.step() 更新学习率标量。
    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.warmup_steps
            )

        return lr_scheduler

    # 流程：检查 loss 是否为 NaN → 若是则抛错，否则直接结束。
    # 输入：self，当前 Trainer；loss，单元素 Tensor，形状 []，如 tensor(0.3)。
    # 输出：None；NaN 时抛 ValueError；这里只检查 NaN，不检查 Inf。
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    # 流程：切换训练模式 → 逐批前向/算 loss/反传/裁剪梯度/更新参数和调度器 → 累加批次 loss。
    # 输入：self，当前 Trainer；train_data，DataLoader，每批 float Tensor [B, D]；
    #       epoch_idx，int，从 0 开始的轮号，如 0 表示第 1 轮。
    # 输出：(total_loss, total_recon_loss)，两个 float 标量，分别为各批总 loss/重建 loss 的和；
    #       此处未除以批次数；具体记录由 fit 调用 metrics 完成。
    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(
                out, rq_loss, xs=data
            )
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()  # 从 [] Tensor 取 float，再按批累加；对应 train/loss_sum。
            total_recon_loss += loss_recon.item()  # 按批累加重建平均 loss；对应 train/reconstruction_loss_sum。

        return total_loss, total_recon_loss  # 返回两个 float 总和，不除以批次数或样本数。

    # 流程：关闭梯度并切换评估模式 → 为所有物品生成索引 → 统计不同编码数 → 计算目录碰撞率。
    # 输入：self，当前 Trainer；valid_data，DataLoader，每批 Tensor [B, D]，fit 实际传入训练目录。
    # 输出：float 标量 (N - 不同编码数)/N；中间索引为 [B, L]；这是目录诊断，不是独立验证集得分。
    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
        )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (  # 目录碰撞率 float，下面用重复编码的物品数量占比计算。
            num_sample - len(list(indices_set))  # 物品总数减去不同编码数，得到超出唯一编码的物品数。
        ) / num_sample  # 除以物品总数；记录为 catalog/collision_rate。

        return collision_rate  # 返回单个 float 诊断值，越小表示编码碰撞越少。

    # 流程：决定权重文件名 → 打包参数/模型/优化器和最佳值 → 附加 Run 来源元信息 → 写盘返回路径。
    # 输入：self，当前 Trainer；epoch，int，零起始轮号，如 0；collision_rate，float 标量，默认 1，
    #       用于默认文件名；ckpt_file，str/None，如 'best_collision_model.pth'，指定时直接使用该文件名。
    # 输出：str 权重路径，如 '.../checkpoints/best_collision_model.pth'；state 为 dict，
    #       state_dict 为 {参数名: Tensor}（各层形状不同），tracking 来源字典含 4 个普通字段。
    #       这里只保存本地文件；上传所选权重由 main 完成。
    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(
                self.ckpt_dir, "epoch_%d_collision_%.4f_model.pth" % (epoch, collision_rate)
            )
        )
        state = {  # checkpoint 字典，含普通配置和多种形状的模型/优化器张量。
            "args": self.args,  # Namespace 完整配置，包含实际使用的 MLflow 地址和实验组。
            "epoch": epoch,  # int 零起始轮号；导出清单时会加 1。
            "best_loss": self.best_loss,  # 当前已知的最小训练 loss 标量。
            "best_collision_rate": self.best_collision_rate,  # 当前已知的最小目录碰撞率标量。
            "state_dict": self.model.state_dict(),  # 模型权重 dict[str, Tensor]，每个张量形状由对应层决定。
            "optimizer": self.optimizer.state_dict(),  # 优化器状态 dict，包含参数组及可能的动量张量。
        }
        # [新增] 导出 SID 时从权重内读取来源 Run 和输入向量指纹。
        if self.tracking:  # 传入记录器才调用记录接口；离线记录器也会进入，由内部决定是否联网。
            state["tracking"] = self.tracking.checkpoint_metadata()  # 加入 4 项来源字典：Run ID、服务地址、实验组、向量指纹。
        torch.save(state, ckpt_path, pickle_protocol=4)  # 将来源信息与训练状态一起存入本地 .pth 文件。

        # [旧实验记录，已停用]
        # self.logger.info(
        #     set_color("Saving current", "blue") + f": {ckpt_path}"
        # )

        return ckpt_path  # 返回路径 str，供保留/删除权重及最终上传使用。

    # 流程：计算耗时 → 拼接轮号、总 loss 和重建 loss → 返回控制台文本。
    # 输入：self，当前 Trainer；epoch_idx，int 零起始轮号，如 0；s_time/e_time，float 秒时间戳；
    #       loss/recon_loss，float 标量，为该轮各批总 loss/重建 loss 之和，如 2.3、1.7。
    # 输出：str，如 'epoch 0 training [time: 1.20s, train loss: 2.3000, reconstruction loss: 1.7000]'。
    #       此文本本身不写入 MLflow，指标由 fit 单独记录。
    def _generate_train_loss_output(
        self, epoch_idx, s_time, e_time, loss, recon_loss
    ):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (
            epoch_idx,
            e_time - s_time,
        )
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output += ", "
        train_loss_output += (
            set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        )
        return train_loss_output + "]"

    # 流程：逐轮训练并记录 loss → 定期及末轮计算/记录目录碰撞率 → 更新最佳权重 → 维护最近/较优
    #       权重保留集合 → 返回最优数值；真正的最小碰撞率选择由本函数的比较语句执行。
    # 输入：self，当前 Trainer；data，DataLoader，每批 Tensor [B, D]，完整遍历覆盖 N 个物品。
    # 输出：(best_loss, best_collision_rate)，两个 float 标量；best_loss 只在评估轮次更新，
    #       两个最小值可能来自不同轮次；同时更新 best_collision_epoch（从 1 开始的 int）并保存权重。
    #       MLflow 曲线 step 从 1 开始，checkpoint 内 epoch 保持从 0 开始。
    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):

            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            # [旧实验记录，已停用]
            # self.logger.info(train_loss_output)

            # [新增] 保持原始损失求和口径，并统一使用从 1 开始的 epoch。
            # 打印与记录曲线一致的两个 loss 总和标量。
            print(f"Epoch {epoch_idx + 1}: loss_sum={train_loss:.6f}, recon_sum={train_recon_loss:.6f}")
            if self.tracking:  # 传入记录器才调用记录接口；离线记录器也会进入，由内部决定是否联网。
                self.tracking.metrics({"train/loss_sum": train_loss,  # 记录当前轮总 loss：各批平均 loss 之和，float 标量。
                                       # 记录重建 loss 的同口径总和，float 标量。
                                       "train/reconstruction_loss_sum": train_recon_loss},
                                      step=epoch_idx + 1)  # 曲线横轴采用从 1 开始的轮号 int。

            # [新增] 最后一轮也做目录诊断，保证短训练仍能产出可导出的权重。
            if (epoch_idx + 1) % self.eval_step == 0 or epoch_idx + 1 == self.epochs:  # 按评估间隔及最后一轮生成目录诊断和可选择的权重。
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)  # 用同一物品目录计算碰撞率 float，不使用独立验证集。
                if self.tracking:  # 传入记录器才调用记录接口；离线记录器也会进入，由内部决定是否联网。
                    self.tracking.metrics({"catalog/collision_rate": collision_rate},  # 将目录碰撞率写入当前轮的 MLflow 曲线。
                                          step=epoch_idx + 1)  # 曲线横轴采用从 1 开始的轮号 int。

                if train_loss < self.best_loss:  # 只在评估轮次比较训练 loss；改进时另存最佳 loss 权重。
                    self.best_loss = train_loss
                    self._save_checkpoint(
                        epoch=epoch_idx, ckpt_file=self.best_loss_ckpt
                    )

                if collision_rate < self.best_collision_rate:  # 实际模型选择规则：碰撞率严格更小才替换最佳权重，相同则保留旧权重。
                    self.best_collision_rate = collision_rate  # 更新所选碰撞率 float 标量。
                    self.best_collision_epoch = epoch_idx + 1  # 记录所选权重的轮号 int，供 main 的摘要使用。
                    cur_eval_step = 0
                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        ckpt_file=self.best_collision_ckpt,  # 写入 best_collision_model.pth，后续 main 上传该文件。
                    )
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (
                    epoch_idx,
                    valid_end_time - valid_start_time,
                    collision_rate,
                )

                # [旧实验记录，已停用]
                # self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(
                    epoch_idx, collision_rate=collision_rate
                )
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if (
                        collision_rate < -self.best_save_heap[0][0]
                    ):
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate  # 返回两个可能来自不同轮次的最小值，供摘要记录。
