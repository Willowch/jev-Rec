import collections
import json
# [旧实验记录，已停用]
# import logging
from pathlib import Path

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

# 流程：统计编码总数和去重后的编码数 → 比较两者是否相等。
# 输入：all_indices_str，NumPy 字符串数组 [N]，如 ["['<a_1>', '<b_2>', '<c_3>']", ...]。
# 输出：bool 标量，True 表示所有编码唯一，即没有碰撞。
def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

# 流程：遍历每个编码字符串 → 累加出现次数 → 返回计数字典。
# 输入：all_indices_str，字符串序列 [N]，如 ['code_a', 'code_a', 'code_b']。
# 输出：defaultdict[str, int]，如 {'code_a': 2, 'code_b': 1}；每个值为次数标量。
def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

# 流程：按编码汇集物品行号 → 只保留行号数量大于 1 的组 → 返回碰撞物品分组。
# 输入：all_indices_str，字符串序列 [N]，如 ['code_a', 'code_b', 'code_a']。
# 输出：list[list[int]]，形状不规则，如 [[0, 2]]；每个内层列表含共享同一编码的行号。
def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


# [新增] 导出入口参数化，并加载共用指纹和 MLflow 配置。
import argparse  # 导出入口的命令行参数工具。
import random  # 恢复训练配置中的种子，使导出可复现。
import sys  # 用于加载 0_repos 下的共用记录工具。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 将 0_repos 的绝对路径 str 加到搜索路径首位。
# 复用服务配置、客户端、内容指纹和 JSON 清单写入工具。
from experiment_tracking import PROJECT_ROOT, load_settings, make_client, sha256_file, write_json

# [旧实验记录，已停用] 原固定权重与输出位置改由命令行指定。
# dataset = "Musical_Instruments"
# project_root = Path(__file__).resolve().parents[3]
# ckpt_path = (project_root / "4_runs" / "01_TIGER" / dataset / "rqvae_ckpt"
#              / "Oct-27-2026_20-15-19" / "best_collision_model.pth")
# output_file = ckpt_path.with_name(f"{dataset}_t5_rqvae.npy")
# device = torch.device("cuda:0")
#


# [新增] 作用：查找向量文件，兼容项目移入 TIGER 后旧 checkpoint 中保存的路径。
# 输入：data_path，str/Path，如旧的 '.../Generative-Rec/02_data/.../item_emb.parquet'。
# 输出：现存文件的绝对 Path，如 '.../Generative-Rec/TIGER/02_data/.../item_emb.parquet'；找不到时抛错。
def resolve_embedding_path(data_path):
    path = Path(data_path).expanduser()  # 输入位置 Path，可为绝对路径或相对工作目录的路径。
    candidates = [path]  # 候选路径 list[Path]；先保留原位置，避免误改仍有效的路径。
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)  # 相对路径也允许从 TIGER 项目根目录读取。
    else:
        try:
            relative = path.relative_to(PROJECT_ROOT.parent)  # 旧根目录内的相对位置，如 '02_data/.../item_emb.parquet'。
        except ValueError:
            pass  # 项目外的路径保持原意，不按文件名猜测替代文件。
        else:
            candidates.append(PROJECT_ROOT / relative)  # 映射到移动后的 TIGER 目录；原文件内容仍由指纹校验。
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到物品向量文件：{data_path}；请用 --data_path 指定移动后的文件。")


# 流程：解析导出参数 → 加载 checkpoint 并核对向量指纹/来源 Run → 重建模型生成 [N, L] 编码
#       → 尝试减少碰撞并追加消歧列 → 保存 SID 和来源清单 → 在线时追加附件到原第一阶段 Run。
# 输入：无显式参数；读取 sys.argv：ckpt_path 为必填路径 str，output_file/data_path/tracking_uri
#       为可选 str/None，device 为 str（如 'cpu'），disable_mlflow 为 bool；如 '--ckpt_path best.pth'。
#       checkpoint 含配置和各层权重，向量文件 embedding 堆叠为 [N, D]。
# 输出：manifest，11 项 dict，含来源 Run、服务地址、路径/指纹、轮号、种子和物品数；
#       同时保存整数 SID 数组 [N, L+1]（默认 [N, 4]）和 '*.npy.manifest.json'。
#       N=物品数，D=向量维度，L=量化层数；不创建新 Run，--disable_mlflow 只跳过来源查询和上传。
def main():
    parser = argparse.ArgumentParser(description="Export SID and attach to its RQ-VAE run")
    parser.add_argument("--ckpt_path", required=True)  # 必填 checkpoint 路径 str，如 'best_collision_model.pth'。
    parser.add_argument("--output_file", default=None)  # SID 输出路径 str/None；默认在权重旁写 item_sid.npy。
    parser.add_argument("--data_path", default=None)  # 向量文件路径 str/None；默认使用 checkpoint 中的训练路径。
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tracking_uri", default=None)  # 可选服务地址 str/None；优先于权重元信息和共用配置。
    parser.add_argument("--disable_mlflow", action="store_true")  # bool：跳过服务端查询和附件上传，仍做本地指纹校验并写清单。
    options = parser.parse_args()  # 导出参数 Namespace，如 options.ckpt_path='best.pth'。
    ckpt_path = Path(options.ckpt_path).resolve(strict=True)  # 确认 checkpoint 存在并取得绝对 Path。
    dataset = ckpt_path.parent.parent.name
    output_file = Path(options.output_file).resolve() if options.output_file else (  # 指定输出时采用绝对 Path。
        ckpt_path.with_name("item_sid.npy")  # 未指定时在 checkpoint 同目录生成 item_sid.npy。
    )
    if output_file.suffix != ".npy":
        raise ValueError("output_file must end with .npy")
    # SID 或来源清单已存在时拒绝覆盖，保护已有导出结果。
    if output_file.exists() or output_file.with_suffix(".npy.manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite SID: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device)
    # 加载 checkpoint dict，包含 args、state_dict、epoch 和可选 tracking 元信息。
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = ckpt["args"]  # 原训练配置 Namespace，如 args.seed=2024、args.num_emb_list=[256, 256, 256]。
    state_dict = ckpt["state_dict"]  # 模型权重 dict[str, Tensor]；各层张量形状由配置决定。

    # [新增] 可移动文件位置，但必须与训练时的向量内容完全一致。
    data_path = resolve_embedding_path(options.data_path or args.data_path)  # 实际输入向量绝对 Path；兼容旧 checkpoint 的原项目路径。
    embedding_sha256 = sha256_file(data_path)  # 实际向量文件的 64 位内容指纹 str。
    source = ckpt.get("tracking", {})  # 来源 dict（通常 4 项）；旧权重无元信息时为 {}。
    # 有训练指纹时必须与当前向量完全一致，离线也校验。
    if source.get("embedding_sha256") and source["embedding_sha256"] != embedding_sha256:
        raise ValueError("导出输入向量与 RQ-VAE 训练输入不一致。")  # 输入版本不一致时停止，避免生成错误来源的 SID。
    # 最终服务地址 str：命令行 > checkpoint 元信息 > 共用配置。
    tracking_uri = options.tracking_uri or source.get("tracking_uri") or load_settings()["tracking_uri"]
    source_run_id = source.get("run_id")  # 原第一阶段 Run ID，str/None；导出附件将挂到该 Run。
    client = None  # 客户端 MlflowClient/None；离线时保持 None。
    if not options.disable_mlflow:  # 在线导出才查询来源 Run 并准备上传。
        if not source_run_id:  # 在线导出要求 checkpoint 带有来源 Run ID。
            raise ValueError("旧权重缺少来源 Run；请用新版训练权重，或显式传 --disable_mlflow。")  # 旧权重和离线权重不能自动关联在线实验。
        client = make_client(tracking_uri)  # 创建连接最终服务地址的显式客户端，不新建 Run。
        source_run = client.get_run(source_run_id)  # 查询来源 Run 对象，包含 info 状态和 data 标签。
        # 来源必须是已成功完成的 RQ-VAE 训练 Run。
        if source_run.info.status != "FINISHED" or source_run.data.tags.get("stage") != "stage1_rqvae":
            raise ValueError("必须从已成功完成的第一阶段 Run 导出 SID。")  # 来源阶段或状态不符合要求时停止导出。
    seed = getattr(args, "seed", 2024)  # 导出种子 int；优先沿用训练种子，历史权重默认 2024。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = EmbDataset(data_path)

    model = RQVAE(in_dim=data.dim,
                      num_emb_list=args.num_emb_list,
                      e_dim=args.e_dim,
                      layers=args.layers,
                      dropout_prob=args.dropout_prob,
                      bn=args.bn,
                      loss_type=args.loss_type,
                      quant_loss_weight=args.quant_loss_weight,
                      # [新增] 导出重建相同模型配置，兼容非默认 beta。
                      beta=args.beta,
                      kmeans_init=args.kmeans_init,
                      kmeans_iters=args.kmeans_iters,
                      sk_epsilons=args.sk_epsilons,
                      sk_iters=args.sk_iters,
                      )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(data,num_workers=args.num_workers,
                                 batch_size=64, shuffle=False,
                                 pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d,use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)  # 字符串 token 数组 [N, L]，如每行 ['<a_1>', '<b_2>', '<c_3>']。
    all_indices_str = np.array(all_indices_str)  # 逐物品完整编码字符串数组 [N]，用于判重。

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon=0.0

    tt = 0

    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)

            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ",len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate",(tot_item-tot_indice)/tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []

    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)  # 整数编码数组 [N, L]，默认 L=3。

    # 追加消歧列，数组变为 [N, L+1]，默认 [N, 4]。
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]

    if len(duplicates) > 0:
        print("Resolving duplicates in codes...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i  # 同一重复编码组内给末列分配 0、1、2...，使完整 SID 唯一。

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print("There still have duplicates:", duplicates)
    else:
        print("There are no duplicates in the codes after resolution.")

    print(f"Saving codes to {output_file}")
    print(f"the first 5 codes: {codes_array[:5]}")
    np.save(output_file, codes_array)  # 保存最终整数 SID [N, L+1]，行顺序沿用向量文件。
    # [新增] 旁置来源清单用于第二阶段自动关联；直接追加附件，不重开训练 Run。
    manifest = {  # 来源清单 dict，共 11 项；所有值为普通字符串、整数或 None。
        "source_rqvae_run_id": source_run_id,  # 来源训练 Run ID str/None；离线导出也保留权重里已有的 ID。
        "tracking_uri": tracking_uri,  # 来源服务地址 str，供第二阶段定位同一服务。
        "checkpoint_path": str(ckpt_path),  # 所用 checkpoint 的绝对路径 str。
        "checkpoint_sha256": sha256_file(ckpt_path),  # 所用 checkpoint 的 64 位内容指纹 str。
        "checkpoint_epoch": int(ckpt["epoch"]) + 1,  # 把权重内零起始轮号转换为从 1 开始的 int。
        "embedding_path": str(Path(data_path).resolve()),  # 当前向量文件的绝对路径 str。
        "embedding_sha256": embedding_sha256,  # 向量文件的 64 位指纹 str，第二阶段会再次核对。
        "sid_path": str(output_file),  # 刚保存的 SID 文件绝对路径 str。
        "sid_sha256": sha256_file(output_file),  # SID 文件的 64 位指纹 str，用于确认文件未被替换。
        "export_seed": seed,  # 导出种子 int 标量。
        "item_count": len(codes_array),  # 物品数 N，int 标量，等于 SID 行数。
    }
    # 把 11 项来源信息写到 SID 旁的 .npy.manifest.json，返回 Path。
    manifest_path = write_json(output_file.with_suffix(output_file.suffix + ".manifest.json"), manifest)
    if client:  # 仅在线模式将导出结果追加到原 Run 的附件区。
        client.log_artifact(source_run_id, str(ckpt_path), "checkpoints")  # 追加所用权重附件，方便确认 SID 由哪份权重导出。
        artifact_dir = "sid/" + manifest["sid_sha256"]  # 附件子目录 str，用 SID 指纹区分不同导出内容。
        client.log_artifact(source_run_id, str(output_file), artifact_dir)  # 将 SID .npy 上传到原第一阶段 Run 的指纹目录。
        client.log_artifact(source_run_id, str(manifest_path), artifact_dir)  # 同时上传对应来源清单，供人工查看和后续关联。
        client.set_tag(source_run_id, "sid_export_status", "FINISHED")  # 附件全部上传成功后，给原 Run 增加导出完成标签；不重开训练。
    print(f"SID manifest: {manifest_path}")  # 打印清单路径，第二阶段需保留其与 SID 的相邻关系。
    return manifest  # 返回 11 项来源 dict；文件与附件已在前面写好。


if __name__ == "__main__":  # 只有直接执行脚本才导出；导入工具函数不会上传附件。
    main()  # 执行一次 SID 导出并关联原 Run。
