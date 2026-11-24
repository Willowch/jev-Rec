# MLflow 共用记录层：参数(params)保存固定说明，指标(metrics)保存标量曲线，附件(artifact)保存文件。
# 本模块不直接计算模型张量；路径、字典、客户端等用类型和示例说明，数值指标均为标量。
# 形状约定：N=物品数，D=物品向量维度，L=RQ 量化层数；默认 SID 为 [N, L+1]=[N, 4]。

import argparse  # 命令行参数工具；记录选项会添加到 ArgumentParser。
import hashlib  # 计算输入文件和权重的 SHA-256 内容指纹。
import json  # 读写配置、清单和结果摘要。
import math  # 检查记录的指标是否为有限数值。
import platform  # 获取操作系统信息，写入环境快照。
import re  # 区分 JSON 字符串与 // 行注释，保留服务地址中的 http://。
import shutil  # 将源码和配置复制到 Run 的 source 目录。
import subprocess  # 把启动参数列表拼成可读的命令字符串。
import sys  # 获取 Python 路径、启动参数和标准错误输出。
import time  # 为指标生成记录时间戳。
import traceback  # 将异常类型、消息和调用栈转成文本。
import uuid  # 离线时生成唯一的本地运行 ID。
from importlib import metadata  # 读取已安装依赖包的版本号。
from pathlib import Path  # 统一处理项目、输入和输出路径。


# [新增] 本文件处理路径、字典和普通对象，不直接处理张量；因此用类型和例子说明数据。
# PROJECT_ROOT：TIGER 项目根目录，Path 对象；例如 Path('D:/.../Generative-Rec/TIGER')。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CONFIG_PATH：共用配置文件的位置，Path 对象；例如 Path('.../3_configs/mlflow.json')。
CONFIG_PATH = PROJECT_ROOT / "3_configs" / "mlflow.json"


# 流程：读取 CONFIG_PATH 的 UTF-8 文本 → 去除字符串外的 // 行注释 → 解析 JSON → 返回共用配置。
# 输入：无显式参数；读取模块常量 CONFIG_PATH 指向的 mlflow.json。
# 输出：dict，5 个配置项；如 {'tracking_uri': 'http://127.0.0.1:5000',
#       'backend_store_path': '4_runs/mlflow/mlflow.db', 'artifact_root': '4_runs/mlflow/artifacts',
#       'run_root': '4_runs/01_TIGER', 'experiment_prefix': 'TIGER'}。
def load_settings():
    text = CONFIG_PATH.read_text(encoding="utf-8")  # 配置原文 str，允许 // 行注释。
    text = re.sub(  # 先匹配完整字符串，避免把 http:// 或字符串里的 // 当成注释。
        r'"(?:\\.|[^"\\])*"|//[^\r\n]*',  # 两类匹配：含转义字符的 JSON 字符串，或到行末的注释。
        lambda match: "" if match.group(0).startswith("//") else match.group(0),  # 注释删掉，字符串原样保留。
        text,  # 只去除注释，保留换行以便 JSON 解析错误定位到原行号。
    )
    return json.loads(text)  # 返回 5 个配置项的 dict；仍由 JSON 解析器检查其余语法。


# 流程：以二进制打开文件 → 每次读取最多 1 MiB → 累积 SHA-256 → 返回指纹。
# 输入：path，str 或 Path，如 'item_sid.npy'；只读文件字节，不读取张量形状。
# 输出：str，64 个十六进制字符，如 'a3f1...'（省略至共 64 位）；同内容得到同指纹。
def sha256_file(path):
    digest = hashlib.sha256()  # 创建指纹累积器，尚未读入文件内容。
    with Path(path).open("rb") as stream:  # 只读二进制流；离开 with 自动关闭文件。
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):  # chunk 为 bytes，每块最多 1 MiB；读到空字节串时停止。
            digest.update(chunk)  # 将当前数据块加入指纹计算，无需一次加载整个文件。
    return digest.hexdigest()  # 返回 64 位十六进制 str，供数据版本对比。


# 流程：统一为 Path → 创建父目录 → 序列化为 JSON 并写盘 → 返回文件路径。
# 输入：path，str/Path，如 'run/config.json'；value，待保存对象，如 {'seed': 2024}、[1, 2]。
# 输出：Path，如 Path('run/config.json')；副作用是生成 UTF-8 文件，NaN/Inf 会报错。
def write_json(path, value):
    path = Path(path)  # 统一为 Path，如 Path('run/config.json')。
    path.parent.mkdir(parents=True, exist_ok=True)  # 补齐父目录；已有目录可复用。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,  # 保留中文、缩进 2 格，未知类型转为字符串。
                               allow_nan=False) + "\n", encoding="utf-8")  # 拒绝 NaN/Inf，末尾加换行后以 UTF-8 写盘。
    return path  # 返回写好的本地文件路径 Path。


# 流程：向已有命令行解析器注册 6 个记录选项 → 由调用方稍后执行 parse_args。
# 输入：parser，ArgumentParser 对象，如 argparse.ArgumentParser()。
# 输出：None；原 parser 被扩充，如可解析 '--dataset Beauty --disable_mlflow'。
def add_tracking_arguments(parser):
    parser.add_argument("--dataset", required=True)  # 数据集标签用于分组和目录。
    parser.add_argument("--tracking_uri", default=None)  # 服务地址 str/None；None 时读取 mlflow.json。
    parser.add_argument("--experiment_name", default=None)  # 实验组名 str/None，如 'TIGER/Beauty'。
    parser.add_argument("--run_name", default=None)  # Run 显示名 str/None；默认显示阶段名。
    parser.add_argument("--run_root", default=None)  # 本地输出根目录 str/None，如 '4_runs/01_TIGER'。
    parser.add_argument("--disable_mlflow", action="store_true")  # bool 开关；传入后只保留本地输出。


# 流程：导入 MLflow → 设置全局服务地址 → 创建使用同一地址的显式客户端。
# 输入：tracking_uri，str，如 'http://127.0.0.1:5000'。
# 输出：MlflowClient 对象，如 client.get_run('a1...') 可查询 Run；此函数不创建 Run。
def make_client(tracking_uri):
    # [新增] MLflow 3.1 的附件代理还读取全局 URI，须与显式客户端保持一致。
    import mlflow  # 延迟导入；离线训练不需要执行此导入。
    from mlflow.tracking import MlflowClient  # 导入显式客户端类型，用 Run ID 操作记录。
    mlflow.set_tracking_uri(tracking_uri)  # 同步全局地址，供附件代理等内部逻辑使用。
    return MlflowClient(tracking_uri=tracking_uri)  # 返回使用相同服务地址的客户端对象。


# [新增] 一次训练对应一个 Run；所有输出使用 Run ID 隔离，异常会明确标记失败。
class TrackingRun:
    # 流程：保存参数和阶段 → 读取默认配置 → 合并命令行覆盖项 → 初始化记录状态。
    # 输入：self，当前 TrackingRun 对象；args，Namespace，含全部 6 个记录选项：
    #       dataset='Beauty'、tracking_uri=None、experiment_name=None、run_name=None、
    #       run_root=None、disable_mlflow=False；stage，str，如 'stage1_rqvae'。
    # 输出：None；更新 self 属性，此时 client/run_id 为 None、inputs={}，尚未创建 Run。
    def __init__(self, args, stage):
        self.args = args  # 保留 Namespace，如 args.dataset='Beauty'。
        self.stage = stage  # 阶段标签 str，如 'stage1_rqvae' 或 'stage2_tiger'。
        self.settings = load_settings()  # 共用配置 dict，含服务地址及路径等 5 项。
        self.enabled = not args.disable_mlflow  # bool：是否允许访问 MLflow 服务。
        self.tracking_uri = args.tracking_uri or self.settings["tracking_uri"]  # 服务地址 str：命令行优先，否则采用配置。
        self.experiment_name = args.experiment_name or (  # 实验组名 str：显式指定的名称优先。
            self.settings["experiment_prefix"] + "/" + args.dataset  # 默认拼为 'TIGER/Beauty' 这样的分组名。
        )
        self.run_id = None  # 尚未创建 Run；进入 with 后变为 ID 字符串。
        self.client = None  # 尚未建立客户端；离线时一直为 None。
        self.inputs = {}  # 输入清单 dict；record_setup 后形如 {名称: {path, sha256}}。

    # 流程：在线时查找/新建实验组并创建 Run，离线时生成本地 ID → 创建独立目录 → 返回自身。
    # 输入：self，已初始化的 TrackingRun，如 TrackingRun(args, 'stage1_rqvae')。
    # 输出：self，绑定到 with ... as tracking；run_id 为 str，output_dir 为 Path，
    #       如 '.../Beauty/stage1_rqvae/a1...'；建目录失败时将已创建的 Run 标记 FAILED 并抛异常。
    def __enter__(self):
        if self.enabled:  # 只有启用记录时才访问 MLflow 服务。
            self.client = make_client(self.tracking_uri)  # 建立后续查询、写入和上传共用的客户端。
            experiment = self.client.get_experiment_by_name(self.experiment_name)  # 查找实验组，得到 Experiment 对象或 None。
            experiment_id = experiment.experiment_id if experiment else (  # 已有实验组直接复用其字符串 ID。
                self.client.create_experiment(self.experiment_name)  # 实验组不存在时创建并取得新 ID。
            )
            run = self.client.create_run(experiment_id, tags={  # 在实验组下创建一次训练记录 Run，并设置可检索标签。
                "mlflow.runName": self.args.run_name or self.stage,  # 显示名 str，如 'stage1_rqvae'；不影响 Run 的唯一 ID。
                "stage": self.stage, "dataset": self.args.dataset,  # 两个 str 标签，说明训练阶段和数据集。
                "metric_schema": "tiger-mvp-v1",  # 指标命名约定的版本标签 str。
            })
            self.run_id = run.info.run_id  # 提取服务端 Run ID，通常为 32 位十六进制 str。
        else:
            self.run_id = "local_" + uuid.uuid4().hex  # 离线 ID 如 'local_a1...'，只用于本地目录隔离。
        try:  # 尝试文件操作或附件上传；后续 except 处理对应失败。
            base = Path(self.args.run_root or self.settings["run_root"])  # 输出根目录 Path，命令行优先。
            if not base.is_absolute():  # 相对路径统一按项目根目录解释。
                base = PROJECT_ROOT / base  # 转为项目内的完整路径。
            # Run 专属 Path，如 '.../Beauty/stage1_rqvae/a1...'。
            self.output_dir = base / self.args.dataset / self.stage / self.run_id
            self.output_dir.mkdir(parents=True, exist_ok=False)  # 创建独立目录；同名目录已存在时直接报错。
        except BaseException:  # 目录创建失败或被中断时执行清理记录状态的逻辑。
            if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
                self.client.set_terminated(self.run_id, "FAILED")  # 将已创建但未能准备目录的 Run 标记失败。
            raise  # 继续抛出原异常，阻止后续训练。
        print(f"Run ID: {self.run_id}\nOutputs: {self.output_dir}", flush=True)  # 立即打印运行 ID 和本地输出位置，便于定位结果。
        return self  # 把当前记录器交给 with 的 as 变量。

    # 流程：若有异常则写入并尝试上传错误栈 → 在线时设置最终状态 → 让原异常继续传播。
    # 输入：self，当前记录器；exc_type，异常类或 None，如 ValueError；
    #       exc，异常实例或 None，如 ValueError('输入错误')；tb，traceback 对象或 None，表示调用栈。
    #       后三项由 with 自动传入，正常结束时均为 None。
    # 输出：bool 标量 False，表示不吞掉异常；正常/键盘中断/其他异常对应 FINISHED/KILLED/FAILED。
    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:  # with 块发生异常时才生成错误附件。
            failure = self.output_dir / "failure.txt"  # 错误附件的本地 Path。
            failure.write_text("".join(traceback.format_exception(exc_type, exc, tb)),  # 把异常类型、消息和完整调用栈拼成 str。
                               encoding="utf-8")  # 将异常文本以 UTF-8 写入 failure.txt。
            if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
                try:  # 尝试文件操作或附件上传；后续 except 处理对应失败。
                    self.artifact(failure)  # 尽量上传错误附件，方便在 Run 中排查失败原因。
                except Exception as upload_error:  # 错误附件上传失败时保留原训练异常。
                    print(f"Failure attachment unavailable: {upload_error}", file=sys.stderr)  # 将上传失败原因输出到标准错误流。
        if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
            status = "FINISHED" if exc_type is None else (  # 无异常时记为正常完成。
                "KILLED" if issubclass(exc_type, KeyboardInterrupt) else "FAILED"  # 键盘中断记为 KILLED，其他异常记为 FAILED。
            )
            self.client.set_terminated(self.run_id, status)  # 将最终状态写回当前 Run。
        return False  # 不拦截 with 块内的原异常。

    # 流程：若有客户端，将每个键和值转为字符串 → 封装 Param 列表 → 批量写入当前 Run。
    # 输入：self，当前记录器；values，dict，含 P 个参数，如 {'seed': 2024, 'layers': [512, 256]}。
    # 输出：None；MLflow Parameters 中记录 P 项；离线时跳过，不另存本地参数文件。
    # 说明：这里只记录说明性参数，不会修改模型、损失计算或 checkpoint 选择逻辑。
    def params(self, values):
        if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
            from mlflow.entities import Param  # 参数记录对象：保存字符串键和值。
            # 构建长度 P 的 list[Param]；键转 str，非字符串值转 JSON。
            encoded = [Param(str(key), json.dumps(value, ensure_ascii=False, default=str)
                             # 原本就是 str 的值保持原样，如 'catalog/collision_rate'。
                             if not isinstance(value, str) else value)
                       for key, value in values.items()]  # 遍历参数字典的 P 个键值对。
            self.client.log_batch(self.run_id, params=encoded)  # 批量写入 Parameters，仅作实验记录，不执行配置逻辑。

    # 流程：先把所有值转为数值并检查有限性 → 在线时添加毫秒时间戳和 step → 批量记录。
    # 输入：self，当前记录器；values，dict，含 M 个标量，如 {'train/loss': 0.3}，不接受整批张量；
    #       step，int 标量，默认 0，如第 5 轮传 5。
    # 输出：None；每项记录为 (名称, 数值, 时间戳, step)；离线也检查数值，NaN/Inf 抛 ValueError。
    def metrics(self, values, step=0):
        if not all(math.isfinite(float(value)) for value in values.values()):  # 所有值必须能转成有限 float；在线/离线都检查。
            raise ValueError(f"Metrics must be finite: {values}")  # 遇到 NaN/Inf 时停止记录并报告异常。
        if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
            from mlflow.entities import Metric  # 指标记录对象：名称、数值、时间戳和步号。
            now = int(time.time() * 1000)  # 当前 Unix 时间戳，int 毫秒标量。
            self.client.log_batch(self.run_id, metrics=[  # 将本步 M 个指标一次写入当前 Run。
                # 每个指标值转 float 标量，并关联同一时间戳和 step。
                Metric(key, float(value), now, step) for key, value in values.items()
            ])

    # 流程：若有客户端，将本地文件上传到当前 Run 的指定附件子目录。
    # 输入：self，当前记录器；path，str/Path，如 'best_model.pth'；
    #       artifact_path，str 或 None，如 'checkpoints'，None 表示附件根目录。
    # 输出：None；离线跳过上传；本地文件仍保留，不读取文件中的张量。
    def artifact(self, path, artifact_path=None):
        if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
            self.client.log_artifact(self.run_id, str(path), artifact_path)  # 上传单个本地文件，artifact_path 是服务端附件子目录。

    # 流程：在 Run 输出目录写 JSON → 调用 artifact 上传 → 返回本地 JSON 路径。
    # 输入：self，当前记录器；name，str 文件名，如 'summary.json'；
    #       value，待保存对象，如 {'best_epoch': 10} 或 [{'loss': 0.3}]。
    # 输出：Path，如 Path('.../a1.../summary.json')；离线也会生成该 JSON 文件。
    def json_artifact(self, name, value):
        path = write_json(self.output_dir / name, value)  # 先保存到当前 Run 的本地输出目录，得到 Path。
        self.artifact(path)  # 在线上传刚写好的 JSON；离线保留本地文件。
        return path  # 返回写好的本地文件路径 Path。

    # 流程：校验输入文件并计算指纹 → 记录参数/输入清单 → 保存环境和启动命令 → 备份源码并上传。
    # 输入：self，已进入 with 的记录器；config，完整配置 dict，如 {'seed': 2024, 'layers': [512, 256]}；
    #       inputs，含 F 个文件的 dict[str, str/Path]，如 {'embeddings': 'item_emb.parquet'}，可为 {}。
    # 输出：None；self.inputs 更新为 F 项嵌套字典，如 {'embeddings': {'path': 'D:/.../item_emb.parquet',
    #       'sha256': '64 位指纹'}}；生成 config/inputs/environment/command.json 和 source 目录。
    #       只记录数据路径和指纹，不上传原始输入数据；离线也保存上述本地文件。
    def record_setup(self, config, inputs):
        # [新增] 保存实际参数、数据指纹、启动命令、环境和源码，不上传原始数据。
        self.inputs = {name: {"path": str(Path(path).resolve(strict=True)),  # 输入清单 dict：逐个确认文件存在，保存绝对路径 str。
                              "sha256": sha256_file(path)}  # 为每个输入补充 64 位文件内容指纹。
                       for name, path in inputs.items()}  # 遍历全部 F 个输入文件名称及路径。
        self.params(config)  # 把完整配置记录到 MLflow Parameters。
        self.params({name + "_sha256": entry["sha256"]  # 将数据指纹也记录为便于筛选的固定参数。
                     for name, entry in self.inputs.items()})  # 每个已登记输入对应一个 '<名称>_sha256' 参数。
        self.json_artifact("config.json", config)  # 保存可直接阅读的完整运行配置。
        self.json_artifact("inputs.json", self.inputs)  # 保存所有输入的路径和指纹，不上传原始数据文件。
        versions = {}  # 依赖版本 dict[str, str]，如 {'torch': '1.10.1'}。
        for package in ("torch", "transformers", "mlflow", "numpy", "pandas", "pyarrow",  # 逐个采集训练、数据和记录相关依赖的版本。
                        "scikit-learn", "scipy"):  # 同时登记聚类和数值计算依赖。
            try:  # 尝试文件操作或附件上传；后续 except 处理对应失败。
                versions[package] = metadata.version(package)  # 读取已安装包的版本号 str。
            except metadata.PackageNotFoundError:  # 环境里没有安装该依赖时继续记录其他包。
                versions[package] = "not installed"  # 用字符串标明缺失依赖。
        self.json_artifact("environment.json", {  # 保存 Python、系统和依赖版本快照。
            # 两个环境描述 str，加一个包版本 dict。
            "python": sys.version, "platform": platform.platform(), "packages": versions,
        })
        self.json_artifact("command.json", {  # 保存本次启动方式，便于复现命令。
            "argv": [sys.executable] + sys.argv,  # 完整参数 list[str]，如 ['python.exe', 'main.py', '--seed', '2024']。
            "command": subprocess.list2cmdline([sys.executable] + sys.argv),  # 将参数列表转换成一条可读的命令 str。
            "cwd": str(Path.cwd()),  # 启动时工作目录 str；用于理解相对路径。
        })
        source_dir = self.output_dir / "source"  # 源码快照根目录 Path，如 '.../a1.../source'。
        for source in (PROJECT_ROOT / "0_repos").rglob("*.py"):  # 递归遍历项目 Python 源码，source 为单个文件的 Path。
            if "__pycache__" not in source.parts:  # 跳过缓存目录中的文件。
                target = source_dir / source.relative_to(PROJECT_ROOT)  # 在快照中保留原来的相对目录结构。
                target.parent.mkdir(parents=True, exist_ok=True)  # 准备快照文件的父目录。
                shutil.copy2(source, target)  # 复制当前源码及文件元信息。
        config_target = source_dir / "3_configs" / "mlflow.json"  # 共用 MLflow 配置在快照中的目标 Path。
        config_target.parent.mkdir(parents=True, exist_ok=True)  # 准备配置快照目录。
        shutil.copy2(CONFIG_PATH, config_target)  # 备份本次使用的服务和输出配置。
        requirements = PROJECT_ROOT / "requirements-baselines.txt"
        if not requirements.is_file():  # 工作区的共享清单位于 TIGER 的上一级。
            requirements = PROJECT_ROOT.parent / requirements.name
        shutil.copy2(requirements, source_dir / requirements.name)
        if self.client:  # 有客户端才向服务端写入；离线时跳过该块。
            self.client.log_artifacts(self.run_id, str(source_dir), "source")  # 递归上传整个快照目录，挂到 Run 的 source 附件目录。

    # 流程：读取当前 Run 信息及已登记的向量指纹 → 组装可写入 checkpoint 的来源字典。
    # 输入：self，已调用 record_setup 且 self.inputs 含 embeddings 的记录器。
    # 输出：dict，共 4 项：run_id（str/None）、tracking_uri（str）、experiment_name（str）、
    #       embedding_sha256（64 位 str）；如 {'run_id': 'a1...', 'tracking_uri': 'http://127.0.0.1:5000',
    #       'experiment_name': 'TIGER/Beauty', 'embedding_sha256': 'a3f1...'}；离线时 run_id=None。
    def checkpoint_metadata(self):
        return {"run_id": self.run_id if self.enabled else None,  # 在线保存来源 Run ID；离线用 None，避免误关联。
                "tracking_uri": self.tracking_uri,  # 来源服务地址 str，导出时用于连接同一服务。
                "experiment_name": self.experiment_name,  # 来源实验组名 str，如 'TIGER/Beauty'。
                "embedding_sha256": self.inputs["embeddings"]["sha256"]}  # 训练向量的 64 位指纹，供 SID 导出核对输入版本。


# 流程：定位旁置清单 → 校验 SID/向量指纹 → 在线时校验服务、来源阶段和状态 → 上传清单 → 返回来源。
# 输入：code_path，str/Path，如 'item_sid.npy'，对应整数数组 [N, L+1]，默认 L=3 时为 [N, 4]；
#       embedding_path，str/Path，如 'item_emb.parquet'，embedding 列堆叠后为 [N, D]；
#       tracking，已进入 with 的第二阶段 TrackingRun，如阶段为 'stage2_tiger' 的记录器。
# 输出：dict，3 个字符串字段：source_rqvae_run_id、source_checkpoint_sha256、sid_sha256，
#       如 {'source_rqvae_run_id': 'a1...', 'source_checkpoint_sha256': '64 位指纹', 'sid_sha256': '64 位指纹'}。
#       离线且无清单时只返回 source_rqvae_run_id='untracked' 和 sid_sha256 两项；校验失败抛异常。
#       本函数只读取清单和文件字节，不加载 SID/向量数组。
def load_sid_source(code_path, embedding_path, tracking):
    code_path = Path(code_path)  # SID 文件 Path，如 Path('item_sid.npy')。
    # 旁置清单 Path，如 'item_sid.npy.manifest.json'。
    manifest_path = code_path.with_suffix(code_path.suffix + ".manifest.json")
    if not manifest_path.is_file():  # 先检查 SID 是否附带来源清单。
        if tracking.enabled:  # 在线训练必须能追溯来源，离线可兼容无清单文件。
            raise ValueError("SID 缺少来源清单，请先运行新版 rqvae/generate_code.py。")  # 在线模式拒绝缺少来源证据的 SID。
        # 离线无清单时返回 2 项：未追踪标记和 SID 内容指纹。
        return {"source_rqvae_run_id": "untracked", "sid_sha256": sha256_file(code_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # 来源清单 dict，含 Run ID、路径、指纹和导出信息。
    if sha256_file(code_path) != manifest["sid_sha256"]:  # 重新计算 SID 指纹，确认文件未被替换。
        raise ValueError("SID 文件指纹与来源清单不一致。")  # SID 内容和清单不匹配时阻止训练。
    if sha256_file(embedding_path) != manifest["embedding_sha256"]:  # 核对当前物品向量与导出时是否为同一文件内容。
        raise ValueError("物品向量文件与生成 SID 时使用的版本不一致。")  # 避免用不同版本的向量解释 SID 行号。
    if tracking.enabled:  # 在线训练必须能追溯来源，离线可兼容无清单文件。
        if manifest["tracking_uri"].rstrip("/") != tracking.tracking_uri.rstrip("/"):  # 忽略末尾斜杠后比较来源与当前服务地址。
            raise ValueError("SID 来源和当前训练必须使用同一个 MLflow 服务。")  # 服务地址不一致时拒绝跨服务误关联。
        source_id = manifest.get("source_rqvae_run_id")  # 来源第一阶段 Run ID，str 或 None。
        if not source_id:  # 在线关联需要真实的来源 Run ID。
            raise ValueError("SID 来源未记录 MLflow Run，请用已记录的 RQ-VAE 权重导出 SID。")  # 来源为离线或历史未追踪权重时，拒绝在线关联。
        source = tracking.client.get_run(source_id)  # 向服务端查询来源 Run 对象及其标签和状态。
        # 来源必须是已成功完成的 RQ-VAE 第一阶段训练。
        if source.data.tags.get("stage") != "stage1_rqvae" or source.info.status != "FINISHED":
            raise ValueError("SID 必须来自成功完成的第一阶段 Run。")  # 错误阶段或失败/未完成的来源不允许使用。
    tracking.artifact(manifest_path, "provenance")  # 把来源清单附到当前第二阶段 Run，便于反查。
    # 返回来源 ID；离线清单缺少 ID 时使用 'untracked'。
    return {"source_rqvae_run_id": manifest.get("source_rqvae_run_id") or "untracked",
            "source_checkpoint_sha256": manifest["checkpoint_sha256"],  # 来源权重的 64 位指纹，取自导出清单。
            "sid_sha256": manifest["sid_sha256"]}  # 当前 SID 的 64 位指纹，已与实际文件核对。
