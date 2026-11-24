"""各阶段共用的项目路径与文件校验。"""
import hashlib
import sys
from pathlib import Path


def enable_comparison():
    """Find the sibling shared package when a model is started from its own directory."""
    workspace = str(Path(__file__).resolve().parents[2])
    if workspace not in sys.path:
        sys.path.insert(0, workspace)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
