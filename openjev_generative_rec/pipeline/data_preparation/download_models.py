"""下载固定版本的公开权重；Windows 网络可用时也可直接运行。"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from ..common import sha256

MODELS = {
    "base": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "qwen3.5-2b"),
    "openjev": ("ZefanCai/Open-Jev-2B", "0c7aa498b1627be8da4acf34c863ff0ee0a92785", "openjev-2b"),
    "embedding": ("sentence-transformers/all-MiniLM-L6-v2", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf", "all-MiniLM-L6-v2"),
}


def snapshot(kind, root):
    repo, revision, directory = MODELS[kind]
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urlopen(url, timeout=60) as response:
        info = json.load(response)
    assert info["sha"] == revision
    out = root / directory
    out.mkdir(parents=True, exist_ok=True)
    files = info["siblings"]
    if kind == "embedding":
        names = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt", "README.md", "1_Pooling/config.json"}
        files = [f for f in files if f["rfilename"] in names]
    elif kind == "base":
        files = [f for f in files if "/" not in f["rfilename"] and f["rfilename"] != ".gitattributes"]

    def fetch(entry):
        name = entry["rfilename"]
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("lfs", {}).get("sha256")
        size = entry.get("size")
        if path.exists() and (size is None or path.stat().st_size == size):
            digest = sha256(path)
            if expected is None or digest == expected:
                return name, {"sha256": digest, "bytes": path.stat().st_size}
        temporary = path.with_name(path.name + ".part")
        for attempt in range(3):
            try:
                print(f"download {repo}/{name}", flush=True)
                with urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120) as src, temporary.open("wb") as dst:
                    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        dst.write(chunk)
                digest = sha256(temporary)
                if size is not None and temporary.stat().st_size != size:
                    raise ValueError("incomplete download: " + name)
                if expected and digest != expected:
                    raise ValueError("checksum mismatch: " + name)
                temporary.replace(path)
                return name, {"sha256": digest, "bytes": path.stat().st_size}
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = dict(pool.map(fetch, files))
    (out / "download_manifest.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": manifest}, indent=2), encoding="utf-8")
    print(f"ready: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--kind", nargs="+", choices=list(MODELS), default=["base", "openjev"])
    args = parser.parse_args()
    for kind in args.kind:
        snapshot(kind, args.root)
