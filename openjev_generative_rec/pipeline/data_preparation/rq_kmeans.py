"""三层 Residual Quantization K-means，输出 255-way 语义 ID。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from ..common import sha256
from ..minionerec import append_identity_tokens


def balanced_kmeans_level(residual, clusters, seed):
    """Adapted from MiniOneRec's rq/rqkmeans_constrained.py (Apache-2.0).

    Keep the cluster capacities and three initializations. Use one job for
    the local comparison, and leave MiniOneRec's Polars export out of this path.
    See MiniOneRec-LICENSE.txt.
    """
    try:
        from k_means_constrained import KMeansConstrained
    except ImportError as error:
        raise RuntimeError("Install k-means-constrained in the quantization environment to use balanced RQ.") from error

    items_per_cluster = len(residual) // clusters
    km = KMeansConstrained(
        n_clusters=clusters, size_min=max(1, items_per_cluster - 1), size_max=items_per_cluster + 1,
        n_init=3, max_iter=100, tol=1e-4, random_state=seed, n_jobs=1,
    )
    labels = km.fit_predict(residual)
    return labels, km.cluster_centers_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--item-meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--clusters", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=("sklearn", "balanced"), default="sklearn")
    parser.add_argument("--collision-policy", choices=("allow", "append"), default="allow")
    args = parser.parse_args()
    if not 2 <= args.clusters <= 255 or args.layers != 3:
        raise ValueError("此实验使用 3 层语义码，每层支持 2 到 255 个候选 token")
    args.out.mkdir(parents=True, exist_ok=True)
    residual = np.load(args.embeddings).astype("float32")
    if residual.ndim != 2 or len(residual) < args.clusters or not np.isfinite(residual).all():
        raise ValueError("embedding 必须有限且物品数不得小于簇数")
    metadata = [json.loads(line) for line in args.item_meta.read_text(encoding="utf-8").splitlines()]
    if len(metadata) != len(residual) or len({row["item_id"] for row in metadata}) != len(metadata):
        raise ValueError("物品元数据必须与 embedding 逐行对应，且物品 ID 不得重复")
    ids = np.zeros((len(residual), args.layers), dtype="int32")
    codebooks, errors, occupancy = [], [], []
    for layer in range(args.layers):
        if args.method == "balanced":
            # Match MiniOneRec's seed derivation for each residual level.
            seed = int(np.random.RandomState(args.seed + layer).randint(0, 2**31 - 1))
            labels, centers = balanced_kmeans_level(residual, args.clusters, seed)
        else:
            km = KMeans(n_clusters=args.clusters, n_init=5, random_state=args.seed + layer)
            labels = km.fit_predict(residual)
            centers = km.cluster_centers_
        centers = centers.astype("float32")
        ids[:, layer] = labels
        residual = residual - centers[labels]
        codebooks.append(centers)
        errors.append(float(np.mean(residual ** 2)))
        sizes = np.bincount(labels, minlength=args.clusters)
        occupancy.append({"used": int(np.count_nonzero(sizes)), "min": int(sizes.min()), "max": int(sizes.max())})
        print(f"layer={layer + 1}, residual_mse={errors[-1]:.6f}, occupancy={occupancy[-1]}", flush=True)
    np.savez_compressed(args.out / "codebooks.npz", *codebooks)
    mapping = {item["item_id"]: [f"sid_{int(x):03d}" for x in semantic]
               for item, semantic in zip(metadata, ids, strict=True)}
    collision_stats = {}
    if args.collision_policy == "append":
        mapping, collision_stats = append_identity_tokens(mapping)
    with (args.out / "item_semantic_ids.jsonl").open("w", encoding="utf-8") as dst:
        for item in metadata:
            item["semantic_ids"] = mapping[item["item_id"]]
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
    np.save(args.out / "semantic_ids.npy", ids)
    unique, counts = np.unique(ids, axis=0, return_counts=True)
    summary = {"items": len(ids), "layers": args.layers, "clusters_per_layer": args.clusters,
               "method": args.method, "embeddings_sha256": sha256(args.embeddings),
               "item_metadata_sha256": sha256(args.item_meta), "cluster_occupancy": occupancy,
               "residual_mse": errors, "unique_codes": len(unique),
               "collision_groups": int((counts > 1).sum()), "colliding_items": int(counts[counts > 1].sum()),
               "max_items_per_code": int(counts.max()), "seed": args.seed,
               "collision_policy": args.collision_policy, "sid_depth": len(next(iter(mapping.values()))),
               **collision_stats}
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
