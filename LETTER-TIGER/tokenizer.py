"""LETTER residual quantization, collaborative alignment and SID export."""

import math
from collections import defaultdict

import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from torch import nn
from torch.nn import functional as F

from comparison.protocol import SEED, write_json
from features import CF_DIM
from ml1m_data import CODEBOOK_SIZE, SID_LAYERS


ALPHA = 0.02
BETA = 0.0001
COMMITMENT = 0.25
CLUSTERS = 10
SINKHORN_EPSILON = 0.003
SINKHORN_STEPS = 50
BATCH_SIZE = 1024
INIT_ITEMS = 4096
HIDDEN = (2048, 1024, 512, 256, 128, 64)


def mlp(widths):
    modules = []
    for index, (left, right) in enumerate(zip(widths, widths[1:])):
        linear = nn.Linear(left, right)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        modules.append(linear)
        if index < len(widths) - 2:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def distances(rows, centroids):
    return rows.square().sum(1, keepdim=True) + centroids.square().sum(1) - 2 * rows @ centroids.T


@torch.no_grad()
def sinkhorn(cost):
    # Work in log space so a small epsilon remains stable for unequal costs.
    scores = -cost.double() / SINKHORN_EPSILON
    for _ in range(SINKHORN_STEPS):
        scores -= scores.logsumexp(1, keepdim=True) + math.log(scores.shape[0])
        scores -= scores.logsumexp(0, keepdim=True) + math.log(scores.shape[1])
    return scores.argmax(1)


def constrained_kmeans(vectors, count, min_size, max_size, seed):
    return KMeansConstrained(n_clusters=count, size_min=min_size, size_max=max_size,
                             max_iter=10, n_init=1, n_jobs=1, random_state=seed).fit(vectors)


class LetterTokenizer(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        widths = (text_dim, *HIDDEN, CF_DIM)
        self.encoder = mlp(widths)
        self.decoder = mlp(widths[::-1])
        self.codebooks = nn.Parameter(torch.empty(SID_LAYERS, CODEBOOK_SIZE, CF_DIM))

    @torch.no_grad()
    def initialize(self, features):
        residual = self.encoder(features)
        if len(residual) < CODEBOOK_SIZE:
            raise ValueError(f"LETTER initialization needs at least {CODEBOOK_SIZE} items")
        min_size = min(len(residual) // (2 * CODEBOOK_SIZE), 50)
        max_size = max(4 * min_size, math.ceil(len(residual) / CODEBOOK_SIZE))
        for level, codebook in enumerate(self.codebooks):
            result = constrained_kmeans(residual.cpu().numpy(), CODEBOOK_SIZE, min_size,
                                        max_size, SEED + level)
            codebook.copy_(torch.as_tensor(result.cluster_centers_, device=features.device))
            residual = residual - codebook[distances(residual, codebook).argmin(1)]

    @torch.no_grad()
    def peers(self, epoch):
        choices = []
        for level, codebook in enumerate(self.codebooks):
            result = constrained_kmeans(codebook.detach().cpu().numpy(), CLUSTERS, 10, 60, SEED + level + epoch)
            labels = torch.as_tensor(result.labels_, device=codebook.device)
            allowed = labels[:, None].eq(labels)
            allowed.fill_diagonal_(False)
            choices.append(allowed.float())
        return choices

    def losses(self, features, cf, warm, peers):
        residual = self.encoder(features)
        quantized = torch.zeros_like(residual)
        quantization, diversity = [], []
        for level, codebook in enumerate(self.codebooks):
            with torch.no_grad():
                cost = distances(residual, codebook)
                assignments = sinkhorn(cost) if level == SID_LAYERS - 1 else cost.argmin(1)
                positives = torch.multinomial(peers[level][assignments], 1).squeeze(1)
            selected = codebook[assignments]
            quantization.append(F.mse_loss(selected, residual.detach())
                                + COMMITMENT * F.mse_loss(residual, selected.detach()))
            affinities = selected @ codebook.T
            affinities = affinities.scatter(1, assignments[:, None], -torch.inf)
            diversity.append(F.cross_entropy(affinities, positives))
            straight_through = residual + (selected - residual).detach()
            residual = residual - straight_through
            quantized = quantized + straight_through

        reconstruction = F.mse_loss(self.decoder(quantized), features)
        active_vectors = cf[warm]
        collaborative = features.new_zeros(())
        if len(active_vectors) >= 2:
            similarities = quantized[warm] @ active_vectors.T
            collaborative = F.cross_entropy(similarities, torch.arange(len(active_vectors), device=features.device))
        return torch.stack((reconstruction, torch.stack(quantization).mean(),
                            collaborative, torch.stack(diversity).mean()))

    @torch.no_grad()
    def encode_catalog(self, features):
        latents, indices = [], []
        for batch in features.split(BATCH_SIZE):
            latent = self.encoder(batch)
            residual = latent
            codes = []
            for codebook in self.codebooks:
                digit = distances(residual, codebook).argmin(1)
                codes.append(digit)
                residual = residual - codebook[digit]
            latents.append(latent.cpu())
            indices.append(torch.stack(codes, 1).cpu())
        return torch.cat(latents), torch.cat(indices)


def collision_rate(codes):
    return 1 - len(set(map(tuple, codes.tolist()))) / len(codes)


def unique_sids(latents, codes, codebooks):
    """Resolve collisions while preserving the four learned codebook levels."""
    groups = defaultdict(list)
    codes = codes.clone()
    for index, code in enumerate(codes.tolist()):
        groups[tuple(code)].append(index)
    for prefix, rows in groups.items():
        if len(rows) < 2:
            continue
        residual = latents[rows] - sum(codebooks[level, digit] for level, digit in enumerate(prefix[:-1]))
        codes[rows, -1] = sinkhorn(distances(residual, codebooks[-1]))

    books = codebooks.numpy()
    vectors = latents.numpy()
    sids = codes.tolist()
    used = defaultdict(set)
    duplicates = []
    # Reserve one item per proposed code before assigning any replacement.
    for row, code in enumerate(sids):
        prefix = tuple(code[:-1])
        if code[-1] in used[prefix]:
            duplicates.append(row)
        used[prefix].add(code[-1])

    for row in duplicates:
        first, second, third, _ = sids[row]
        residual = vectors[row] - books[0, first] - books[1, second]
        third_order = np.argsort(np.sum((books[2] - residual) ** 2, axis=1), kind="stable")
        # Preserve all three leading digits unless all 256 terminal slots are occupied.
        for digit in (third, *(int(value) for value in third_order if value != third)):
            prefix = first, second, digit
            if len(used[prefix]) == CODEBOOK_SIZE:
                continue
            last_residual = residual - books[2, digit]
            order = np.argsort(np.sum((books[3] - last_residual) ** 2, axis=1), kind="stable")
            terminal = next(int(value) for value in order if value not in used[prefix])
            used[prefix].add(terminal)
            sids[row] = [*prefix, terminal]
            break
        else:
            raise ValueError(f"LETTER SID prefix capacity exhausted for {first, second}")
    return sids


def train_tokenizer(text, embeddings, support, smoke, device, directory, *, epochs=None):
    directory.mkdir()
    features = torch.as_tensor(text, device=device)
    cf = torch.as_tensor(embeddings, device=device)
    warm = torch.as_tensor(support, device=device)
    model = LetterTokenizer(features.shape[1]).to(device)
    rng = torch.Generator().manual_seed(SEED)
    if smoke:
        active = torch.from_numpy(np.flatnonzero(support))
        cold = torch.from_numpy(np.flatnonzero(~support))
        # Keep collaborative alignment active during the single smoke batch.
        rows = torch.cat((active, cold[torch.randperm(len(cold), generator=rng)[:BATCH_SIZE - len(active)]]))
    else:
        rows = torch.arange(len(features))
    init_rows = rows[torch.randperm(len(rows), generator=rng)[:INIT_ITEMS]].to(device)
    model.initialize(features[init_rows])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_weights = features.new_tensor([1, 1, ALPHA, BETA])
    best, history = None, []
    epochs = 1 if smoke else (epochs or 3000)
    checkpoint = directory / "best.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        peers = model.peers(epoch)
        counts = torch.zeros(len(loss_weights), device=device)
        batches = rows[torch.randperm(len(rows), generator=rng)].split(BATCH_SIZE)
        for batch in batches:
            indices = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(features[indices], cf[indices], warm[indices], peers)
            (losses @ loss_weights).backward()
            optimizer.step()
            counts += losses.detach()
        stats = (counts / len(batches)).tolist()
        if not all(map(math.isfinite, stats)):
            raise FloatingPointError(f"LETTER tokenizer produced a non-finite loss at epoch {epoch}")
        record = dict(zip(("reconstruction", "quantization", "collaborative", "diversity"), stats))
        record.update(epoch=epoch, updates=len(batches))
        history.append(record)
        if epoch % 100 == 0 or epoch == epochs:
            model.eval()
            _, codes = model.encode_catalog(features)
            rate = collision_rate(codes)
            record["raw_collision_rate"] = rate
            if best is None or rate < best["raw_collision_rate"]:
                best = record
                torch.save(model.state_dict(), checkpoint)
            print(f"LETTER tokenizer epoch {epoch}: {record}", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    latents, codes = model.encode_catalog(features)
    sids = unique_sids(latents, codes, model.codebooks.detach().cpu())
    if len(set(map(tuple, sids))) != len(text):
        raise ValueError("LETTER tokenizer must export one unique SID per item")
    write_json(directory / "training.json", {
        "settings": {"alpha": ALPHA, "beta": BETA, "commitment": COMMITMENT, "clusters": CLUSTERS,
                     "batch_size": BATCH_SIZE, "init_items": len(init_rows), "hidden": HIDDEN,
                     "sinkhorn_epsilon": SINKHORN_EPSILON, "sinkhorn_steps": SINKHORN_STEPS},
        "epochs": history, "best": best, "export_collision_rate": 0.0,
    })
    return sids
