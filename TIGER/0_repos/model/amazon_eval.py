"""Batched TIGER beam search with the ML-1M full-catalog ranking rules."""

import json
import random
from collections import Counter, defaultdict
from heapq import nsmallest

import torch
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

from ml1m_data import (CODEBOOK_SIZE, SID_LAYERS, SOURCE_LAYERS, history_tokens,
                       holdout, model_tokens, token_offset)
from ml1m_eval import PROTOCOL, comparison_protocol, fit_temperature, select_users


class Catalog:
    def __init__(self, codes, dataset=None):
        self.codes = codes
        self.items = {code: item for item, code in codes.items()}
        self.tokens = {item: model_tokens(code) for item, code in codes.items()}
        if dataset is None:
            self.order = str
        else:
            protocol = comparison_protocol()
            self.order = {item: protocol.item_key(dataset, item) for item in codes}.__getitem__
        self.children = defaultdict(Counter)
        for code in codes.values():
            for layer in range(SID_LAYERS):
                self.children[code[:layer]][code[layer]] += 1

    def recommend(self, histories, score, *, beam_size, top_k, temperature):
        # A catalog branch survives if at least one of its items is still unseen.
        seen = [Counter(self.codes[item][:length] for item in set(history)
                        for length in range(1, SID_LAYERS + 1))
                for history in histories]
        beams = [[((), 0.0)] for _ in histories]
        for _ in range(SID_LAYERS):
            active = [(index, prefix, total) for index, beam in enumerate(beams)
                      for prefix, total in beam]
            if not active:
                break
            logits = score([(index, prefix) for index, prefix, _ in active]).float()
            probabilities = (logits / temperature).log_softmax(-1).cpu().tolist()
            expansions = [[] for _ in histories]
            for (index, prefix, total), scores in zip(active, probabilities):
                for digit, count in self.children[prefix].items():
                    candidate = prefix + (digit,)
                    if count > seen[index][candidate]:
                        expansions[index].append((candidate, total + scores[digit]))
            beams = [nsmallest(beam_size, choices, key=lambda pair: (-pair[1], pair[0]))
                     for choices in expansions]

        return [
            sorted(
                ({"item_id": self.items[code], "semantic_ids": [f"sid_{digit:03d}" for digit in code],
                  "log_probability": total}
                 for code, total in beam),
                key=lambda row: (-row["log_probability"], self.order(row["item_id"])),
            )[:top_k]
            for beam in beams
        ]


class HistoryBatch:
    def __init__(self, model, histories, item_tokens, max_history, device):
        self.model = model.model
        self.device = device
        ids = torch.tensor([history_tokens(history, item_tokens, max_history) for history in histories],
                           device=device).flatten(1)
        self.mask = ids.ne(0).long()
        self.hidden = self.model.get_encoder()(input_ids=ids, attention_mask=self.mask).last_hidden_state

    def __call__(self, cases):
        indices, prefixes = zip(*cases)
        rows = torch.tensor(indices, device=self.device)
        decoder_ids = torch.tensor([(0,) + model_tokens(prefix) for prefix in prefixes],
                                   device=self.device)
        outputs = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=self.hidden.index_select(0, rows)),
            attention_mask=self.mask.index_select(0, rows),
            decoder_input_ids=decoder_ids,
            use_cache=False,
        )
        start = token_offset(len(prefixes[0]))
        return outputs.logits[:, -1, start:start + CODEBOOK_SIZE]


@torch.no_grad()
def calibrate(model, data, catalog, args, device):
    count = data.count("calibration")
    indices = random.Random(args.seed + 1).sample(range(count), min(args.calibration_rows, count))
    logits, labels = [], []
    for index in indices:
        user_index, layer = divmod(index, SOURCE_LAYERS)
        history, target = holdout(data.users[user_index], "calibration")
        code = catalog.codes[target]
        score = HistoryBatch(model, [history], catalog.tokens, data.manifest["max_history"], device)
        logits.append(score([(0, code[:layer])])[0].float().cpu().tolist())
        labels.append(code[layer])
    return {"temperature": fit_temperature(logits, labels), "split": "calibration",
            "indices": indices, "normalization": "full"}


@torch.no_grad()
def recommendation_entries(model, catalog, users, split, temperature, device, *,
                           max_history, batch_size, beam_size, top_k):
    protocol = comparison_protocol()
    for start in tqdm(range(0, len(users), batch_size), desc=split):
        batch = users[start:start + batch_size]
        histories, targets = zip(*(protocol.holdout(user, split) for user in batch))
        score = HistoryBatch(model, histories, catalog.tokens, max_history, device)
        predictions = catalog.recommend(histories, score, beam_size=beam_size, top_k=top_k,
                                        temperature=temperature)
        for user, target, recommendations in zip(batch, targets, predictions):
            yield protocol.ranked_entry(user["user_id"], target, recommendations)


def evaluate_split(model, data, catalog, split, user_limit, temperature, args, device, output, cohort=None,
                   *, users=None, selection=None, train_selection=None):
    protocol = comparison_protocol()
    if users is None:
        users = cohort[0] if cohort else select_users(data.users, args.seed, user_limit)
    predictions = recommendation_entries(model, catalog, users, split, temperature, device,
                                         max_history=data.manifest["max_history"], batch_size=args.eval_batch_size,
                                         beam_size=args.beam_size, top_k=args.top_k)
    details = {"protocol": PROTOCOL, "policy_normalization": "full",
               "beam_size": args.beam_size, "seed": args.seed}
    if cohort:
        return protocol.write_evaluation(output, predictions, users, catalog.codes, cohort[1], "TIGER",
                                         validation_protocol=selection, train_selection=train_selection, **details)
    if output is None:
        return {**details, **protocol.validation_metrics(predictions, users, args.top_k)}

    ranks = []
    with output.open("x", encoding="utf-8") as handle:
        handle.write('{"entries":[\n')
        for index, entry in enumerate(predictions):
            if index:
                handle.write(",\n")
            ranks.append(entry["rank"])
            handle.write(json.dumps(entry, allow_nan=False, separators=(",", ":")))
        metrics = {**details, "users": len(ranks), **protocol.ranking_metrics(ranks, (args.top_k,))}
        handle.write("\n]," + json.dumps(metrics, indent=2).removeprefix("{") + "\n")
    return metrics
