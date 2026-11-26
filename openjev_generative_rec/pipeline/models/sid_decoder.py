"""Predict each SID layer with a vocabulary projection and decode through the catalog."""

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

from ..data_preparation.build_jev_data import input_record
from .sid_tokens import ITEM_START, sid_token


class SIDDecoder(nn.Module):
    def __init__(self, decision, catalog, max_history):
        super().__init__()
        if getattr(decision, "history_memory", None) is None or decision.sid_depth != catalog.depth:
            raise ValueError("The SID decoder needs a layered checkpoint with text history memory")
        if decision.sharded or torch.device(decision.device_name).type == "meta":
            raise ValueError("The SID decoder needs a backbone resident on one device")
        self.decision, self.catalog, self.max_history = decision, catalog, max_history
        self.device = decision.head.weight.device
        self.depth = catalog.depth
        decision.head.requires_grad_(False)
        self.item_codes = {item: code for code, items in catalog.items_by_code.items() for item in items}
        values = [set() for _ in range(self.depth)]
        self.prefix_counts = Counter()
        for code, items in catalog.items_by_code.items():
            for level, token in enumerate(code):
                values[level].add(token)
                self.prefix_counts[code[:level + 1]] += len(items)
        self.tokens = [sorted(options) for options in values]
        self.columns = [{token: index for index, token in enumerate(layer)} for layer in self.tokens]
        tokenizer = decision.tokenizer
        self.token_ids = [dict(zip(layer, tokenizer.convert_tokens_to_ids(
            [sid_token(token, level) for token in layer]), strict=True))
            for level, layer in enumerate(self.tokens, 1)]
        self.item_start_id = tokenizer.convert_tokens_to_ids(ITEM_START)
        width = decision.backbone.config.hidden_size
        self.heads = nn.ModuleList(nn.Linear(width, len(layer), device=self.device) for layer in self.tokens)
        self.support = {}
        for prefix, options in catalog.options.items():
            legal = set(options)
            self.support[prefix] = [token in legal for token in self.tokens[len(prefix)]]

    def prompts(self, histories):
        tokenizer = self.decision.tokenizer
        prompts = []
        for history in histories:
            recent = history[-self.max_history:]
            row = input_record([self.item_codes[item] for item in recent], (), recent,
                               options=self.catalog.options[()][:1])
            state = self.decision.prepare_record(row)["state"]
            prompt = (f"History (oldest to newest): {state['history']}\n{state['history_text_memory']}\n"
                      "Generate the semantic ID of the next item.")
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))
        ids = tokenizer(prompts, add_special_tokens=False, truncation=False)["input_ids"]
        if any(not row or len(row) + self.depth > self.decision.max_length for row in ids):
            raise ValueError("A SID history exceeds the checkpoint context limit")
        return [row + [self.item_start_id] for row in ids]

    def prefix_ids(self, prefix):
        return [self.token_ids[level][token] for level, token in enumerate(prefix)]

    def padded(self, sequences, *, left=False):
        lengths = torch.tensor([len(row) for row in sequences], device=self.device)
        width = max(map(len, sequences))
        pad = self.decision.tokenizer.pad_token_id
        padding = [[pad] * (width - len(row)) for row in sequences]
        rows = [p + row if left else row + p for row, p in zip(sequences, padding, strict=True)]
        positions = torch.arange(width, device=self.device)
        mask = positions >= width - lengths[:, None] if left else positions < lengths[:, None]
        return torch.tensor(rows, device=self.device), mask.long(), lengths

    def layer_logits(self, hidden, level, prefixes):
        valid = torch.tensor([self.support[prefix] for prefix in prefixes], device=self.device)
        return self.heads[level](hidden.float()).masked_fill(~valid, -torch.inf), valid

    def sequence_log_probs(self, prompts, codes):
        inputs, mask, _ = self.padded([prompt + self.prefix_ids(code[:-1])
                                       for prompt, code in zip(prompts, codes, strict=True)])
        hidden = self.decision.backbone(input_ids=inputs, attention_mask=mask, use_cache=False,
                                        return_dict=True).last_hidden_state
        offsets = torch.tensor([len(prompt) - 1 for prompt in prompts], device=self.device)
        rows = torch.arange(len(prompts), device=self.device)
        log_probs, decisions = [], []
        for level, columns in enumerate(self.columns):
            logits, valid = self.layer_logits(hidden[rows, offsets + level], level,
                                              [code[:level] for code in codes])
            labels = torch.tensor([columns[code[level]] for code in codes], device=self.device)
            log_probs.append(logits.log_softmax(dim=-1).gather(1, labels[:, None]).squeeze(1))
            decisions.append(valid.sum(dim=1) > 1)
        return torch.stack(log_probs, dim=1), torch.stack(decisions, dim=1)

    def training_loss(self, prompts, codes):
        log_probs, decisions = self.sequence_log_probs(prompts, codes)
        return (-log_probs.sum(dim=1) / decisions.sum(dim=1).clamp_min(1)).mean()

    def _prefill(self, prompts):
        from jev.prefix_cache import new_cache

        inputs, mask, _ = self.padded(prompts, left=True)
        cache = new_cache(self.decision.backbone.config)
        output = self.decision.backbone(input_ids=inputs, attention_mask=mask,
                                        position_ids=(mask.cumsum(dim=1) - 1).clamp_min(0),
                                        past_key_values=cache, use_cache=True, return_dict=True)
        return output.last_hidden_state[:, -1], output.past_key_values, mask

    def _extend(self, cache, mask, layer, active):
        parents = torch.tensor([row[3] for row in active], device=self.device)
        cache.reorder_cache(parents)
        mask = F.pad(mask.index_select(0, parents), (0, 1), value=1)
        ids = torch.tensor([[self.token_ids[layer][prefix[-1]]] for _, prefix, _, _ in active],
                           device=self.device)
        output = self.decision.backbone(input_ids=ids, attention_mask=mask,
                                        position_ids=mask.sum(1, keepdim=True) - 1,
                                        past_key_values=cache, use_cache=True, return_dict=True)
        return output.last_hidden_state[:, -1], output.past_key_values, mask

    def _recompute(self, prompts, active):
        ids, mask, lengths = self.padded([prompts[user] + self.prefix_ids(prefix)
                                          for user, prefix, _, _ in active])
        hidden = self.decision.backbone(input_ids=ids, attention_mask=mask, use_cache=False,
                                        return_dict=True).last_hidden_state
        return hidden[torch.arange(len(active), device=self.device), lengths - 1]

    @torch.inference_mode()
    def sample_codes(self, prompt, count, batch_size, generator):
        """Draw independent SID paths, sharing scores for equal prefixes within a group."""
        prefixes = [()] * count
        for level, tokens in enumerate(self.tokens):
            unique = list(dict.fromkeys(prefixes))
            locations = {prefix: index for index, prefix in enumerate(unique)}
            parts = []
            for start in range(0, len(unique), batch_size):
                batch = unique[start:start + batch_size]
                hidden = self._recompute([prompt], [(0, prefix, 0.0, 0) for prefix in batch])
                parts.append(self.layer_logits(hidden, level, batch)[0].softmax(dim=-1))
            order = torch.tensor([locations[prefix] for prefix in prefixes], device=self.device)
            probs = torch.cat(parts).index_select(0, order)
            chosen = torch.multinomial(probs, 1, generator=generator).flatten().tolist()
            prefixes = [(*prefix, tokens[index]) for prefix, index in zip(prefixes, chosen, strict=True)]
        return prefixes

    @torch.inference_mode()
    def audit_logits(self, histories, *, cached):
        """Use observed item codes to check every layer without reading holdout labels."""
        prompts = self.prompts(histories)
        codes = [self.item_codes[history[-1]] for history in histories]
        values = []
        if cached:
            hidden, cache, mask = self._prefill(prompts)
        for level in range(self.depth):
            active = [(row, code[:level], 0, row) for row, code in enumerate(codes)]
            if not cached:
                hidden = self._recompute(prompts, active)
            elif level:
                hidden, cache, mask = self._extend(cache, mask, level - 1, active)
            logits, valid = self.layer_logits(hidden, level, [row[1] for row in active])
            values.extend(logits[valid].tolist())
        return values

    @torch.inference_mode()
    def recommend(self, prompts, histories, beam_size, item_key, *, cached=True):
        seen = [set(history) for history in histories]
        seen_prefixes = [Counter(self.item_codes[item][:level] for item in items
                                 for level in range(1, self.depth + 1)) for items in seen]
        active = [(index, (), 0.0, index) for index in range(len(prompts))]
        if cached:
            hidden, cache, mask = self._prefill(prompts)
        for level, columns in enumerate(self.columns):
            if not active:
                break
            if not cached:
                hidden = self._recompute(prompts, active)
            logits, _ = self.layer_logits(hidden, level, [prefix for _, prefix, _, _ in active])
            probabilities = logits.log_softmax(dim=-1).tolist()
            expanded = [[] for _ in prompts]
            for parent, (user, prefix, score, _) in enumerate(active):
                for token in self.catalog.options[prefix]:
                    code = (*prefix, token)
                    if self.prefix_counts[code] > seen_prefixes[user][code]:
                        expanded[user].append((user, code, score + probabilities[parent][columns[token]], parent))
            active = [row for candidates in expanded
                      for row in sorted(candidates, key=lambda row: (-row[2], row[1]))[:beam_size]]
            if cached and active and level + 1 < self.depth:
                hidden, cache, mask = self._extend(cache, mask, level, active)
        results = [[] for _ in prompts]
        for user, code, score, _ in active:
            results[user].extend({"item_id": item, "semantic_ids": list(code), "log_probability": score}
                                 for item in self.catalog.items_by_code[code] if item not in seen[user])
        return [sorted(rows, key=lambda row: (-row["log_probability"], item_key(row["item_id"])))[:beam_size]
                for rows in results]
