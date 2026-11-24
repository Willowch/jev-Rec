"""按真实 item ID 提取历史文字记忆；随 checkpoint 固化，不反查有碰撞的 SID。"""
import json
from pathlib import Path

from ..common import sha256
from ..data_preparation.build_jev_data import sid_layout

FORMAT = "history_text_v1"
FILENAME = "history_memory.json"


class HistoryMemory:
    def __init__(self, items):
        if not items:
            raise ValueError("history memory catalog is empty")
        for item, entry in items.items():
            if not isinstance(item, str) or not isinstance(entry["text"], str) or not entry["text"].strip():
                raise ValueError(f"invalid history memory item: {item}")
        sid_layout({item: entry["semantic_ids"] for item, entry in items.items()})
        self.items = items

    @classmethod
    def from_catalog(cls, path):
        items = {}
        with Path(path).open(encoding="utf-8") as handle:
            for row in map(json.loads, handle):
                item = str(row["item_id"])
                if item in items:
                    raise ValueError(f"duplicate history memory item: {item}")
                text = row.get("text")
                if not text:
                    title, genres = row.get("title"), row.get("genres")
                    if not title:
                        raise ValueError(f"missing description for item {item}")
                    text = f"{title}. Genres: {genres}." if genres else title
                # 仅保留物品静态描述和映射，不保存用户记录或任何目标标签。
                items[item] = {"text": " ".join(text.split()), "semantic_ids": row["semantic_ids"]}
        return cls(items)

    def enrich(self, row):
        history = row.get("metadata", {}).get("history_items")
        codes = row["state"]["history"].split()
        if not isinstance(history, list) or len(history) != len(codes):
            raise ValueError("history memory requires history_items aligned with the SID history")
        descriptions = []
        for position, (item, code) in enumerate(zip(history, codes), 1):
            if item not in self.items:
                raise ValueError(f"missing history description for item {item}")
            entry = self.items[item]
            if entry["semantic_ids"] != code.split("/"):
                raise ValueError(f"history item/SID mismatch: {item}")
            descriptions.append(f"{position}. {entry['text']}")
        memory = "Previously interacted items, oldest to newest; same order as history SIDs:\n"
        memory += "\n".join(descriptions) if descriptions else "(no previous interactions)"
        # 不读取 target_item、target、候选物品描述；仅使用当前样本允许的历史。
        return {**row, "state": {**row["state"], "history_text_memory": memory}}

    def save(self, checkpoint):
        path = Path(checkpoint) / FILENAME
        path.write_text(json.dumps({"format": FORMAT, "items": self.items},
                                   ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return {"format": FORMAT, "sha256": sha256(path), "items": len(self.items)}


def load_history_memory(checkpoint, config, mode="auto", item_catalog=None):
    """auto 恢复 checkpoint 的输入格式；新增记忆由 SFT 入口显式启用。"""
    if mode not in ("auto", "on", "off"):
        raise ValueError("unknown history memory mode")
    path = Path(checkpoint) / FILENAME
    saved = config.get("history_memory")
    if path.exists() != (saved is not None):
        raise ValueError("history memory checkpoint metadata is missing or inconsistent")
    if saved is not None:
        if mode == "off":
            raise ValueError("cannot disable memory for a checkpoint trained with history memory")
        if saved.get("format") != FORMAT or sha256(path) != saved.get("sha256"):
            raise ValueError("history memory format/checksum mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["format"] != FORMAT:
            raise ValueError("unsupported history memory format")
        memory = HistoryMemory(payload["items"])
        if len(memory.items) != saved["items"]:
            raise ValueError("history memory catalog size mismatch")
        if item_catalog and memory.items != HistoryMemory.from_catalog(item_catalog).items:
            raise ValueError("history memory descriptions or SID mapping differ from checkpoint")
        return memory
    if mode == "on":
        if item_catalog is None:
            raise ValueError("enabling history memory requires an item catalog with descriptions")
        return HistoryMemory.from_catalog(item_catalog)
    return None
