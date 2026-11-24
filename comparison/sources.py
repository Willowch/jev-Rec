"""Read the shared raw datasets without importing a model runtime."""

import gzip
import html
import json
import re
import zipfile
from collections import defaultdict

from .protocol import DATA_ROOT, item_key


def raw_directory(dataset, raw_root=None):
    return (raw_root or DATA_ROOT / "raw") / dataset


def amazon_rows(path):
    with gzip.open(path, "rb") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def amazon_sequences(reviews, *, min_rating=0.0):
    users = defaultdict(dict)
    review_count = 0
    for row in amazon_rows(reviews):
        user, asin, timestamp = row.get("reviewerID"), row.get("asin"), row.get("unixReviewTime")
        if not isinstance(user, str) or not user or not isinstance(asin, str) or not asin or type(timestamp) is not int:
            raise ValueError(f"{reviews}: review {review_count + 1} needs reviewerID, asin and unixReviewTime")
        if min_rating != 0.0 and float(row["overall"]) < min_rating:
            continue
        user_items = users[user]
        user_items[asin] = min(timestamp, user_items.get(asin, timestamp))
        review_count += 1

    sequences = []
    for user, items in sorted(users.items()):
        if len(items) >= 5:
            ordered = sorted(items.items(), key=lambda pair: (pair[1], pair[0]))
            sequences.append({"user_id": user, "items": [item for item, _ in ordered],
                              "timestamps": [timestamp for _, timestamp in ordered]})
    return sequences, {"raw_reviews": review_count, "raw_users": len(users),
                       "duplicate_user_item_reviews": review_count - sum(map(len, users.values())),
                       "users_below_min_items": len(users) - len(sequences)}


def clean_text(value):
    return " ".join(html.unescape(value).split()) if isinstance(value, str) else ""


def tiger_product_text(row):
    def clean(value):
        return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(value)).split()) if isinstance(value, str) else ""

    title = clean(row.get("title"))
    categories = [clean(value) for value in (row.get("category") or [])]
    if not any(categories):
        categories = [clean(row.get("main_cat"))]
    categories = ", ".join(dict.fromkeys(value for value in categories if value))
    return " ".join(part for part in (title, f"Categories: {categories}." if categories else "") if part)


def amazon_metadata(path, item_ids):
    """Keep the first usable titled record for each item, including model text variants."""
    metadata = {}
    tiger_texts = {}
    for row in amazon_rows(path):
        item = row.get("asin")
        if item not in item_ids:
            continue
        if item not in metadata or not metadata[item]["title"]:
            title = clean_text(row.get("title"))
            category = row.get("category")
            categories = ", ".join(filter(None, map(clean_text, category))) if isinstance(category, list) else ""
            if item not in metadata or title:
                metadata[item] = {"item_id": item, "title": title, "categories": categories,
                                  "main_category": clean_text(row.get("main_cat"))}
        if item not in tiger_texts and (text := tiger_product_text(row)):
            tiger_texts[item] = text
    return [{**metadata.get(item, {"item_id": item, "title": "", "categories": "", "main_category": ""}),
             "metadata_found": item in metadata, "tiger_text": tiger_texts.get(item, f"Amazon product {item}"),
             "tiger_fallback": item not in tiger_texts} for item in sorted(item_ids)]


def ml1m_movies(path):
    with zipfile.ZipFile(path) as archive, archive.open("ml-1m/movies.dat") as handle:
        for line in handle:
            item_id, title, genres = line.decode("latin-1").rstrip("\r\n").split("::", 2)
            genres = genres.replace("|", ", ")
            yield {"item_id": item_id, "title": title, "genres": genres, "text": f"{title}. Genres: {genres}."}


def ml1m_ratings(path):
    users = defaultdict(list)
    with zipfile.ZipFile(path) as archive, archive.open("ml-1m/ratings.dat") as handle:
        for line in handle:
            user, item, rating, timestamp = line.decode("latin-1").strip().split("::")
            users[user].append((int(timestamp), item, float(rating)))
    for events in users.values():
        events.sort(key=lambda event: (event[0], int(event[1])))
    return users


def ml1m_sequences(path, items, min_rating=0.0):
    for user, events in sorted(ml1m_ratings(path).items(), key=lambda pair: item_key("ml1m", pair[0])):
        eligible = [event for event in events if event[2] >= min_rating and event[1] in items]
        if len(eligible) >= 5:
            yield {"user_id": user, "items": [item for _, item, _ in eligible],
                   "timestamps": [timestamp for timestamp, _, _ in eligible]}


def ml100k_source(path):
    with zipfile.ZipFile(path) as archive:
        genre_rows = archive.read("ml-100k/u.genre").decode("latin-1").splitlines()
        genre_names = [row.split("|")[0] for row in genre_rows if row.strip()]
        catalog = []
        for row in archive.read("ml-100k/u.item").decode("latin-1").splitlines():
            item, title, _, _, _, *flags = row.split("|")
            genres = ", ".join(name for name, flag in zip(genre_names, flags, strict=True) if flag == "1")
            catalog.append({"item_id": item, "title": title, "genres": genres,
                            "text": f"{title}. Genres: {genres}."})
        events = defaultdict(list)
        for row in archive.read("ml-100k/u.data").decode("ascii").splitlines():
            user, item, _, timestamp = row.split()
            events[user].append((int(timestamp), item))
    users = []
    for user, rows in sorted(events.items(), key=lambda pair: int(pair[0])):
        ordered = sorted(rows, key=lambda event: (event[0], int(event[1])))
        users.append({"user_id": user, "items": [item for _, item in ordered],
                      "timestamps": [timestamp for timestamp, _ in ordered]})
    if len(users) != 943 or sum(len(rows) for rows in events.values()) != 100_000:
        raise ValueError("Expected the complete MovieLens-100K source before pilot sampling")
    return users, catalog
