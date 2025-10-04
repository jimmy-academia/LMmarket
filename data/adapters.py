def city_view(global_data, city):
    reviews = []
    users = {}
    items = {}
    info = {}

    pooled_items = global_data.get("items") or {}
    pooled_reviews = global_data.get("reviews") or {}
    pooled_users = global_data.get("users") or {}

    for item_id, meta in pooled_items.items():
        if meta.get("city") != city:
            continue
        review_ids = list(meta.get("review_ids") or [])
        items[item_id] = review_ids
        info[item_id] = {key: value for key, value in meta.items() if key != "review_ids"}

    valid_rids = set()
    for rids in items.values():
        for rid in rids:
            valid_rids.add(rid)

    for rid in valid_rids:
        record = pooled_reviews.get(rid)
        if not record:
            continue
        entry = {"review_id": rid}
        entry.update(record)
        reviews.append(entry)

    for uid, meta in pooled_users.items():
        linked = []
        for rid in meta.get("review_ids") or []:
            if rid in valid_rids:
                linked.append(rid)
        if linked:
            users[uid] = linked

    return {"USERS": users, "ITEMS": items, "REVIEWS": reviews, "INFO": info}
