import json
from llm import query_llm
from tqdm import tqdm
from pathlib import Path
from utils import load_make
from .yelp import load_yelp_data

def load_ontology(ontology_path):
    with open(ontology_path, "r") as fp:
        ontology = json.load(fp)
    for node in ontology.values():
        for uid, stat in node['user2score'].items():
            if stat['avg'] == -100: stat['avg'] = 0
        for iid, stat in node['item2score'].items():
            if stat['avg'] == -100: stat['avg'] = 0
    return ontology

def split_nodes(ontology, uid, top_ratio):
    node_name_user_avg_pairs = []
    for node_name, node in ontology.items():
        node_name_user_avg_pairs.append((node_name, node['user2score'][uid]['avg']))
    node_name_user_avg_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    top_cnt = int(top_ratio * len(ontology))
    top_nodes = [node_name for node_name, _ in node_name_user_avg_pairs[:top_cnt]]
    last_nodes = [node_name for node_name, _ in node_name_user_avg_pairs[top_cnt:]]

    return top_nodes, last_nodes

def collect_scores(ontology, top_nodes, last_nodes, id, type_key):
    top_avgs = []
    for topx_node_name in top_nodes:
        top_avgs.append(ontology[topx_node_name][type_key][id]['avg'])
    last_avgs = []
    for last_node_name in last_nodes:
        last_avgs.append(ontology[last_node_name][type_key][id]['avg'])
    return top_avgs, last_avgs

def cal_similarity(user_top, user_last, item_top, item_last, alpha):
    top_sim = 0
    for user_avg, item_avg in zip(user_top, item_top):
        top_sim += (item_avg * user_avg)
    last_sim = 0
    for user_avg, item_avg in zip(user_last, item_last):
        last_sim += (item_avg * user_avg)
    return alpha * top_sim + (1 - alpha) * last_sim

def generate_description(ITEM, REVIEWS):
    sel_reviews = [review['text'] for rid in ITEM['review_ids'] for review in REVIEWS if review['review_id'] == rid]
    prompt = f"""
Based on the following restaurant reviews from different customers, summarize the restaurant’s features and generate a concise description for the restaurant in natural language without any other explanation. Emphasize both the positive and negative aspects mentioned in reviews multiple times.

### reviews
{"\n".join(sel_reviews)}
""".strip()
    response = query_llm(prompt)
    # print(response)
    return response

def generate_request(top_nodes, user_top_scores):
    prompt = f"""
You are given multiple pairs of **(feature, score)**, where each score ranges from **1 to -1**.

- **1** represents the strongest preference (most liked).
- **-1** represents the strongest aversion (least liked).

Your task is to **analyze the user’s likes and dislikes** based on these feature–score pairs.
Then, **generate a request in natural language** describing the kind of restaurant or cuisine the user would be most interested in without any other explanation.

### (feature, score)
{"\n".join([f"({node}, {score})" for node, score in zip(top_nodes, user_top_scores)])}
""".strip()
    response = query_llm(prompt)
    # print(response)
    return response

# main function
def build_benchmark(USERS, ITEMS, REVIEWS, ontology_path, top_ratio=0.2, alpha=0.8):
    ontology = load_ontology(ontology_path)

    uids = [user['user_id'] for user in USERS]
    iids = [item['business_id'] for item in ITEMS]

    # collect reviews
    # uid2rev = {}
    # for user in USERS:
    #     revs = []
    #     for review in REVIEWS:
    #         if review["review_id"] in user["review_ids"][:10]: revs.append(review["text"])
    #     uid2rev[user['user_id']] = revs
    # with open("./cache/uid2review.json", "w") as fp:
    #     json.dump(uid2rev, fp, indent=2)
    # iid2rev = {}
    # for item in ITEMS:
    #     revs = []
    #     for review in REVIEWS:
    #         if review["review_id"] in item["review_ids"][:10]: revs.append(review["text"])
    #     iid2rev[item['business_id']] = revs
    # with open("./cache/iid2review.json", "w") as fp:
    #     json.dump(iid2rev, fp, indent=2)

    iid2desc = {item['business_id']: generate_description(item, REVIEWS) for item in tqdm(ITEMS, desc="Description generation")}

    data = []
    for uid in tqdm(uids, desc="Request generation"):
        top_nodes, last_nodes = split_nodes(ontology, uid, top_ratio)

        user_top_scores, user_last_scores = collect_scores(ontology, top_nodes, last_nodes, uid, "user2score")

        iid_sim_pairs = []
        for iid in iids:
            item_top_scores, item_last_scores = collect_scores(ontology, top_nodes, last_nodes, iid, "item2score")
            sim = cal_similarity(user_top_scores, user_last_scores, item_top_scores, item_last_scores, alpha)
            iid_sim_pairs.append((iid, sim))
        
        iid_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        # 用 REVIEWS 生 user request
        data.append({
            'user_id': uid,
            'request': generate_request(top_nodes, user_top_scores),
            'item_ids_rank': [iid for iid, _ in iid_sim_pairs]
        })

    with open("cache/new/benchmark.json", "w") as fp:
        json.dump({'iid2desc': iid2desc, 'data': data}, fp, indent=2)
    print("new benchmark is saved to \"cache/new/benchmark.json\"")

def build_benchmark_tmp(args):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    if args.dset == 'yelp':
        data_path = cache_dir / "yelp_data.json"
        data = load_make(data_path, load_yelp_data)
        print('load or created', data_path)
    else:
        input(f"{args.dset} not implemented yet!")
    
    # filter user + item + review
    USERS = data["USERS"]
    with open("/home/user/work/academia/LMmarket_tmp/cache/new/user_profiles.json", "r") as fp:
        sel_user = json.load(fp)
    USERS = [user for user in tqdm(USERS, desc="USER filtering") if user["user_id"] in sel_user]

    ITEMS = data["ITEMS"]
    with open("/home/user/work/academia/LMmarket_tmp/cache/new/item_profiles.json", "r") as fp:
        sel_item = json.load(fp)
    ITEMS = [item for item in tqdm(ITEMS, desc="ITEMS filtering") if item['business_id'] in sel_item]

    REVIEWS = data["REVIEWS"]
    with open("/home/user/work/academia/LMmarket_tmp/cache/new/review_to_features.json", "r") as fp:
        sel_review = json.load(fp)
    REVIEWS = [review for review in tqdm(REVIEWS, desc="REVIEWS filtering") if review["review_id"] in sel_review]

    # update ontology
    with open("/home/user/work/academia/LMmarket_tmp/cache/new/ontology.json", "r") as fp:
        ontology = json.load(fp)

    uids = [user['user_id'] for user in USERS]
    iids = [item['business_id'] for item in ITEMS]
    for node_name, node in ontology.items():
        ontology[node_name]['user2score'] = {}
        for uid in uids: ontology[node_name]['user2score'][uid] = {'scores': [], 'avg': 0}
        ontology[node_name]['item2score'] = {}
        for iid in iids: ontology[node_name]['item2score'][iid] = {'scores': [], 'avg': 0}

    with open("/home/user/work/academia/LMmarket_tmp/cache/new/user_node_scores.json", "r") as fp:
        user_node_score = json.load(fp)
    for node_name, user_stats in user_node_score.items():
        for uid, stat in user_stats.items():
            ontology[node_name]['user2score'][uid] = stat
    
    with open("/home/user/work/academia/LMmarket_tmp/cache/new/item_node_scores.json", "r") as fp:
        item_node_score = json.load(fp)
    for node_name, item_stats in item_node_score.items():
        for iid, stat in item_stats.items():
            ontology[node_name]['item2score'][iid] = stat

    ontology_path = "/home/user/work/academia/LMmarket_tmp/cache/new/ontology_updated.json"
    with open(ontology_path, "w") as fp:
        json.dump(ontology, fp, indent=2)

    build_benchmark(USERS, ITEMS, REVIEWS, ontology_path, top_ratio=0.2, alpha=0.8)

# cost = current - 0.27