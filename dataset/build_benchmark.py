import json
from llm import query_llm
from tqdm import tqdm

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
def build_benchmark(USERS, ITEMS, REVIEWS, ontology_path, user_scores, item_scores, benchmark_path, top_ratio=0.2, alpha=0.8):
    with open(ontology_path, "r") as fp:
        ontology = json.load(fp)

    uids = [user['user_id'] for user in USERS]
    iids = [item['business_id'] for item in ITEMS]

    # initialize user2score/item2score
    for node_name, node in ontology.items():
        ontology[node_name]['user2score'] = {}
        for uid in uids: ontology[node_name]['user2score'][uid] = {'scores': [], 'avg': 0}
        ontology[node_name]['item2score'] = {}
        for iid in iids: ontology[node_name]['item2score'][iid] = {'scores': [], 'avg': 0}
    # collect user2score/item2score
    for node_name, user_stats in user_scores.items():
        for uid, stat in user_stats.items():
            ontology[node_name]['user2score'][uid] = stat
    for node_name, item_stats in item_scores.items():
        for iid, stat in item_stats.items():
            ontology[node_name]['item2score'][iid] = stat

    # generate description per item for retrieval
    iid2desc = {item['business_id']: generate_description(item, REVIEWS) for item in tqdm(ITEMS, desc="Description generation")}

    # generate request per user for retrieval
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

    with open(benchmark_path, "w") as fp:
        json.dump({'iid2desc': iid2desc, 'data': data}, fp, indent=2)
    print(f"new benchmark is saved to \"{benchmark_path}\"")
