import json

def get_task_loader(dset):
    if dset == "yelp":
        with open("cache/new/benchmark.json", "r") as fp:
            benchmark = json.load(fp)
        return benchmark
    else:
        raise NotImplementedError(f"task {dset} not implemented")
