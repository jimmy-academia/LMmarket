import re
import json

def readf(path):
    with open(path, 'r') as f:
        return f.read()

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(filepath, dictionary):
    with open(filepath, "w") as f:
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def _cache_call(cache_fn, path, obj):
    try:
        return cache_fn(path, obj) 
    except:
        return cache_fn(obj, str(path))


def load_or_build(paths, cache_fns, load_fns, build_fn, *args, **kwargs):
    if not isinstance(paths, (list, tuple)):
        paths, cache_fns, load_fns = [paths], [cache_fns], [load_fns]

    if all(path.exists() for path in paths):
        print(f'[load_or_build] >>> {paths} exists, loading...')
        vals = [loadf(path) for loadf, path in zip(load_fns, paths)]
        return vals[0] if len(vals) == 1 else vals
    else:
        print(f'[load_or_build] >>> {paths} does not exist, building...')
        results = build_fn(*args, **kwargs)
        print(f'[load_or_build] >>> saving build result to {paths}...')
        if not isinstance(results, (list, tuple)):
            results = [results]
        for path, result, cache_fn in zip(paths, results, cache_fns):
            _cache_call(cache_fn, path, result)
        print(f'[load_or_build] >>> saving complete.')
        return result

