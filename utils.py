# utils.py
import re
import json
import pickle
from pathlib import Path

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

def dumpp(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def loadp(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_or_build(path, save_fn, load_fn, build_fn, *args, **kwargs):
    path = Path(path)
    exists = path.exists() if hasattr(path, "exists") else Path(path).exists()
    if exists:
        print(f"[load_or_build] >>> {path} exists, loading...")
        return load_fn(path)

    print(f"[load_or_build] >>> {path} does not exist, building...")
    result = build_fn(*args, **kwargs)
    print(f"[load_or_build] >>> saving build result to {path}...")
    save_fn(path, result)
    print("[load_or_build] >>> saving complete.")
    return result