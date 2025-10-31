# utils.py
import os
import re
import json
import pickle
import argparse

import torch
import random
import numpy as np

import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm 

# % --- load and save functions ---
def readf(path):
    with open(path, 'r') as f:
        return f.read()

def writef(path, content):
    with open(path, 'w') as f:
        f.write(content)

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

def load_or_build(path, build_fn, *args, save_fn=dumpj, load_fn=loadj, **kwargs):
    path = Path(path)
    if path.exists():
        logging.info(f"[load_or_build] >>> {path} exists, loading...")
        return load_fn(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"[load_or_build] >>> {path} does not exist, building...")
    result = build_fn(*args, **kwargs)
    logging.info(f"[load_or_build] >>> saving build result to {path}...")
    save_fn(path, result)
    logging.info("[load_or_build] >>> saving complete.")
    return result

class JSONCache:
    """Simple disk-backed cache with write-through saves."""
    def __init__(self, path):
        self.path = Path(path)
        self.data = loadj(self.path) if self.path.exists() else {}

    def __contains__(self, key):
        return key in self.data

    def get_or_build(self, key, builder, *args, **kwargs):
        if key in self.data:
            return self.data[key]
        value = builder(*args, **kwargs)
        self.data[key] = value
        dumpj(self.path, self.data)
        return value


class InternalCache:
    """Per-review JSON blobs keyed by (key, tag)."""
    # key is review_id or aspect
    def __init__(self, root):
        self.root = _ensure_dir(root)

    def _path(self, key):
        return self.root / f"{key}.json"

    def get(self, key, tag=None, default=None):
        """If tag is None → return full dict; else return payload for tag."""
        path = self._path(key)
        if not path.exists():
            return default
        cache = loadj(path)
        return cache if tag is None else cache.get(tag, default)

    def set(self, key, tag, payload, overwrite=True):
        """Store raw JSON-serializable payload under tag."""
        path = self._path(key)
        cache = loadj(path) if path.exists() else {}
        if overwrite or tag not in cache:
            cache[tag] = payload
            dumpj(path, cache)

# % --- ensures ---

def _ensure_dir(_dir):
    _dir = Path(_dir)
    _dir.mkdir(exist_ok=True, parents=True)
    return _dir

def _ensure_pathref(pathref):
    pathref = Path(pathref)
    if pathref.is_file():
        return readf(pathref).strip()
    else:
        logging.warning(f"[_ensure_pathref] Warning: {pathref} does not exists")
        content = input(f'...enter desired content for {pathref}').strip()
        writef(pathref, content)
        logging.info(f"[_ensure_pathref] content saved to {pathref}")
        return content
        
# % --- logging & seed ---

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_logging(verbose, log_dir, prefix='exp'):
    # usages: logging.warning; logging.error, logging.info, logging.debug

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    logging.info(f"Logging initialized → {log_path}")

# % --- iteration ---

def _iter_line(filepath, total=None, desc=""):
    if total is None:
        with open(filepath, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

    desc = f"[iter_line: {desc}]"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, ncols=90, desc=desc):
            yield line