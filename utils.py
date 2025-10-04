# utils.py
import os
import re
import json
import pickle

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

def load_or_build(path, save_fn, load_fn, build_fn, *args, **kwargs):
    path = Path(path)
    exists = path.exists() if hasattr(path, "exists") else Path(path).exists()
    if exists:
        logging.info(f"[load_or_build] >>> {path} exists, loading...")
        return load_fn(path)

    logging.info(f"[load_or_build] >>> {path} does not exist, building...")
    result = build_fn(*args, **kwargs)
    logging.info(f"[load_or_build] >>> saving build result to {path}...")
    save_fn(path, result)
    logging.info("[load_or_build] >>> saving complete.")
    return result

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

def _iter_line(filepath, total=None):
    if total is None:
        with open(filepath, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, ncols=90):
            yield line