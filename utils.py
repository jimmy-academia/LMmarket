import re
import json
from pathlib import Path
from functools import partial
from tqdm import tqdm

def readf(path):
    with open(path, 'r') as f:
        return f.read()

def writef(path, content):
    with open(path, 'w') as f:
        f.write(content)

def get_dset_root():
    path = Path('.dset_root')
    root = ''

    if path.exists():
        root = readf(path).strip()

    if not root:
        root = input('Enter dataset root path (before /yelp/...csv): ').strip()
        writef(path, root)

    return Path(root)

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

def iter_line(filepath, total=None):
    if total is None:
        with open(filepath, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, ncols=90):
            yield line

def load_make(path, build_fn, cache_fn=dumpj):
    if path.exists():
        print(f'{path} exists, loading...')
        return loadj(path)
    else:
        print(f' [load_make] >>> {path} do not exists, building...')
        result = build_fn()
        print(f' [load_make] >>> saving build result to {path}...')
        cache_fn(path, result)
        print(f' [load_make] >>> saving complete.')
        return result

        
def vprint(msg, flag=True):
    if flag:
        print(msg)

def pause_if(msg="Press Enter to continue...", flag=True):
    if flag:
        input(msg)

### --- Flags --- ###
VERBOSE = True
PAUSE = True
vlog = partial(vprint, flag=VERBOSE)
ppause = partial(pause_if, flag=PAUSE)

def clean_phrase(phrase):
    return phrase.lower().strip("* ").split(". ")[-1].strip().replace("*", "").strip().replace(" ", "_")


def version_path(path):
    path = Path(path)
    if not path.exists():
        return path

    count = 1
    while True:
        new_name = f"{path.name}.{count}"
        new_path = path.with_name(new_name)
        if not new_path.exists():
            return new_path
        count += 1

        