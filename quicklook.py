from pathlib import Path
from utils import loadj

def search_aspect(aspect):
    print("---", aspect, "---")
    count = 0
    root_dir = Path('cache/full_review')
    for path in root_dir.iterdir():
        content = loadj(path)
        key = list(content.keys())[0]
        if aspect in key:
            print(content)
            count += 1
        if count > 3:
            break

search_aspect('cafe')
search_aspect('cozy')
search_aspect('quiet')
search_aspect('comfortable')