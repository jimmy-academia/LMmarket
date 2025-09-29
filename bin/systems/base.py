from collections import defaultdict

from utils import load_or_build, dumpj, loadj

class BaseSystem:
    def __init__(self, args, DATA):
        self.args = args
        self.data = DATA
        self.test = DATA['test']

    