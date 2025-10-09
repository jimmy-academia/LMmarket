from .base import BaseSystem

class SABRE(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend(self, query):
        print("TODO!!")