from .base import BaseSystem
from wtpsplit import SaT
from debug import check

class BestSystem(BaseSystem):
    def __init__(self, args, reviews):
        super().__init__(args, reviews)

class SATBaseline(BaseSystem):
    def __init__(self, args, DATA, city):
        super().__init__(args, DATA, city)
        self.onnx = True

        if self.onnx:
            # ONNX Runtime on GPU (falls back to CPU if CUDA not available)
            self.sat = SaT("sat-3l", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        else:
            self.sat = SaT("sat-3l")
            self.sat.half().to("cuda")  # comment out if no GPU

        self.segmentation()

    def segmentation(self):
        review_texts = [r['text'] for r in self.reviews]
        # sat.split accepts a list and yields per-text segments (iterator)
        segmented_iter = self.sat.split(review_texts[:50])
        segmented_reviews = list(segmented_iter)
        # (optional) attach back to reviews
        # for r, segs in zip(self.reviews, segmented_reviews):
        #     r["segments"] = segs
        print('checking...')
        check()
        return segmented_reviews

# https://github.com/segment-any-text/wtpsplit?tab=readme-ov-file
# https://github.com/PKU-TANGENT/NeuralEDUSeg?tab=readme-ov-file

