import os
import sys

prodiff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ProDiff"))

print(prodiff_path, os.path.isdir(prodiff_path))

sys.path.insert(0, prodiff_path)
print(os.getcwd())

from inference.ProDiff_Teacher import ProDiffTeacherInfer


def get_model(hparams):
    infer = ProDiffTeacherInfer(hparams)
    return infer.model

if __name__ == "__main__":
    model = get_model()
    print("Model type:", type(model))
    print("Model device:", next(model.parameters()).device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    print("\nModel architecture:")
    print(model)
