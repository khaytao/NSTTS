# Import necessary libraries for system and OS operations
import sys
import os
# Append the parent directory to the system path


prodiff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ProDiff"))

print(prodiff_path, os.path.isdir(prodiff_path))

sys.path.insert(0, prodiff_path)
print(os.getcwd())


# print(hparams)
# command = f"cd ProDiff && python -m inference.ProDiff_Teacher --config modules/ProDiff/config/prodiff_teacher.yaml --exp_name ProDiff_Teacher --reset --hparams={hparams}"

# os.system(command)

# this is here on purpose, after sys appends
import get_model
txt = "Hello this is a test."  # Replace with your desired text
hparams = f"N=8,text='{txt}'"

from utils.hparams import set_hparams
from utils.hparams import hparams as hp

set_hparams(hparams_str=hparams)

model = get_model.get_model(hp)

print("Model type:", type(model))
print("Model device:", next(model.parameters()).device)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
print("\nModel architecture:")
print(model)
