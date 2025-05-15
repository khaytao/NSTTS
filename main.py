# Import necessary libraries for system and OS operations
import sys
import os
# Append the parent directory to the system path

import torch
import numpy as np

prodiff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ProDiff"))

print(prodiff_path, os.path.isdir(prodiff_path))

sys.path.insert(0, prodiff_path)
print(os.getcwd())

txt = "Hello this is a test."  # Replace with your desired text
hparams = f"N=8,text='{txt}'"
print(hparams)
command = f"cd ProDiff && python -m inference.ProDiff_Teacher --config modules/ProDiff/config/prodiff_teacher.yaml --exp_name ProDiff_Teacher --reset --hparams={hparams}"

os.system(command)
