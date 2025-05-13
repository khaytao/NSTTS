# Import necessary libraries for system and OS operations
import sys
import os
# Append the parent directory to the system path

import torch
import numpy as np

prodiff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Prodiff"))

print(prodiff_path, os.path.isdir(prodiff_path))

sys.path.insert(0, prodiff_path)
print(os.getcwd())

command = "cd ProDiff && python -m inference.ProDiff_Teacher --config modules/ProDiff/config/prodiff_teacher.yaml --exp_name ProDiff_Teacher --reset --hparams='N=8,text=\"Hello, how are you?\",pitch_shift_semitones=2,sigma=0.5'"

os.system(command)
