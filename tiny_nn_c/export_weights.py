"""
Export trained FashionCNN parameters to fashion_cnn_weights.inc.

Assumes your Python model class is named FashionCNN and a trained state_dict
has been saved with torch.save(model.state_dict(), "fashion_cnn_state.pt").

Run:
    python export_weights.py

Then follow the README instructions to replace placeholder arrays in
fashion_cnn.c with generated definitions (or adapt the declarations to
include the generated .inc file).
"""
import numpy as np
import torch
from your_model_file import FashionCNN  # Change this import to your file.

model = FashionCNN()
state = torch.load("fashion_cnn_state.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

names = [
    ("conv1.weight", "conv1_weight"),
    ("conv1.bias",   "conv1_bias"),
    ("conv2.weight", "conv2_weight"),
    ("conv2.bias",   "conv2_bias"),
    ("fc1.weight",   "fc1_weight"),
    ("fc1.bias",     "fc1_bias"),
    ("fc2.weight",   "fc2_weight"),
    ("fc2.bias",     "fc2_bias"),
]

def c_array(name, a):
    a = np.asarray(a, dtype=np.float32)
    dims = "".join(f"[{d}]" for d in a.shape)
    flat = ", ".join(f"{float(v):.9g}f" for v in a.ravel(order="C"))
    return f"static const float {name}{dims} = {{{flat}}};\n"

with open("fashion_cnn_weights.inc", "w", encoding="utf-8") as f:
    for key, c_name in names:
        f.write(c_array(c_name, state[key].numpy()))
        f.write("\n")

print("Wrote fashion_cnn_weights.inc")
