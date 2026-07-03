# Neural Networks, ML, and AI Tutorials

Hands-on notebooks and scripts for teaching core machine learning and neural network concepts, from gradient descent and K-means to Fashion-MNIST fully connected and CNN models.

## Quick Notes (Version 2)

- [General one-page notes (v2)](<1 page notes v2.png>)
- [CNN one-page notes (v2)](<CNN 1 page v2.png>)

## What is in this repo

- `grad_des.ipynb` and `man_grad_des.ipynb`: gradient descent intuition and manual derivation walkthroughs
- `man_grad_des.html`: exported version of manual gradient descent notes
- `kmeans_scratch.py`: K-means implemented from scratch with annotated steps
- `fashionMnistFC.ipynb`: Fashion-MNIST classification with a fully connected network
- `fashionMnistCNN copy.ipynb`: CNN-based Fashion-MNIST classification
- `audio-spectrogram.ipynb`: early experiments showing audio as image-like/spectrogram data
- `titanic_age_demo.ipynb` and `titanic- mlai3.ipynb`: Titanic dataset examples
- `mlai-2.ipynb`: additional ML/AI tutorial notebook

## Data files

- `data/FashionMNIST/raw/*`: local Fashion-MNIST files
- `titanic.csv`: Titanic sample data
- `image_50000.csv`: image-related sample data used in demos

## Environment

Project metadata and dependencies are managed in `pyproject.toml` and include:

- PyTorch / TorchVision
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn
- tqdm, torchinfo, ipykernel

Target Python in this project: `>=3.14`.

## Run locally

1. Install dependencies (using `uv`):
	```bash
	uv sync
	```
2. Activate the environment:
	```powershell
	.venv\Scripts\Activate.ps1
	```
3. Launch Jupyter and open any notebook:
	```bash
	jupyter notebook
	```

Notes:
![FC](<1 page notes v2.png>)

![CNN](<CNN 1 page v2.png>)
