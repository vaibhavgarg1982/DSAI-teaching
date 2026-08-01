# Neural Networks, ML, and AI Tutorials

Hands-on notebooks and scripts for teaching core machine learning and neural network concepts, from gradient descent and K-means to Fashion-MNIST fully connected and CNN models.

## Quick Notes (Version 2)

- [General one-page notes (v2)](<notes_docs/1 page notes v2.png>)
- [CNN one-page notes (v2)](<notes_docs/CNN 1 page v2.png>)
- [CNN advanced topics notes](<notes_docs/CNN-2- Advanced topics.png>)
- [ML/DL terms and definitions (markdown)](<notes_docs/Machine Learning Deep Learning.md>)

## Core notebooks (ordered)

- `0000_grad_des.ipynb`: gradient descent foundations
- `0100_man_grad_des.ipynb`: manual gradient derivation walkthrough
- `0125_class_objects.ipynb`: Python class/object refresher used in examples
- `0150_mlai-2.ipynb`: intermediate ML tutorial notebook
- `0175_titanic_rf_overfit_underfit.ipynb`: random forest overfit/underfit demo
- `0200_titanic- mlai3.ipynb`: Titanic modeling notebook
- `0250_titanic_age_demo.ipynb`: feature engineering and age-focused Titanic demo
- `0300_kmeans_scratch.py`: K-means from scratch with annotated steps
- `0310_colab filtering and k means demo.ipynb`: filtering + K-means walkthrough
- `0400_fashionMnistFC.ipynb`: Fashion-MNIST fully connected model
- `0410_dsdl.ipynb`: additional deep learning tutorial content
- `0500_fashionMnistCNN copy.ipynb`: Fashion-MNIST CNN model
- `0510_convolutions.ipynb`: convolution intuition and implementation examples
- `0600_transfer_learning_ft.ipynb`: transfer learning and fine-tuning experiments
- `0900_Refactoring_training_callbacks.ipynb`: training loop/callback refactoring notes
- `audio-spectrogram.ipynb`: spectrogram experiments for audio-as-image intuition
- `PINN_basic_demo.ipynb`: introductory PINN (Physics-Informed Neural Network) demo

## Additional modules

- `0700_image_classifier/img_cls.ipynb`: image classifier workflow notebook
- `0700_image_classifier/PeepInsideCNN.ipynb`: CNN internals and feature-map inspection
- `full_wf/irisnet.py`: end-to-end Iris model script
- `full_wf/irisnet_dataloader.py`: Iris dataloader-based training workflow
- `full_wf/titanicnet.py`: Titanic model workflow script
- `full_wf/scaffolding.py`: reusable training/evaluation scaffolding

## Data files

- `data/FashionMNIST/raw/*`: local Fashion-MNIST files
- `titanic.csv`: Titanic sample data
- `image_50000.csv`: image-related sample data used in demos
- `freesound_community-g-open-thumb-mid-soft-26660.wav`: sample audio file used in spectrogram/audio experiments
- `tree.dot`: exported decision-tree graph

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

## Notes and docs

- `notes_docs/Machine Learning Deep Learning.md`: ML/DL terms, definitions, and concept notes
- `notes_docs/Post CNN Notes.ipynb`: post-CNN notebook notes
- `notes_docs/Deep Learning Notes-[Tess Ferrandez].pdf`: reference PDF notes

## Printable notes (v2)

![FC](<notes_docs/1 page notes v2.png>)

![CNN](<notes_docs/CNN 1 page v2.png>)
