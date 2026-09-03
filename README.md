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
- `0120_common_activations_non_linearities.ipynb`: activation functions and nonlinearities
- `0125_class_objects.ipynb`: Python class/object refresher used in examples
- `0150_mlai-2.ipynb`: intermediate ML tutorial notebook
- `0175_titanic_rf_overfit_underfit.ipynb`: random forest overfit/underfit demo
- `0200_titanic- mlai3.ipynb`: Titanic modeling notebook
- `0200_titanic-overunder.ipynb`: Titanic neural-network overfit/underfit demo
- `0250_titanic_age_demo.ipynb`: feature engineering and age-focused Titanic demo
- `0300_dimensionality_reduction.ipynb`: SVD and PCA dimensionality-reduction walkthrough
- `0301_dimensionality_reduction_autoenc.ipynb`: autoencoder-based dimensionality reduction
- `0305_kmeans_scratch.py`: K-means from scratch with annotated steps
- `0308_anomaly_gaussian.ipynb`: multivariate Gaussian anomaly detection
- `0310_colab filtering and k means demo.ipynb`: filtering + K-means walkthrough
- `0400_fashionMnistFC.ipynb`: Fashion-MNIST fully connected model
- `0410_dsdl.ipynb`: additional deep learning tutorial content
- `0500_fashionMnistCNN copy.ipynb`: Fashion-MNIST CNN model
- `0510_convolutions.ipynb`: convolution intuition and implementation examples
- `0600_transfer_learning_ft.ipynb`: transfer learning and fine-tuning experiments
- `0800_char_rnn.ipynb`: character-level language modeling with manual RNN and PyTorch LSTM examples
- `0900_Refactoring_training_callbacks.ipynb`: training loop/callback refactoring notes
- `0901_tensorboard.ipynb`: TensorBoard experiment tracking
- `0999_MMR.ipynb`: Max Marginal Relevance retrieval reranking
- `audio-spectrogram.ipynb`: spectrogram experiments for audio-as-image intuition
- `PINN_basic_demo.ipynb`: introductory PINN (Physics-Informed Neural Network) demo

## Topics covered in the 0xxx notebooks

### Engineering aspects of DL and ML
-  Overfitting and underfitting
-  Train/test splitting and stratification
-  Tracking loss and accuracy over epochs
-  Training, validation and test set discipline
-  Why loss functions and metric that matters might be different.
-  Data leakage, example of audio classification and spectrograms project, where test set included random spectrograms from the raw set, which implied (since there was an overlap by design) that it was being evaluated on data already seen by the model during training. This led to 98%+ accuracy on the test set, which was misleading and not representative of real-world performance.

### `0000_grad_des.ipynb`
- defining and evaluating loss functions for optimization
- computing partial derivatives with respect to multiple parameters
- learning-rate choice and its effect on update size
- simultaneous parameter updates for gradient descent
- convergence checks based on loss change
- using PyTorch autograd with `requires_grad=True`
- tracking loss over iterations and plotting convergence on a log scale

### `0100_man_grad_des.ipynb`

- mean squared error for linear regression
- manual derivation of gradients for slope and intercept
- why gradients should be computed before both parameters are updated
- visualizing optimization on 3D loss surfaces and contour plots
- using `.backward()` for automatic differentiation
- applying parameter updates inside `torch.no_grad()`
- zeroing gradients between iterations with `.grad.zero_()`
- detaching tensors before plotting or analysis

### `0125_class_objects.ipynb`

- Python object-oriented programming with `__init__`, `__call__`, `__str__`, and `__repr__`
- how instance state and methods work together
- subclassing `nn.Module` and implementing `forward()`
- using dataclasses to package training state cleanly
- abstract base classes for defining callback contracts
- train/test splitting with a fixed `random_state`
- converting features and labels into correctly typed tensors
- multi-class classification with `CrossEntropyLoss`
- callback hook design for training lifecycle events

### `0150_mlai-2.ipynb`

- converting NumPy arrays into PyTorch tensors with explicit dtypes
- building models with `nn.Linear` and `nn.Sequential`
- ReLU, Sigmoid, Tanh, and Softmax activation behavior
- choosing MSE for regression versus cross-entropy for classification
- the standard optimization loop: zero gradients, backpropagate, step optimizer
- using `model.eval()` and `torch.no_grad()` for inference
- applying `argmax` to logits for multi-class predictions
- binary-output reasoning versus multi-class output reasoning

### `0175_titanic_rf_overfit_underfit.ipynb`

- stratified train/test splitting to preserve class balance
- one-hot encoding categorical variables with `drop_first=True`
- handling numeric and categorical tabular features before modeling
- controlling random forest complexity with depth and split constraints
- overfitting versus underfitting in tree-based models
- out-of-bag scoring for ensemble validation
- feature importance interpretation
- why accuracy can be misleading on imbalanced data
- precision, recall, F1 score, and confusion-matrix analysis
- exporting and visualizing decision trees

### `0200_titanic- mlai3.ipynb`

- scaling numeric features with `StandardScaler`
- fitting preprocessing on train data only to avoid leakage
- combining scaled numeric inputs with one-hot encoded categorical inputs
- binary classification with a neural network
- using `BCEWithLogitsLoss` for numerical stability
- thresholding sigmoid outputs for class predictions
- weight decay as L2 regularization - mention Dropout etc. Also mention early stopping and learning rate scheduling
- train versus validation loss tracking to spot overfitting
- confusion-matrix heatmaps for error analysis
- switching between `model.train()` and `model.eval()` correctly

### `0250_titanic_age_demo.ipynb`

- framing age prediction as a regression problem
- MSE loss for continuous targets
- designing a multi-layer perceptron for regression
- hidden-layer ReLU activations with a linear output layer
- training-loop mechanics for regression models
- using weight decay to regularize the model
- comparing train and test loss to detect overfitting
- evaluating predictions with correlation and scatter plots
- interpreting predicted-versus-actual plots for model quality

### `0300_dimensionality_reduction.ipynb`

- singular value decomposition (SVD) and its low-rank approximation
- principal component analysis (PCA) for projecting data into fewer dimensions
- selecting the number of components using explained variance
- visualizing reduced-dimensional data

### `0301_dimensionality_reduction_autoenc.ipynb`

- encoder-decoder networks for dimensionality reduction
- latent-dimension selection and ReLU activations in the encoder
- reconstruction loss and reconstruction-quality evaluation

### `0308_anomaly_gaussian.ipynb`

- multivariate Gaussian anomaly detection
- Mahalanobis distance for measuring multivariate deviation
- selecting an anomaly threshold from the estimated distribution

### `0310_colab filtering and k means demo.ipynb`

- K-means from scratch with initialization, assignment, and centroid update steps
- Euclidean distance for cluster assignment
- fixed-iteration versus convergence-based stopping logic
- normalizing vectors with the L2 norm before clustering
- plotting clustered data and centroids
- comparing a manual K-means implementation with `sklearn.cluster.KMeans`
- collaborative filtering with a user-item rating matrix
- matrix factorization into user and item embeddings
- RMSE-style reconstruction loss for recommendations
- using `nn.Embedding` for learnable latent factors
- SGD-based optimization for recommendation models

### `0400_fashionMnistFC.ipynb`

- the FashionMNIST dataset structure and task setup
- batching and shuffling with DataLoader
- flattening images from `28x28` into feature vectors
- building a fully connected classifier for images
- using `CrossEntropyLoss` for 10-class classification
- separating training mode from evaluation mode
- computing accuracy from class logits with `argmax`
- summarizing model architecture and parameter counts
- tracking epoch and batch progress with `tqdm`

### `0410_dsdl.ipynb`

- implementing a custom Dataset with `__len__` and `__getitem__`
- traversing directory-based datasets
- mapping class names to integer labels
- loading images with PIL and normalizing channel format
- composing transforms for resize, grayscale conversion, and tensor conversion
- applying transforms dynamically during sample retrieval
- pairing a custom Dataset with a DataLoader

### `0500_fashionMnistCNN copy.ipynb`

- configuring convolution layers with channels, kernels, and padding
- max pooling for spatial downsampling
- tracking tensor shapes through a CNN pipeline
- flattening convolutional feature maps before dense layers
- separating feature extraction from final classification
- weight decay for CNN regularization
- training and evaluation loops for image classifiers
- computing accuracy from predicted classes
- learning rate scheduling with `one cycle_lr` etc. (See Pytorch docs)

### `0510_convolutions.ipynb`

- designing edge-detection kernels by hand
- understanding convolution as element-wise multiply then sum
- applying kernels to image patches manually
- reasoning about batch, channel, height, and width dimensions
- using `F.conv2d()` with correctly shaped image and kernel tensors
- applying multiple kernels to multiple images in batch form
- zero padding to preserve spatial dimensions
- stride and its effect on output size
- the convolution output-size formula
- adding dimensions with `unsqueeze()`
- visualizing feature maps and comparing kernel effects

### `0600_transfer_learning_ft.ipynb`

- loading pretrained models for downstream tasks
- replacing the classification head for a new number of classes
- freezing backbone parameters with `requires_grad = False`
- unfreezing or training only selected layers
- optimizing only the trainable subset of parameters
- adapting grayscale images to RGB for pretrained vision models
- resizing and preprocessing inputs to match model expectations
- transfer learning as a way to reduce compute and data demands
- timing forward, backward, and optimizer steps during training

### `0800_char_rnn.ipynb`

- character-level language modeling with a small training corpus
- vocabulary construction, sequence windows, and next-character targets
- a manual recurrent neural network and a PyTorch LSTM implementation
- text generation from predicted character distributions

### `0900_Refactoring_training_callbacks.ipynb`

- refactoring the training loop into reusable abstractions
- callback base classes and hook methods
- early stopping with patience and best-model tracking
- logging progress at epoch boundaries
- storing loss histories for later analysis
- using a dataclass as a callback context object
- encapsulating training behavior in a Trainer class
- restoring the best model weights with saved state dictionaries
- separating orchestration concerns from model definition
- training lifecycle flow from begin, to per-epoch hooks, to end

### `0901_tensorboard.ipynb`

- writing training metrics for TensorBoard visualization
- inspecting loss and experiment runs with TensorBoard

### `0999_MMR.ipynb`

- Max Marginal Relevance (MMR) for balancing relevance and diversity
- vectorized selection and fetch-then-rerank retrieval approaches

## Additional modules

- `0700_image_classifier/img_cls.ipynb`: image classifier workflow notebook
- `0700_image_classifier/PeepInsideCNN.ipynb`: CNN internals and convolutional feature-map inspection
- `full_wf/irisnet.py`: end-to-end Iris model script
- `full_wf/irisnet_dataloader.py`: Iris dataloader-based training workflow
- `full_wf/titanicnet.py`: Titanic model workflow script
- `full_wf/scaffolding.py`: reusable training/evaluation scaffolding

## Data files

- `data/FashionMNIST/raw/*`: local Fashion-MNIST files
- `titanic.csv`: Titanic sample data
- `image_50000.csv`: image-related sample data used in demos
- `0309_colab-filtering.xlsx`: collaborative-filtering optimization exercise using Excel Solver
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

- `notes_docs/Machine Learning Deep Learning.md`: ML/DL terms, definitions, concept notes, and default loss values
- `notes_docs/Post CNN Notes.ipynb`: post-CNN notebook notes
- `notes_docs/Deep Learning Notes-[Tess Ferrandez].pdf`: reference PDF notes

## Printable notes (v2)

![FC](<notes_docs/1 page notes v2.png>)

![CNN](<notes_docs/CNN 1 page v2.png>)

## Progression
1. Session 1 - Machine learning classification, classification based on learning methods, Slope(gradient)

2. Session 2 - Matrix multiplication as affine functions, chain rule, functions, composition, composition of affine functions. Scalar, vector, tensor. 

3. Session 3 - gradient descent, hadamard product, L2 norm and frobenious norm.

4. Session 4 - Machine Learning  libraries ,bias, MSE , MAE. 

5. Session 5 - functions, optimization, loss functions, gradient descent, manual demo with linear data, inference and training, generalisability of loss functions. OOPs basics.

6. Session 6 - Extend with pytorch concepts, Autograd, optim.SGD, MSE loss etc.

7. Session 7 - Neuron , Universal Approximation theorem, Brief of Historical milestones in field of NN, XOR problem, Selection of non linearity, Sigmoid function.

8. Session 8 - Common Non Linear Activations( Sigmoid, tanh, ReLU, leakage ReLU, ELU, Soft Max), IRIS data set, Multi Layer NN, basic FC/dense, Multi class classification, nll and cross entropy loss.

9. Session 9 - Numerical Data, Catagorical ( Ordinal and Non-Ordinal) Data, One-Hot encoding, Drop First, Train set, validation/dev set, Test set, Overfitting,  Under-fitting, Fan-In, Fan-Out.

10. Session 10 - Titanic data set, handling of missing data, stratify and deterministic randomisation in train-test split, F1 Score, Precision and recall.

11. Session 11 - end to end MLP design, overfit and underfit, Regularisation, L2 reg and weight decay.

12. Session 12 - Unsupervised learning use cases, Dimensionality reduction, wine dataset, SVD and PCA, Auto encoder, Auto encoder as anomaly detection.

13. Session 13 - Clustering (K- means) , anomaly detection, Multivariate Gaussian, Recommendation engine, collaborative filtering.

14. Session 14 - Collaborative Filtering, Dataset and dataloaders introduction.

15. Session 15 - Exponentially Weighted Moving average(EWMA), momentum, RMSProp, Adam, Adam with decoupled weights (AdamW), Image representation,  training a FC model with Fashion-MNIST image data set after flatten, concepts of spatial invariance violation in FC for images

16. Session 16 - Convolution, feature extraction from image using convolution, 2D convolution, CNN benefits, padding, stride, maxpool, multiple filters.

17. Session 17 - CNN as combination of feature extractor and classifier/regressor FC network, small CNN used on FashionMnist. Data engineering and EDA, baseline naive Bayesian prior models and single number metrics, default loss values for different use cases as sanity checks, robust data split strategies as the starting points reiterated.

18. Session 18 - End to End CNN flow with emphasis on data engineering principles, train test split for temporally arranged data, data leakage.


### A few Recommended papers and references that came up during the course of the sessions

| Link | Title |
| --- | --- |
| [arXiv:1802.01528](https://arxiv.org/pdf/1802.01528v3.pdf) | The Matrix Calculus You Need For Deep Learning |
| [arXiv:1604.06737](https://arxiv.org/pdf/1604.06737.pdf) | Entity Embeddings of Categorical Variables |
| [arXiv:1706.03762](https://arxiv.org/pdf/1706.03762.pdf) | Attention Is All You Need |
| [arXiv:1512.03385](https://arxiv.org/pdf/1512.03385v1.pdf) | Deep Residual Learning for Image Recognition |
| [arXiv:1502.03167](https://arxiv.org/pdf/1502.03167.pdf) | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift |
| [arXiv:2003.05991](https://arxiv.org/pdf/2003.05991.pdf) | Autoencoders |
| [arXiv:1805.11604](https://arxiv.org/pdf/1805.11604) | How Does Batch Normalization Help Optimization? |
| [arXiv:1708.07120](https://arxiv.org/pdf/1708.07120) | Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates |
| [arXiv:1812.01187](https://arxiv.org/pdf/1812.01187.pdf) | Bag of Tricks for Image Classification with Convolutional Neural Networks |
| [arXiv:1710.09412](https://arxiv.org/pdf/1710.09412.pdf) | mixup: Beyond Empirical Risk Minimization |
| [arXiv:1911.02685](https://arxiv.org/pdf/1911.02685.pdf) | comprehensive survey on Transfer Learning |