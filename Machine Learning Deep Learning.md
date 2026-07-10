# Machine Learning & Deep Learning Revision Notes

## Why Machine Learning?

* **Machine Learning** - Learn rules from data.
* **Artificial Intelligence** - Machines performing intelligent tasks.
* **Training** - Learn model parameters.
* **Inference** - Use trained model.
* **Generalization** - Perform well on unseen data.
* **Overfitting** - Memorizes training data.
* **Underfitting** - Model too simple.
* **Function Approximation** - Learn $\hat f(x)\approx f(x)$.

---

# Data

* **Sample** - One data point.
* **Feature** - Input variable.
* **Label / Target** - Desired output.
* **Dataset** - Collection of samples.
* **Batch** - Small subset of data.
* **Epoch** - One complete pass over data.
* **Iteration** - One optimizer update.
* **Training Set** - Learn parameters.
* **Validation Set** - Tune hyperparameters.
* **Test Set** - Final evaluation.
* **Data Leakage** - Test information enters training.
* **Normalization** - Scale to fixed range (common in image/audio).
* **Standardization** - Zero mean, unit variance (common in tabular).
* **One-Hot Encoding** - Categories to binary vectors.
* **Missing Value Imputation** - Fill NaN with median/mean.
* **Stratified Split** - Keep class ratio similar across splits.

---

# Problem Types

* **Regression** - Predict continuous value (ex: age, house price).
* **Classification** - Predict category.
* **Binary Classification** - Two classes (ex: survived / not survived).
* **Multiclass Classification** - One of many classes (ex: Fashion-MNIST).
* **Multilabel Classification** - Multiple labels per sample.
* **Clustering** - Group similar samples without labels.
* **Recommendation** - Predict user-item preference score.
* **Anomaly Detection** - Identify unusual samples. ***Todo***
---

# Neural Networks

* **Neuron** - Weighted sum + activation.
* **Weight** - Trainable coefficient.
* **Bias** - Trainable offset.
* **Parameter** - Weight or bias.
* **Hyperparameter** - User-chosen setting.
* **Layer** - Collection of neurons.
* **Hidden Layer** - Internal computation.
* **Output Layer** - Final prediction.
* **Fully Connected Layer** - Every input connects to every output.
* **MLP / Feedforward Network/ FC network** - Stack of fully connected layers.

---

# Forward Computation

* **Forward Pass** - Compute prediction. No training happens in forward pass.
* **Prediction** - Model output. 
* **Logit** - Raw output before activation. $$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
* **Probability** - Activated output. $$y^{[l]} = \sigma(z^{[l]})$$
* **Inference** - Forward pass only.

---

# Activations

* **Linear** - $f(x)=x$
* **Sigmoid** - $(1+e^{-x})^{-1}$
* **tanh** - Outputs in $[-1,1]$.
* **ReLU** - $\max(0,x)$
* **Leaky ReLU** - Small negative slope.
* **Softmax** - Probability distribution over classes.
* **GELU** - Smooth ReLU variant.

---

# Loss Functions

* **Loss** - Prediction error.
* **Cost** - Average loss over dataset/batch.
* **MSE** - Regression loss.
* **MAE** - Absolute error.
* **Cross Entropy** - Classification loss.
* **Binary Cross Entropy** - Binary classification.
* **BCEWithLogitsLoss** - Sigmoid + BCE in stable form.
* **Categorical Cross Entropy** - Multiclass loss.
* **RMSE** - Root mean squared error (used in rating prediction).

---

# Optimization

* **Gradient** - Direction of steepest increase.
* **Gradient Descent** - Move opposite gradient.
* **Learning Rate** - Step size.
* **Optimizer** - Updates parameters.
* **Backpropagation** - Compute gradients by chain rule.
* **Automatic Differentiation** - Automatic gradient computation.
* **Convergence** - Optimization stabilizes.
* **Local Minimum** - Nearby optimum.
* **Saddle Point** - Neither minimum nor maximum.(minimum wrt some variables, maximum wrt others)
* **Convex Function** - One global minimum.
* **Non-Convex Function** - Multiple minima/saddles.

---

# Gradient Update

* **Gradient Update Rule**

$$
w \leftarrow w-\eta\frac{\partial L}{\partial w}
$$

---

# Optimizers

* **SGD** - Basic gradient descent.
* **Momentum** - Velocity-based SGD.
* **Nesterov** - Look-ahead momentum.
* **AdaGrad** - Adaptive learning rates.
* **RMSProp** - Running squared gradients.
* **Adam** - Momentum + RMSProp.
* **AdamW** - Adam with decoupled weight decay.
* **EWMA** - Exponentially weighted moving average.
* **Bias Correction** - Remove early-step EWMA bias.

---

# Learning Rate Scheduling

* **Step Decay** - Lower LR at fixed milestones.
* **Exponential Decay** - Multiply LR by a fixed factor.
* **Polynomial Decay** - LR follows polynomial decrease.
* **Warmup** - Start small, increase LR gradually.
* **Cosine Annealing** - LR follows cosine curve.
* **SGDR** - Cosine annealing with warm restarts.
* **OneCycleLR** - Increase then decrease LR in one cycle.
* **ReduceLROnPlateau** - Reduce LR when validation stalls.

---

# Regularization & Stabilization

* **Regularization** - Reduce overfitting.
* **L1** - Sparse weights.
* **L2 / Weight Decay** - Penalize large weights.
* **Dropout** - Randomly disable neurons during training.
* **Early Stopping** - Stop before overfitting grows.
* **Batch Normalization** - Normalize mini-batch activations.
* **Layer Normalization** - Normalize features per sample.
* **Data Augmentation** - Artificially enlarge dataset. PyTorch transforms, eg. random crop, flip, rotate, color jitter.
* **Vanishing Gradients** - Gradients become too small.
* **Exploding Gradients** - Gradients become too large.

---

# Initialization

* **Xavier / Glorot Init** - Good default for tanh/sigmoid-like setups.
* **Kaiming / He Init** - Good default for ReLU-like activations.
* **Goal of Init** - Keep activation/gradient scales stable across layers.

---

# Model Evaluation

* **Accuracy** - Correct predictions ratio.
* **Precision** - Correct positive predictions.
* **Recall** - Fraction of true positives found.
* **F1 Score** - Balance of precision and recall.
* **Confusion Matrix** - Count TP, FP, TN, FN by class.
* **ROC Curve** - TPR vs FPR across thresholds.
* **AUC** - Area under ROC curve.

---

# CNNs

* **CNN** - Learn spatial features from images.
* **Convolution** - Sliding filter over input.
* **Kernel / Filter** - Learnable convolution weights.
* **Stride** - Step size of filter movement.
* **Padding** - Border extension for shape control.
* **Feature Map** - Convolution output tensor.
* **Pooling** - Downsampling operation.
* **Max Pooling** - Keep local maximum.
* **Average Pooling** - Keep local average.
* **Flatten** - Tensor to vector before FC layer.
* **1x1 Convolution** - Channel mixing/reduction with low compute.
* **Inception Block** - Parallel conv branches fused together.
* **Depthwise Convolution** - One filter per input channel.
* **Pointwise Convolution** - $1\times1$ convolution across channels.
* **Depthwise Separable Convolution** - Depthwise + pointwise for efficiency.

---

# Parameter Counting

* **Linear Layer**

$$
P=n_{in}n_{out}+n_{out}
$$

* **Conv2D Layer**

$$
P=(k_hk_wn_{in}+1)n_{out}
$$

* **BatchNorm Layer**

$$
2\times \text{channels} \quad (\gamma,\beta)
$$

---

# Transfer Learning

* **Pretrained Model** - Backbone already trained on large data.
* **Feature Extractor** - Freeze backbone, train small head.
* **Fine-Tuning** - Unfreeze selected layers and continue training.
* **Classifier Head Replacement** - Swap final layer for new classes.
* **Domain Adaptation by Transforms** - Resize and channel conversion (ex: grayscale to 3-channel).

---

# Clustering

* **K-Means** - Iterative centroid-based clustering.
* **Centroid** - Mean of points in a cluster.
* **Initialization** - Pick initial centroids (often random points).
* **Assignment Step** - Assign each point to nearest centroid.
* **Update Step** - Recompute centroid as cluster mean.
* **Distance Metric** - Usually Euclidean distance.
* **Convergence** - Stop when centroids stop moving or max iters reached.
* **From-Scratch Loop** - Init -> assign -> update -> repeat.

---

# Recommendation Systems

* **Collaborative Filtering** - Learn from user-item interaction patterns.
* **User Embedding** - Latent vector for user preferences.
* **Item Embedding** - Latent vector for item properties.
* **Matrix Factorization** - Approximate rating matrix as user x item factors.
* **Predicted Rating** - Dot product of user/item embeddings (+ bias terms).
* **Rating Loss** - MSE/RMSE on known interactions.

---

# Audio & Spectrograms

* **Waveform** - Signal amplitude over time.
* **Frequency Domain** - Signal represented by frequencies.
* **FFT** - Fast Fourier Transform for frequency analysis.
* **Spectrogram** - Time-frequency energy map.
* **Window Function** - Hann/Hamming to reduce spectral leakage.
* **nperseg** - Samples per analysis window.
* **noverlap** - Overlap between adjacent windows.
* **dB Scale** - Log-scale magnitude, often $20\log_{10}(x)$.
* **Mono Conversion** - Convert multi-channel audio to single channel.

---

# Tabular ML Patterns (Titanic-style)

* **Tabular Data** - Rows are samples, columns are features.
* **Numerical Features** - Continuous or integer-valued columns.
* **Categorical Features** - Non-numeric categories requiring encoding.
* **Nominal Features** - Categorical values with no natural order.
* **Ordinal Features** - Categorical values with meaningful order.
* **Binary Features** - Two possible values (0/1, yes/no).
* **Datetime Features** - Time-based columns transformed into useful parts.
* **Feature Engineering** - Create better predictors from raw columns.
* **One-Hot Encoding** - Convert each category into a binary indicator column.
* **drop_first in One-Hot** - Avoid dummy-variable redundancy in linear models.
* **Ordinal Encoding** - Map ordered categories to ordered integers.
* **Label Encoding** - Integer codes for categories (use carefully for models sensitive to order).
* **High Cardinality** - Too many unique categories; can inflate one-hot dimension.
* **Missing Numerical Values** - Impute with mean/median.
* **Missing Categorical Values** - Impute with mode or a dedicated "Unknown" class.
* **StandardScaler** - Zero mean and unit variance scaling.
* **MinMaxScaler** - Scale values into a fixed range (often [0,1]).
* **RobustScaler** - Scale using median/IQR; less sensitive to outliers.
* **Train-Fit / Test-Transform Rule** - Fit encoders/scalers on train only.
* **Data Leakage Guard** - Never learn preprocessing statistics from test/validation.
* **Target for Classification** - Discrete label (ex: survived / not survived).
* **Target for Regression** - Continuous value (ex: age).

---

# PyTorch Essentials

* **Tensor** - Multidimensional array.
* **Module** - Neural network class.
* **Parameter** - Trainable tensor.
* **Forward()** - Defines computation.
* **Backward()** - Computes gradients.
* **Optimizer.step()** - Update weights.
* **zero_grad()** - Clear gradients.
* **Dataset** - Data interface.
* **DataLoader** - Batch iterator.
* **model.train()** - Training mode.
* **model.eval()** - Inference mode.
* **torch.no_grad()** - Disable gradients.

---

# Universal Training Loop

1. Forward pass
2. Compute loss
3. Zero gradients
4. Backward pass
5. Optimizer step
6. Repeat

---

# Practical Assets in This Workspace

* **FashionMNIST Raw Files** - Image classification dataset for FC/CNN notebooks.
* **titanic.csv** - Structured tabular dataset for classification/regression demos.
* **image_50000.csv** - Image-like tabular representation for ML experiments.
* **Sample WAV File** - Audio source for spectrogram notebook.

---

# The Big Picture

* **Programming** - Humans write exact rules.
* **Machine Learning** - Computer learns rules from examples.
* **Deep Learning** - Learn complex functions with layered representations.
* **Everything is Function Approximation** - Fit $\hat f(x)$ by minimizing loss.
* **Everything is Optimization** - Update parameters to reduce loss.
