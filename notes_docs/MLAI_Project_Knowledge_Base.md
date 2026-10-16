# MLAI Tutorials — Project Knowledge Base

## 1. Project identity and goal

The project is a practical, conceptually rigorous introduction to machine learning and deep learning, taught by Vaibhav to his son (an incoming engineering student) and a small cohort of R&D professionals. It is not intended to be a generic “AI for beginners” course. The learners should acquire a connected mental model of the field, understand why methods exist, and progressively become able to derive, implement, evaluate, and improve them.

The course has two related but distinct contexts:

1. **Teaching the son:** a mathematically comfortable incoming engineering student, with the freedom to move at a university-style pace and connect ML to mathematics, physics, and engineering.
2. **Teaching a cohort of R&D professionals:** mixed backgrounds; use the same conceptual rigor but adapt depth and pacing to the audience.

The project’s central educational idea is to reconstruct ML as an evolving set of solutions to identifiable problems—not as a catalogue of algorithms to memorize.

## 2. Teaching philosophy and interaction contract

### 2.1 Multi-pass teaching (user-established)

The instructor explicitly describes himself as a **multi-pass teacher**.

- **Pass 1 — Orientation:** demonstrations, concepts, narratives, vocabulary, and the overall map.
- **Pass 2 — Mechanism:** how each component actually works.
- **Pass 3 — Derivation:** mathematical details and first-principles reasoning.
- **Pass 4 — Implementation:** coding and framework mechanics.
- **Pass 5 — Extension:** limitations, alternatives, optimization, and generalization.

Not every concept must be mastered on first exposure. Early exposure is deliberate: learners should see the destination before all prerequisites are in place. Subsequent passes replace fuzzy understanding with precision.

### 2.2 Start with the whole machine, then unpack it

The teaching sequence initially went **top-down**, prompted by the learner asking whether ML requires too much mathematics. The instructor presented a working neural network/backpropagation picture first, then used it to motivate the required ideas.

The framing is roughly:

> We have a parameterized function. It makes predictions, incurs error, and we need to know how to change its parameters to reduce that error.

This makes later concepts purposeful:
- derivatives tell us how outputs change;
- the chain rule handles compositions of functions;
- matrices package many linear equations and permit efficient computation;
- nonlinear activations prevent a stack of linear layers from collapsing into one linear transformation;
- gradient descent uses derivatives to update parameters;
- batches trade off gradient-noise, computation, and memory.

**Important:** Do not insist that a learner must be able to reproduce a derivation before they are allowed to understand what it means. The first pass aims at conceptual ownership, not exam-style reproduction.

### 2.3 Build a connected semantic network, not a dictionary of isolated definitions

Vocabulary is important because it allows learners to read documentation, papers, blogs, repositories, and lectures. However, definitions should be relational.

Example:
- A parameter is a quantity the optimizer may change.
- Weights and biases are the usual trainable parameters of a neural network.
- A forward pass produces predictions.
- Predictions are compared with targets to compute a loss.
- Backpropagation differentiates the loss with respect to parameters.
- The optimizer uses those gradients to update parameters.

Terms should be introduced with their role in the system and linked to related concepts.

### 2.4 Explain why a concept exists

A recurring teaching prompt should be **“Why does this exist?”** Before introducing a new algorithm, architecture, loss, or training trick, identify the limitation of the previous approach that motivates it.

Examples:
- Why do we need hidden layers?
- Why do we need nonlinear activations?
- Why are CNNs suited to images?
- Why is cross-entropy useful for classification?
- Why does regularization help?
- Why did sequence modelling evolve from RNNs toward attention and transformers?

### 2.5 Tone and response preferences for future AI collaborator

- Be concise but technically substantive; do not pad responses with generic encouragement.
- Give clean narratives when requested, without meta-commentary about the narrative.
- Be precise about distinctions, assumptions, tensor shapes, and implementation details.
- Correct misconceptions directly and explain the reason.
- Do not assume a question is a request to redesign the curriculum; answer the specific question first.
- Treat the instructor as a knowledgeable technical collaborator, not a novice.
- Preserve the instructor’s authorship. Help test, refine, and structure ideas rather than taking over unnecessarily.
- No emojis in informational responses.
- Do not end responses with a question or an offer.
- If code is requested, prefer clear, executable code and explain non-obvious choices.

---

## 3. Learner context and pedagogical implications

### Son
- Entering engineering; mathematically comfortable.
- The course can use undergraduate-level ideas and first-principles explanations.
- He has already worked through a neural-network example, understood ReLU, dimensions, matrix multiplication, function composition, and the chain rule at an intuitive level.
- He manually used the chain rule in backpropagation; when asked why, he replied, “What else is there?” This is a positive sign that the rule is becoming a reasoning tool rather than a memorized theorem.
- He counted parameters in a layer such as `nn.Linear(784, 512)`, connecting API syntax to the count \(784\times512+512\).
- The goal is not to demand that he reproduce every derivation immediately; it is to ensure that he understands what each component is doing.

### R&D cohort
- Seven professionals were mentioned in the teaching context; backgrounds are mixed.
- The material should remain grounded in practical engineering examples while retaining mathematical integrity.
- Cheat sheets, posters, notebooks, and concise visual aids have been used or requested.

---

## 4. Course architecture: broad map

A broad long-term curriculum was sketched. Treat it as a **roadmap**, not as a commitment that every module has already been taught.

1. **Why AI/ML?** Intelligence, classical AI vs learning from examples, function approximation, generalization vs memorization.
2. **Mathematical foundations:** functions and composition, vectors, matrices, linear transformations, probability, expectations, derivatives, partial derivatives, chain rule, convexity intuition.
3. **Optimization:** objectives, loss/cost, gradients, gradient descent, learning rate, batch/mini-batch/stochastic methods, momentum.
4. **Data:** samples, features, labels, missing data, numerical/categorical variables, encoding, normalization/standardization, splits, bias/variance, leakage.
5. **Classical ML:** linear/logistic regression, k-NN, decision trees, random forests, SVMs, Naïve Bayes, clustering, PCA.
6. **Evaluation:** training/validation/test performance, accuracy, precision, recall, F1, confusion matrix, ROC/AUC, MAE/RMSE, baselines.
7. **Neural networks:** neuron/perceptron, hidden layers, activations, loss, feedforward networks, backpropagation, vanishing/exploding gradients, universal approximation.
8. **Deep learning:** CNNs, pooling, feature hierarchies, representation learning, transfer learning, autoencoders.
9. **Sequence models:** time series, language, RNN/LSTM/GRU and their limitations.
10. **Attention:** attention, self-attention, multi-head attention, positional encoding.
11. **Transformers:** encoder/decoder, encoder-decoder models, GPT/BERT, scaling, context windows.
12. **Generative AI:** next-token prediction, sampling, temperature, beam search, hallucinations, fine-tuning, instruction tuning, RLHF, RAG.
13. **Modern AI:** vision and multimodal models, agents, tool use, reasoning models, diffusion, modern reinforcement learning.
14. **Engineering:** NumPy, pandas, Matplotlib, scikit-learn, PyTorch, Hugging Face, ONNX, CUDA/GPUs.
15. **Deployment:** inference, latency, throughput, quantization, edge AI, serving, APIs, monitoring.
16. **Ethics and philosophy:** bias, explainability/interpretability, fairness, privacy, copyright, AI safety, and the meaning of intelligence.

### Proposed capstone progression
1. Linear regression from scratch in Python.
2. Gradient descent from scratch.
3. Logistic regression from scratch.
4. Tiny neural network with manual backpropagation.
5. Same network in NumPy.
6. Same network using PyTorch autograd.
7. CNN on MNIST.
8. Character-level language model.
9. Tiny transformer from scratch.
10. Fine-tune an open-source language model.
11. Build a RAG system.
12. Deploy it as a web service.

Again, this is a proposed longer-range roadmap, not a claim that all stages have been completed.

---

## 5. Teaching sequence actually evidenced in the project

The project conversation describes a sequence in which each notebook exposes a limitation and motivates the next concept. The clearest established sequence is:

1. **A working neural network/backpropagation overview** — the initial “map” of the territory.
2. **Terminology/taxonomy pass** — inputs, outputs, loss functions, activations, regularization, architecture types, and relevant history.
3. **Simple regression and optimization** — linear models, gradient descent, and the training loop.
4. **Automatic differentiation and PyTorch** — understanding what autograd does and why.
5. **`nn.Module` and network structure** — translating the mathematical model into framework code.
6. **Iris/classification** — moving from regression to classification while reusing the same training machinery.
7. **Titanic tabular classification** — preprocessing and train/test workflow, with a small MLP.
8. **Overfitting and regularization** — train/test gap, L2/weight decay, dropout/early stopping concepts, and learning curves.
9. **CNN/image work and transfer learning** — later project discussions include CNN mechanics, ResNet, transfer learning, and fine-tuning.

The conversation includes a number of later technical follow-ups. Do not infer that every topic in the broad roadmap was taught in sequence; some were exploratory discussions or later follow-up questions.

---

## 6. Core conceptual model of ML

A useful recurring architecture map is:

```text
Input data
   ↓
Representation / preprocessing
   ↓
Model architecture
   ↓
Activation choices
   ↓
Prediction / logits
   ↓
Loss function  ←  target / label
   ↓
Backpropagation (gradients)
   ↓
Optimizer and parameter update
   ↓
Evaluation on validation/test data
   ↓
Regularization, model selection, iteration
```

The main design dimensions are:
- input modality and representation;
- model family/architecture;
- activation functions;
- output interpretation;
- loss function;
- optimizer;
- regularization;
- data and evaluation protocol.

The key unification is that many ML methods differ primarily in the parameterized function being fitted. Much of the training machinery—loss, gradients, optimizer, batches, epochs—can remain the same.

A useful board-level comparison:
\[
y=mx+c,\qquad y=ax^2+bx+c,\qquad
y=W_2\,\mathrm{ReLU}(W_1x+b_1)+b_2.
\]
These are different families of functions, but the fitting/optimization framework is conceptually shared.

---

## 7. Datasets, notebooks, and implementation decisions

### 7.1 Iris / early neural-network notebooks
- Used as a simple early classification problem.
- The intention was to preserve the training loop while changing the task/model/data only as needed.
- Mathematical and tensor dimensions should be made explicit.
- Count parameters manually where pedagogically useful.

### 7.2 Titanic tabular classification
The Seaborn Titanic dataset was recommended for live teaching rather than the Kaggle competition download, because it avoids irrelevant file-handling and competition setup.

Example:
```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

The conversation discusses keeping a compact feature set, including:
- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`
- sex-derived features
- target `survived`

Preprocessing lessons and decisions:
- Handle missing values explicitly.
- Categorical variables require encoding.
- For a binary categorical feature such as sex, a single 0/1 column can represent the two categories; full one-hot encoding is also useful pedagogically.
- For nominal categories with more than two values, arbitrary integer encoding falsely introduces an order.
- `pclass` is a useful discussion point: it is numeric-looking but ordinal; treating it numerically is defensible, but the assumption should be explicit.
- Scaling continuous features such as `age` and `fare` is reasonable.
- For teaching clarity, leaving binary indicators and small count/ordinal variables unscaled may avoid muddying their interpretation. This is a pedagogical choice, not a universal rule.
- **Avoid data leakage:** split first; fit the scaler on the training data only; transform training and test data with the training-set statistics.
- The test set should not influence preprocessing parameters.
- Keep the target separate from the features and ensure expected tensor dtypes/shapes.

Representative tensor conversion:
```python
X = df.drop("survived", axis=1).values.astype("float32")
y = df["survived"].values.astype("float32")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
```

The precise preprocessing code evolved during discussion; when continuing, inspect the current notebook before assuming which version is canonical.

### 7.3 Small Titanic MLP
A small model of the form `6 → 4 → 1` was discussed, with 33 trainable parameters. Results around 87.8% accuracy were considered plausible for this dataset and model, but should be verified against the actual split and current code rather than treated as a universal benchmark.

Teaching points:
- Compare against a majority-class baseline; raw accuracy alone has no context.
- Titanic is small, noisy, incomplete, and limited by missing predictive information. A more complex model cannot recover information absent from the features.
- Inspecting hidden-layer weights is useful for asking whether neural-network parameters are directly interpretable; generally, they are not as straightforward as coefficients in a simple linear model.
- Compare with logistic regression to show that a neural network is not automatically the right model for every task.
- Avoid wasting learner attention on thousands of epochs after the loss has plateaued; use a reasonable epoch budget and plot learning curves.

### 7.4 Classification outputs and loss functions
A recurring distinction:
- **Logits** are raw model outputs; they are not probabilities.
- Sigmoid maps one logit to a binary probability.
- Softmax maps multiclass logits to a distribution across classes.
- The appropriate loss depends on the prediction task and output representation.

Binary example:
```python
criterion = torch.nn.BCEWithLogitsLoss()
logits = model(X)
loss = criterion(logits, y)
```
Do **not** put a sigmoid in the model when using `BCEWithLogitsLoss`; the loss combines sigmoid and binary cross-entropy in a numerically stable way. Apply sigmoid for probability interpretation during inference:
```python
prob = torch.sigmoid(logits)
prediction = (prob > 0.5).float()
```

For multiclass classification, the project discussed the conceptual relationship between softmax, log-softmax, cross-entropy, and NLL loss. Keep the API conventions precise: `CrossEntropyLoss` expects logits and incorporates the log-softmax/NLL computation; `NLLLoss` expects log-probabilities.

### 7.5 Data splits and evaluation
- Training data is used to fit parameters.
- Validation data is used for model/epoch/hyperparameter selection.
- Test data is held back for final evaluation.
- Fit data-dependent preprocessing only on training data.
- Monitor training and validation loss, not accuracy alone.
- A gap between training and validation performance can indicate overfitting.
- Baselines are essential, especially for imbalanced classification.
- Precision, recall, and F1 require careful treatment of zero denominators. In particular, if a model makes no positive predictions and there are actual positives, precision is mathematically undefined (0/0); software libraries may conventionally return zero with a warning or configurable `zero_division` policy. Do not claim that the mathematical F1 is defined merely because a library returns a value.

---

## 8. Neural-network mechanics and architecture discussions

### 8.1 Function composition and the chain rule
A network is a composition:
\[
f(x)=f_n(f_{n-1}(\cdots f_2(f_1(x))\cdots)).
\]
Backpropagation applies the chain rule through this composition. The goal is not to mystify backpropagation: it computes derivatives of the loss with respect to parameters efficiently by reusing intermediate quantities.

### 8.2 Linear layers and parameter counts
For `nn.Linear(in_features, out_features)`, the weight matrix has `out_features × in_features` entries and the bias has `out_features` entries, for a total of:
\[
\text{parameters} = \text{in\_features}\times\text{out\_features}+\text{out\_features}.
\]
Track matrix dimensions carefully; PyTorch stores the weight shape as `(out_features, in_features)`.

### 8.3 Activations
The course has discussed:
- Linear
- Sigmoid
- tanh
- ReLU
- Leaky ReLU
- GELU
- Softmax

Teach these through the trade-offs they address: nonlinearity, differentiability, computational cost, output range, and gradient behaviour. A stack of linear transformations without nonlinearities is still a single linear transformation.

### 8.4 Losses
The taxonomy discussed includes:
- MSE and MAE for regression;
- binary cross-entropy for binary classification;
- categorical cross-entropy for multiclass classification;
- negative log-likelihood;
- hinge loss;
- Dice/IoU-related losses for segmentation (as future examples).

Start from “what should wrongness mean for this task?” rather than “which loss is normally used?”

### 8.5 Optimization and regularization
Topics discussed in the project include:
- gradient descent, SGD, mini-batches, and batch gradient descent;
- momentum as an exponentially weighted moving average of gradients;
- RMSProp as adaptive scaling based on squared gradients;
- Adam combining first- and second-moment estimates;
- AdamW separating weight decay from Adam’s gradient-based update;
- learning-rate schedules, warmup, cosine annealing, and plateaus;
- L1/L2 regularization, weight decay, dropout, early stopping, and augmentation;
- initialization, symmetry breaking, Xavier/Glorot and He initialization;
- exploding and vanishing gradients.

Important nuance: Adam and AdamW are not interchangeable names in PyTorch. `torch.optim.Adam` implements Adam; `torch.optim.AdamW` implements decoupled weight decay. When explaining why, distinguish an L2 penalty added to the objective from multiplicative/decoupled weight decay, especially with adaptive optimizers.

### 8.6 CNNs
Concepts to connect:
- channels and input/output channels;
- kernel size, stride, padding, pooling;
- local receptive fields;
- sparse connectivity;
- weight sharing;
- receptive-field growth through layers;
- feature hierarchies;
- translation equivariance versus invariance (do not casually treat them as identical);
- parameter efficiency relative to fully connected layers.

### 8.7 Residual networks
A previous discussion covered `BasicBlock`, `drop_block`, and skip connections.
- A residual block commonly adds the block input to the transformed branch: \(y=F(x)+x\), possibly using a projection when dimensions differ.
- An identity module such as `nn.Identity()` simply returns its input unchanged; it is not itself the skip connection. The addition in the forward path implements the residual connection.
- ResNet-50 differs from shallower basic-block ResNets by using bottleneck blocks (typically 1×1, 3×3, 1×1 convolutions) to control computation while allowing greater depth.

### 8.8 Transfer learning and fine-tuning
A transfer-learning notebook was reviewed because results were unexpected and training at larger input sizes was extremely slow (32×32 was used as a workaround; one run was estimated to take about five hours at a larger setup). The actual notebook should be inspected before drawing a final diagnosis.

When debugging such experiments, check:
- model architecture and expected input resolution;
- pretrained weights and normalization transforms;
- which layers are frozen/unfrozen;
- optimizer parameter groups;
- label mapping and loss/output compatibility;
- training/evaluation mode;
- train/validation split and leakage;
- whether the validation pipeline matches the pretrained model’s expected preprocessing;
- learning-rate and scheduler settings;
- whether the apparent mismatch is a data issue or an optimization issue.

Do not assume a result is wrong solely because it differs from an expectation; establish the expected baseline and inspect the pipeline.

---

## 9. Other deep-learning experiments and technical topics

These are part of the wider project context; confirm the current status before treating them as active deliverables.

### 9.1 MNIST / FashionMNIST
- MNIST dataset constructor argument `train=True` selects the training split; `train=False` selects the test split.
- FashionMNIST CNN and fully connected models were explored.
- `AdaptiveAvgPool` and parameter counting were discussed.
- Resizing FashionMNIST images to 224×224 and converting grayscale to three channels was used in a transfer-learning context; this can be computationally expensive.
- CPU training at 224×224 was observed to be slow (roughly 3–4 minutes per epoch in one context).

### 9.2 Audio / voice classification
A side project distinguished two family members’ voices:
- audio segmented into 2-second windows with a 1-second stride;
- spectrograms used as CNN inputs;
- source audio at 48 kHz, converted to 16 kHz WAV;
- validation accuracy around 93.96% was reported, with runs fluctuating roughly 89–93%;
- a data-leakage concern was identified because overlapping windows from the same recording can land in both train and validation/test sets;
- sorting files alphabetically and taking the first 80% was not considered a robust group split;
- RMS-based silence detection was proposed to avoid creating spectrograms for silent segments.

Key lesson: split by source recording/person/session before windowing or ensure all windows from a source remain in one partition. Otherwise validation can be unrealistically optimistic.

### 9.3 General implementation/tooling context
- Python, NumPy, pandas, DuckDB, PyTorch, scikit-learn, Jupyter, VS Code, `uv`/virtual environments, and Git are in the working environment.
- CUDA/toolchain compatibility has come up, including an older GTX 1080 (compute capability 6.1) and a Dell Precision laptop with Quadro T1200.
- DataLoader `pin_memory` should generally be enabled when using CUDA if appropriate; check actual bottleneck rather than enabling performance flags blindly.
- `view` versus `permute`, tensor shapes, autograd, and PyTorch API semantics have been discussed.
- The project is technically hands-on: notebooks should be reviewed as executable systems, not only as prose.

---

## 10. Glossary and terminology work

A comprehensive ML vocabulary list was developed. It is intended as a connected reference, not a demand that every term be taught at once.

### General
AI, ML, deep learning, model, architecture, algorithm, training, inference, evaluation.

### Data
Dataset, sample, observation, instance, example, feature, input, target, label, ground truth, batch, mini-batch, epoch, iteration, shuffle.

### Mathematics
Parameter, weight, bias, coefficient, tensor, scalar, vector, matrix, dimension, shape, rank, dot product, matrix multiplication, vectorization, broadcasting.

### Neural networks
Neuron, perceptron, layer, input/hidden/output layer, dense layer, activation, activation function, logit, probability, decision boundary, linear separability, XOR, universal approximation, computational graph, chain rule, Jacobian, Hadamard product.

### Optimization
Objective, loss, cost, gradient, gradient descent, batch/mini-batch/stochastic gradient descent, momentum, RMSProp, Adam, AdamW, learning rate, schedule, annealing, convergence, local/global minimum, saddle point, convex/non-convex, vanishing/exploding gradients.

### Loss and probability
MSE, MAE, cross-entropy, binary cross-entropy, categorical cross-entropy, NLL, hinge loss, likelihood, log-likelihood, maximum-likelihood estimation, likelihood versus probability, log probability.

### Generalization and regularization
Underfitting, overfitting, generalization, memorization, bias-variance tradeoff, L1/L2, weight decay, dropout, early stopping, batch normalization, layer normalization, data augmentation.

### Splits and evaluation
Training, validation, test and hold-out sets; baseline model; majority-class baseline; train/validation/test loss; learning curve; accuracy; precision; recall; F1; confusion matrix; ROC/AUC; decision threshold; calibration; argmax; one-hot label.

### Input modalities and architectures
Tabular data, image, audio, text, sequence, time series, video, graph; FNN/MLP, CNN, RNN, LSTM, GRU, transformer, encoder, decoder, encoder-decoder, autoencoder.

### CNN vocabulary
Channel, kernel size, input/output channels, local receptive field, receptive field, weight sharing, sparse connectivity, stride, feature hierarchy, translation equivariance/invariance.

### Modern AI/LLMs
Token, vocabulary, embedding, embedding vector, context window, attention, self-attention, multi-head attention, positional encoding, prompt, zero-shot, few-shot, RAG, hallucination, temperature, fine-tuning, transfer learning, instruction tuning, RLHF, multimodal models, agents, tool use.

### PyTorch
Tensor, module, parameter, forward, backward, optimizer, Dataset, DataLoader, state dict, checkpoint, device, CPU, GPU, autograd.

### Recommended glossary additions after comparison with Michael Nielsen’s *Neural Networks and Deep Learning*
The companion “Teaching Understanding Questions” file identifies gaps such as:
- perceptron, decision boundary, linear separability, XOR;
- universal approximation theorem;
- computational graph, chain rule, Jacobian, Hadamard product;
- batch versus mini-batch versus stochastic GD;
- learning-rate schedule, annealing, plateaus, exploding/vanishing gradients;
- random initialization, symmetry breaking, Xavier/He initialization;
- local receptive field, weight sharing, sparse connectivity, translation invariance;
- likelihood, log-likelihood, maximum-likelihood estimation;
- one-hot labels, argmax, decision threshold, calibration;
- vectorization, broadcasting, reproducibility, random seeds, checkpoints;
- baselines and learning curves;
- representation learning, feature learning, hierarchical/internal representations;
- end-to-end learning and learned versus handcrafted features.

The suggested “canonical” short list was about 25 terms: Perceptron; Decision Boundary; Linear Separability; XOR; Universal Approximation; Computational Graph; Chain Rule; Jacobian; Hadamard Product; Batch GD; Mini-batch GD; SGD; Learning Rate Schedule; Random Initialization; Xavier Initialization; He Initialization; Symmetry Breaking; Weight Sharing; Sparse Connectivity; Local Receptive Field; Translation Invariance; Representation Learning; Feature Learning; Hierarchical Features; Learning Curve.

Concepts deliberately deferred in the initial pass included RNN/LSTM/GRU, sequence models, language models, encoder/decoder, Seq2Seq, attention, and transformers. Their deferral was a sequencing choice, not a claim that they are unimportant.

---

## 11. Research and historical narratives compiled

### 11.1 Neural-network history
A proposed historical narrative:
1. Perceptron and early artificial-neuron ideas.
2. Minsky and Papert’s analysis of perceptron limitations.
3. The period commonly described as an AI winter.
4. Revival through backpropagation and multilayer networks.
5. LeCun and convolutional networks.
6. GPU acceleration and large-scale datasets.
7. AlexNet as a major modern deep-learning milestone.
8. Attention and transformers.
9. Large language models.

Present history carefully: avoid oversimplifying the AI winters as a single event or attributing progress to one paper/person. The narrative is for motivation, not a replacement for historical detail.

### 11.2 Biological analogy
The course should not overstate the resemblance between artificial neurons and biological cortical neurons. A useful framing is that the historical inspiration came from simplified biological neurons, but modern neural networks are primarily expressive parameterized functions trained by optimization—not faithful simulations of biological neurons.

### 11.3 Nielsen reference
The glossary review compared coverage with Michael Nielsen’s *Neural Networks and Deep Learning* (all six chapters). The companion note estimated that the existing glossary already covered roughly 85–90% of core vocabulary, with a short list of additions expected to make it substantially more complete. Treat those percentages as qualitative estimates, not measured coverage statistics.

---

## 12. Decisions, preferences, and caveats

### Established decisions or clear instructor preferences
- Teach in multiple passes, with narrative and concepts before full derivation/implementation.
- Show the whole system early, then revisit it with increasing precision.
- Teach concepts relationally, not as disconnected definitions.
- Use code to validate ideas, not as a substitute for mathematical understanding.
- Prefer examples that avoid irrelevant setup friction (e.g., Seaborn Titanic for live teaching).
- Keep the training loop stable while changing one major element at a time, so learners can attribute changes.
- Make tensor dimensions and parameter counts concrete.
- Teach preprocessing as part of the model pipeline and explicitly avoid leakage.
- Use baselines and learning curves; accuracy alone is insufficient.
- Demonstrate that a neural network is one model family, not a universal solution.
- Preserve conceptual distinctions (logits vs probabilities; equivariance vs invariance; L2 penalty vs decoupled weight decay; validation vs test).

### Suggestions that were discussed but should not be mistaken for decisions
- The full 16-module roadmap and capstone progression.
- The exact “canonical” glossary selection.
- A precise epoch budget for every teaching notebook.
- Any one particular Titanic preprocessing choice where there are legitimate alternatives.
- Any specific transfer-learning diagnosis before reviewing the current notebook and results.

### Accuracy guardrails
- Do not invent experimental outcomes or imply a notebook has been fixed unless it has been inspected and tested.
- Distinguish mathematical definitions from library conventions (e.g., undefined precision/F1 edge cases versus `zero_division` return values).
- Treat reported accuracy as dataset/split/model-specific.
- Avoid claiming translation invariance where the technically precise property is equivariance.
- Check actual PyTorch loss/optimizer semantics rather than relying on names or informal shorthand.
- Split related windows/groups before augmentation or segmentation to prevent leakage.

---

## 13. Current status and likely continuation points

The latest visible thread in the project export was discussing regularization and how to make the progression conceptually self-discoverable. The sequence had moved from linear regression and autograd through `nn.Module`, classification, train/test splits, overfitting, and regularization. A particularly favoured teaching technique is to let the learner infer a concept—for example, asking at which epoch training should stop and then naming the idea **early stopping**.

Other recent project conversations indicate active or recently revisited work on:
- transfer-learning fine-tuning notebook whose results were unexpected and whose larger input configuration was prohibitively slow;
- loss-function behaviour and baseline losses for untrained models;
- softmax/log-softmax/cross-entropy/NLL conceptual clarity;
- Adam → AdamW and the relationship between momentum, RMSProp, first/second moments, and decoupled weight decay;
- CNN residual blocks, identity/drop-block modules, skip connections, and ResNet-50 bottlenecks;
- imbalance, precision/recall/F1, and undefined edge cases.

These are the most natural continuation points, but verify the latest code/results before assuming a task remains unresolved.

---

## 14. Recommended working conventions for the next AI

When continuing the project:

1. **Establish context from this file first.** Do not restart with a generic curriculum.
2. **Answer the exact question before broadening.** The instructor often asks a narrow technical question to clarify one point, not to request a redesign.
3. **Use derivations when they illuminate the mechanism.** Avoid formula dumping.
4. **Prefer a coherent narrative** when explicitly requested; omit meta-commentary and unnecessary preambles.
5. **Use code that matches the current project’s stack** (Python/PyTorch/pandas/scikit-learn) and make tensor shapes explicit.
6. **When reviewing notebooks, inspect the actual file, code, transforms, metrics, and logs.** Do not diagnose from the filename or one metric alone.
7. **Separate facts, recommendations, and hypotheses.** Label uncertain conclusions.
8. **Maintain the instructor’s multi-pass sequence.** Do not force advanced derivations into the first pass, but do not conceal the deeper structure either.
9. **Use Socratic prompts sparingly and purposefully.** They are useful for letting the learner derive ideas, not as a substitute for answering the instructor.
10. **Keep the narrative evolutionary:** each new concept should answer a limitation or question raised by the previous one.

---
