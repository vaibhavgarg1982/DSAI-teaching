# MLAI Tutorials — Detailed Project Knowledge Base

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

## 15. Source inventory

- `teaching MLAI.md` — extensive project conversation export covering the course, teaching philosophy, notebook development, implementation questions, and conceptual discussions.
- `Teaching Understanding Questions.txt` — companion glossary-gap review against Michael Nielsen’s *Neural Networks and Deep Learning*, with recommended additions and deliberate deferrals.
- Related project conversations available in context include discussions of precision/recall/F1 edge cases, untrained-model loss baselines, tensor `view` versus `permute`, transfer-learning notebook results, `BasicBlock`/residual connections/ResNet-50, AdamW, and CNN training.

This document is a synthesis for continuity. For exact historical wording, code, or experimental state, consult the original source files and the latest notebook/version.

# Technical Appendix — Detailed State, Equations, APIs, and Experimental Findings

This appendix deliberately prioritizes technical continuity over teaching-method discussion. It consolidates concrete equations, shapes, implementation semantics, experiments, debugging observations, and unresolved distinctions from the source conversation and related project discussions. Code snippets are representative; where the original notebook evolved, inspect the latest notebook before treating a snippet as the current implementation.

## A. Regression and gradient descent: the first executable model

### A.1 Linear regression as a parameterized function

The initial optimization work used a model of the form

\[
\hat y = mx+c.
\]

Here \(x\) is an input, \(\hat y\) is the prediction, and \(m,c\) are trainable parameters. With observations \((x_i,y_i)\), mean squared error is

\[
L(m,c)=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
=\frac{1}{N}\sum_{i=1}^{N}(y_i-(mx_i+c))^2.
\]

The derivatives are

\[
\frac{\partial L}{\partial m}
=-\frac{2}{N}\sum_i x_i(y_i-\hat y_i),
\qquad
\frac{\partial L}{\partial c}
=-\frac{2}{N}\sum_i (y_i-\hat y_i).
\]

Gradient descent updates each parameter against its derivative:

\[
m \leftarrow m-\eta\frac{\partial L}{\partial m},
\qquad
c \leftarrow c-\eta\frac{\partial L}{\partial c},
\]

where \(\eta\) is the learning rate. The notebook initially exposed the gradient calculations explicitly, then progressively replaced manual work with PyTorch autograd and an optimizer.

A later synthetic example used a quadratic signal with noise:

```python
X = np.linspace(0, 5, 100)
y = 2 * X + (-5) + X**2 + np.random.randn(X.size)
```

The model was then extended to

\[
\hat y = w_0x+w_1+w_2x^2.
\]

This is still regression with MSE and gradient-based optimization, but the model family is now quadratic rather than linear. The data-generating function is not known to the learner in a real problem; here it is deliberately known so the fit can be inspected.

### A.2 The loss surface is not the data-generating model

A separate visualization considered a non-convex two-parameter function such as

\[
L(m,c)=m^2+c^2+2\sin(3m)\sin(3c).
\]

This function is useful for plotting an optimization landscape, but it is not itself a regression loss computed from input/target pairs: it contains no \(X\), \(y\), or predictions \(\hat y\). That distinction was explicitly noticed during the discussion. Keep the two demonstrations separate:

- **Regression objective:** loss is computed from predictions and targets.
- **Toy optimization landscape:** an arbitrary scalar function is visualized to show local structure and gradient-based movement.

Do not describe the toy function as a data-fitting objective unless it is explicitly redefined that way.

### A.3 The progression from hand-derived gradients to autograd

The progression used was:

1. Calculate the prediction.
2. Calculate MSE.
3. Derive or code the derivatives.
4. Update the parameters manually.
5. Express the parameters as tensors with `requires_grad=True`.
6. Let `loss.backward()` calculate derivatives.
7. Let an optimizer perform parameter updates.
8. Move to `nn.Module` and `nn.Linear` once the basic mechanism is understood.

A minimal manual-parameter pattern is conceptually:

```python
w = torch.tensor(..., requires_grad=True)
b = torch.tensor(..., requires_grad=True)

y_pred = w * X + b
loss = ((y_pred - y) ** 2).mean()

loss.backward()
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
```

Important API semantics:
- `requires_grad=True` asks autograd to track operations that depend on the tensor.
- `loss.backward()` accumulates gradients into leaf tensors' `.grad` fields.
- Gradients accumulate by default; they are not automatically cleared after each backward pass.
- Manual parameter updates should normally occur under `torch.no_grad()`, or via an optimizer.
- `.item()` converts a scalar tensor into a Python scalar for logging; it should not be used in the differentiable computation path.
- Autograd computes derivatives of the operations actually represented in the graph. It does not infer the intended mathematics if the code is wrong.

### A.4 `nn.Module` and the function-composition view

The official PyTorch-style network was introduced as a preview while the learner was still acquiring the terminology. The core abstraction is:

```python
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
```

Mathematically, the network is a composition:

\[
f(x)=f_2(\operatorname{ReLU}(f_1(x))).
\]

`nn.Module` registers submodules and parameters so methods such as `model.parameters()`, `model.train()`, `model.eval()`, `state_dict()`, and device movement can operate on the complete model. `forward` defines the forward computation; normally call `model(x)` rather than calling `model.forward(x)` directly, because the module call path handles hooks and other framework machinery.

## B. Tensor shapes, linear layers, and parameter accounting

### B.1 `nn.Linear` convention

For a batch of \(B\) examples with \(I\) input features:

- Input tensor: \((B,I)\)
- Weight matrix stored by PyTorch: \((O,I)\)
- Bias vector: \((O,)\)
- Output tensor: \((B,O)\)

The operation is

\[
Y=XW^\top+b,
\]

where \(X\in\mathbb R^{B\times I}\), \(W\in\mathbb R^{O\times I}\), \(b\in\mathbb R^O\), and \(Y\in\mathbb R^{B\times O}\). The transpose is important because PyTorch stores weights as `(out_features, in_features)`.

The number of trainable parameters in a fully connected layer with bias is

\[
P=IO+O=O(I+1).
\]

For `nn.Linear(784, 512)`, this is \(784\times512+512=401{,}920\) parameters. The parameter count is independent of batch size.

### B.2 Iris example: `4 → 10 → 5 → 3`

The Iris dataset has 4 input features and 3 target classes. A network with hidden widths 10 and 5 and 3 output logits has:

| Layer | Weight shape | Bias shape | Parameters |
|---|---:|---:|---:|
| `Linear(4, 10)` | `(10, 4)` | `(10,)` | \(4·10+10=50\) |
| `Linear(10, 5)` | `(5, 10)` | `(5,)` | \(10·5+5=55\) |
| `Linear(5, 3)` | `(3, 5)` | `(3,)` | \(5·3+3=18\) |
| **Total** | | | **123** |

For a batch of 32 examples, the shape flow is:

```text
X:          (32, 4)
fc1(X):     (32, 10)
ReLU:       (32, 10)
fc2(...):   (32, 5)
ReLU:       (32, 5)
fc3(...):   (32, 3)   # one logit per class
```

The 150 rows in the full Iris dataset do not change layer parameter counts. They change how many examples are available to estimate the parameters.

### B.3 Titanic MLP parameter count

The compact binary-classification model was:

```python
class TitanicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 4)
        self.fc2 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

Parameter count:
- `Linear(6, 4)`: \(6·4+4=28\)
- `Linear(4, 1)`: \(4·1+1=5\)
- Total: **33 trainable parameters**

A one-layer model `Linear(6, 1)` has 7 trainable parameters and is logistic regression when paired with sigmoid/BCE-style binary classification. Removing hidden layers is a useful way to show that logistic regression is a zero-hidden-layer neural network under this output/loss convention.

The project also included a larger `6 → 16 → 8 → 1` model:
- `Linear(6,16)`: 112 parameters
- `Linear(16,8)`: 136 parameters
- `Linear(8,1)`: 9 parameters
- Total: **257 parameters**

## C. Classification: targets, logits, probabilities, and loss APIs

### C.1 Binary classification

A binary model typically emits one real-valued logit \(z\). The sigmoid maps it to a probability:

\[
p=\sigma(z)=\frac{1}{1+e^{-z}}.
\]

For target \(y\in\{0,1\}\), binary cross-entropy is

\[
L=-\left[y\log p+(1-y)\log(1-p)\right].
\]

Use:

```python
criterion = torch.nn.BCEWithLogitsLoss()
logits = model(X)
loss = criterion(logits, y)
```

`BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable implementation. Therefore:
- The model should return raw logits.
- Do not apply a sigmoid in `forward()` when using this loss.
- Targets should be floating-point values, commonly shaped `(B, 1)` to match logits `(B, 1)`.
- For inference, apply `torch.sigmoid(logits)` when probabilities are required.
- A 0.5 decision threshold is a convention, not a law; it can be changed according to costs, precision/recall requirements, and calibration.

### C.2 Multiclass classification

For \(K\) mutually exclusive classes, the model usually emits \(K\) logits \(z_1,\ldots,z_K\). Softmax converts them into probabilities:

\[
p_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
\]

For a target class \(y\), the negative log-likelihood is

\[
L=-\log p_y.
\]

PyTorch conventions:
- `CrossEntropyLoss` expects raw logits with shape `(B, K)` and class-index targets with shape `(B,)`, typically `torch.long`. It internally performs the stable log-softmax/NLL computation.
- `NLLLoss` expects log-probabilities, typically from `log_softmax`, and class-index targets.
- Do not manually apply softmax before `CrossEntropyLoss`.
- Class indices need not be literally 0, 1, 2 in a conceptual dataset, but PyTorch class-index targets must be integer indices in the valid range `0 ... K-1`; arbitrary strings must first be mapped to indices.
- One-hot targets are not required for the usual `CrossEntropyLoss` path. If using one-hot or soft probability targets, verify the exact loss API and expected target dtype/shape.

Example:

```python
criterion = torch.nn.CrossEntropyLoss()
logits = model(X)                  # (B, 3)
loss = criterion(logits, y.long()) # y: (B,)
predicted_class = logits.argmax(dim=1)
```

`argmax` is used to select the class with the highest logit/probability. Since softmax is monotonic in each logit relative to the shared denominator, the class with maximum logit is also the class with maximum softmax probability.

### C.3 Why not optimize accuracy directly?

Accuracy changes discretely when predictions cross a decision boundary and is not a useful differentiable objective for gradient descent. Cross-entropy changes smoothly with the logits and penalizes confident incorrect predictions more strongly than weakly incorrect ones. It can continue to improve even after accuracy stops changing.

### C.4 Regression losses discussed

| Loss | Formula / defining property | Practical interpretation |
|---|---|---|
| MSE | \(\frac1N\sum_i(y_i-\hat y_i)^2\) | Strongly penalizes large residuals; smooth derivative; sensitive to outliers. |
| MAE | \(\frac1N\sum_i|y_i-\hat y_i|\) | More robust to large residuals; absolute value has a kink at zero. |
| Huber | \(\frac{1}{2}e^2\) for \(\|e\|\le\delta\); \(\delta(\|e\|-\frac{1}{2}\delta)\) otherwise | Quadratic near zero, linear for large errors; balances smoothness and robustness. |
| BCE | \(-[y\log p+(1-y)\log(1-p)]\) | Binary classification when a probability is supplied; `BCEWithLogitsLoss` is usually the numerically stable PyTorch choice. |
| Multiclass cross-entropy | \(-\log p_y\) | Penalizes low probability assigned to the true class. |
| Hinge | Margin-based penalty, e.g. \(\max(0,1-yf(x))\) for labels \(y\in\{-1,+1\}\) | Classic maximum-margin classification objective. |

For Huber loss, \(\delta\) controls the transition:
- Small residuals use a quadratic penalty, like MSE.
- Large residuals use a linear penalty, like MAE.
- Increasing \(\delta\) makes the quadratic region wider.
The exact convention may differ by a factor of \(1/2\) across definitions and libraries; match the formula to the API being used.

### C.5 Baseline loss values for untrained models: interpret, do not memorize blindly

A loss at initialization depends on the task, target distribution, output initialization, reduction convention, and whether outputs are logits or probabilities. Useful reference cases:

- **MSE with targets standardized to mean 0 and variance 1, and a constant zero prediction:** expected MSE is approximately 1, provided the variance convention and finite-sample calculation align.
- **Binary cross-entropy with a zero logit for every example:** sigmoid probability is 0.5, so the per-example loss is \(-\log(0.5)=\log 2\approx0.6931\), independent of class balance.
- **Multiclass cross-entropy with all logits equal:** predicted probability is \(1/K\), so loss is \(\log K\). For three classes, \(\log 3\approx1.0986\).
- **MAE with standardized targets:** no universal value of 1; it depends on the target distribution.
- Randomly initialized models do not necessarily emit exactly zero logits, so observed initial loss may be near—but not exactly equal to—these references.

These are sanity checks under stated assumptions, not universal “default losses.”

## D. Classification metrics and edge cases

### D.1 Confusion matrix

For binary classification:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

\[
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}.
\]

Accuracy is \((TP+TN)/(TP+TN+FP+FN)\).

Precision answers: among predicted positives, how many are truly positive? Recall answers: among actual positives, how many did the model find?

### D.2 Undefined metric cases

If there are actual positives but the model predicts no positives, then \(TP=0\), \(FP=0\), \(FN>0\):
- Precision is \(0/0\), mathematically undefined.
- Recall is \(0/(0+FN)=0\).
- F1, in its formula involving precision, is not defined when precision is undefined; equivalently, many libraries use the count-based formula and apply a convention for a zero denominator.
- Software may return 0 and emit a warning, or use a configurable `zero_division` policy. That is a software convention, not a proof that the mathematical ratio is defined.

If the dataset contains no actual positives, recall itself has a zero denominator. Any reported value needs its convention stated.

For a dataset with 99% positive and 1% negative labels, an always-positive classifier has 99% accuracy but zero specificity and poor utility if negatives matter. Which class is called “positive” must be explicit; the user's initial phrasing about imbalance was later corrected as a dictation slip.

### D.3 Evaluation protocol

- Training set: estimates model parameters.
- Validation set: selects epochs, hyperparameters, thresholds, and model variants.
- Test set: final unbiased estimate after choices are complete.
- `stratify=y` in a random split preserves class proportions approximately across the split, especially useful with small or imbalanced datasets. It does not fix distribution shift or leakage.
- Use `random_state` for reproducible splits; this does not guarantee full reproducibility across all hardware, libraries, kernels, or nondeterministic GPU operations.
- Always compare against a meaningful baseline. For Titanic, an always-predict-majority-class classifier is a useful minimum benchmark.

## E. Data processing and leakage: Titanic notebook details

### E.1 Dataset and selected columns

The notebook used:

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

The compact feature set discussed was `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, with `survived` as the target. This gives six input features after encoding `sex` as one binary column.

Relevant meanings:
- `survived`: target, 0 = no, 1 = yes.
- `pclass`: passenger ticket class, 1/2/3; ordinal, though often stored numerically.
- `age`: age in years; has missing values.
- `sibsp`: number of siblings/spouses aboard.
- `parch`: number of parents/children aboard.
- `fare`: fare paid.
- `sex`: categorical variable, usually represented as a binary indicator for this compact example.

A representative preparation snippet:

```python
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()
df['age'] = df['age'].fillna(df['age'].median())
df = pd.get_dummies(df, columns=['sex'], drop_first=True)
```

`drop_first=True` drops one dummy column from a one-hot encoding. With two categories, one indicator can encode both states (e.g. 1 = female and 0 = male, depending on category ordering). With an intercept, retaining every dummy creates exact linear dependence because the columns sum to one. Dropping a reference category avoids this dummy-variable redundancy in linear models. It is not mandatory for every model, but it is a sensible compact representation here.

`pclass` is ordinal rather than nominal: numeric encoding imposes a linear/ordered interpretation. It can be retained as numeric for a simple baseline, but the assumption should be known. One-hot encoding would express category-specific effects without assuming equal numeric spacing.

### E.2 Split before fitting data-dependent preprocessing

The corrected order is:

```text
Load / clean / encode
        ↓
Separate target from features
        ↓
Train / test split (optionally stratified)
        ↓
Fit imputer/scaler on training features only
        ↓
Transform train and test with training-fitted objects
        ↓
Convert to tensors
        ↓
Train and evaluate
```

Fitting `StandardScaler` on the full dataset leaks information from the test distribution into the preprocessing pipeline. For continuous columns, keep a DataFrame until the split and scaling are complete:

```python
continuous = ['age', 'fare']
scaler = StandardScaler()

train_df[continuous] = scaler.fit_transform(train_df[continuous])
test_df[continuous] = scaler.transform(test_df[continuous])
```

Then form `X` and `y` from each DataFrame and convert them to tensors. The same principle applies to imputation, feature selection, PCA, and any other transformation that learns statistics from data.

The original conversation deliberately debated whether to keep DataFrames longer because column names make it easier to scale only continuous features. This is technically useful, not merely a stylistic preference.

### E.3 Tensor conversion and shape contract

For the binary task:

```python
X = torch.tensor(feature_values, dtype=torch.float32)  # (N, 6)
y = torch.tensor(target_values, dtype=torch.float32).view(-1, 1)  # (N, 1)
```

A frequent error occurred when `X` was created from a DataFrame containing an unencoded string column; NumPy then produced `dtype=object`, which cannot be converted directly to a float tensor. Encode categorical data and verify that all feature columns are numeric before tensor conversion.

For a binary `Linear(..., 1)` output and `BCEWithLogitsLoss`, use target shape `(N,1)` to match logits. For multiclass `CrossEntropyLoss`, the usual target shape is `(N,)` with integer class indices.

### E.4 Why standardize features?

Standardization uses training-set statistics:

\[
x'=\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}.
\]

It can improve optimization when continuous features have very different scales. It is not required for every model or every feature. In the teaching version, scaling `age` and `fare` while leaving binary flags and small integer counts alone was considered easier to interpret. For a production pipeline, use a principled column transformer and validate the choice rather than assuming one scaling rule fits every feature.

### E.5 Reported Titanic outcomes and caveats

Reported values across evolving notebook versions included:
- a compact model result around **87.77%** accuracy in one run;
- an initial larger model discussed with training accuracy around 86% and test accuracy around 80%;
- a regularized model with training accuracy around 85%, test accuracy around 81%, and an example confusion matrix `[[403, 36], [72, 201]]` in a particular evaluation context.

These values should **not** be merged into one canonical benchmark: different splits, training/evaluation sets, preprocessing, model sizes, and stages of notebook debugging were involved. The confusion-matrix counts sum to 712, so they describe a particular evaluated set, not automatically the full dataset. Always recover the notebook and split before quoting a result.

A particularly important debugging catch was that the test set had not been scaled with the scaler fitted on training data. This creates a train/test preprocessing mismatch and can materially damage performance. The correct rule is `fit_transform` on train and `transform` on test.

## F. PyTorch training-loop semantics and common debugging traps

### F.1 Canonical full-batch loop

```python
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(Xtrain)
    loss = criterion(logits, ytrain)
    loss.backward()
    optimizer.step()
```

This is full-batch gradient descent if the entire training tensor is used each step. It is not mini-batch SGD unless the data is iterated in batches.

### F.2 What each operation does

- `model.train()` sets training mode for modules such as Dropout and BatchNorm.
- `optimizer.zero_grad()` clears gradients accumulated from previous backward passes.
- `model(X)` executes the module's forward path.
- `criterion(prediction, target)` computes the scalar objective.
- `loss.backward()` calculates gradients.
- `optimizer.step()` updates parameters.
- `model.eval()` switches modules such as Dropout and BatchNorm to evaluation behaviour.
- `torch.no_grad()` disables gradient recording in evaluation code; it does not itself switch Dropout/BatchNorm to evaluation mode.

For a network made only of `Linear` and ReLU layers, `train()` and `eval()` may produce identical predictions. The distinction becomes material once Dropout or BatchNorm is added.

### F.3 Measuring losses and model mode

A discussed loop computed training and test losses before the optimizer update. That is valid, but the losses represent the model before that epoch's update. If logging a training loss from the same forward pass used for backpropagation, `.item()` can be saved before `optimizer.step()`. Test evaluation should be performed without gradients and in evaluation mode:

```python
model.eval()
with torch.no_grad():
    test_logits = model(Xtest)
    test_loss = criterion(test_logits, ytest).item()
model.train()
```

Do not accidentally calculate test loss with training-mode Dropout/BatchNorm. Do not tune repeatedly against the test set and then report it as an untouched final evaluation; use a validation set for selection.

### F.4 Accuracy from logits

Binary case:

```python
with torch.no_grad():
    logits = model(X)
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    accuracy = (predictions == y).float().mean().item()
```

Multiclass case:

```python
with torch.no_grad():
    logits = model(X)                  # (N, K)
    predictions = logits.argmax(dim=1) # (N,)
    accuracy = (predictions == y).float().mean().item()
```

Use `model.eval()` as well when appropriate. For multiclass `CrossEntropyLoss`, argmax of logits is sufficient; a softmax is not required just to choose the most likely class.

### F.5 Train loss versus validation loss

A healthy training loss can keep falling while validation loss stops improving or begins rising. This is the signature that motivated early stopping and regularization in the Titanic notebook. A common procedure is:
- monitor validation loss;
- save the checkpoint when validation loss improves;
- stop after a chosen patience window with no improvement;
- restore the best checkpoint rather than automatically using the final epoch.

The exact patience and epoch budget were not established as universal project constants.

## G. Optimizers: equations and conceptual lineage

Let \(\theta_t\) be parameters and \(g_t=\nabla_\theta L_t\) the gradient at step \(t\). Learning rate is \(\eta\).

### G.1 Plain SGD

\[
\theta_{t+1}=\theta_t-\eta g_t.
\]

One global learning rate scales the gradient. Batch Gradient Descent uses the full dataset per update; Stochastic Gradient Descent in strict terminology uses one example per update; mini-batch SGD uses a subset. In everyday ML libraries, “SGD” often means mini-batch SGD.

### G.2 Momentum

One common velocity convention is:

\[
v_t=\beta v_{t-1}+g_t,\qquad
\theta_{t+1}=\theta_t-\eta v_t.
\]

Another equivalent convention absorbs the learning rate into the velocity and writes the update with a plus sign. Keep the convention consistent when comparing formulas.

Intuition: an exponentially weighted history smooths the direction of movement. Consistent gradient components accumulate; rapidly changing components are damped. This can accelerate progress along shallow directions and reduce oscillation across steep directions. Momentum smooths the **direction/history of the gradient**, not the parameter values themselves.

### G.3 AdaGrad

\[
G_t=G_{t-1}+g_t\odot g_t,\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{G_t}+\epsilon}.
\]

The accumulator grows over time, so coordinates with large accumulated gradients receive smaller effective steps. It can work well for sparse features, but its effective learning rate may become too small because the accumulator never forgets old gradients.

### G.4 RMSProp

\[
s_t=\rho s_{t-1}+(1-\rho)(g_t\odot g_t),\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}.
\]

RMSProp replaces AdaGrad's indefinitely growing sum with an exponentially weighted average of squared gradients. It adapts step sizes by coordinate while forgetting older gradient magnitudes.

### G.5 Adam

Adam combines first- and second-moment estimates:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]
\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

Because these moving averages start at zero, Adam uses bias-corrected estimates:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

Then

\[
\theta_{t+1}=\theta_t-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

Interpretation:
- The first moment is an EWMA of signed gradients; it smooths direction.
- The second moment is an EWMA of squared gradients; it estimates scale and produces coordinate-wise normalization.
- Dividing by the square root of the second moment adapts the effective update magnitude.
- These terms are not simply “direction and magnitude” in a literal geometric decomposition, but that is a useful first intuition.

### G.6 AdamW and decoupled weight decay

AdamW separates weight decay from the adaptive gradient update. In a simplified form:

\[
\theta_{t+1}=\theta_t-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_t.
\]

The final term shrinks parameters multiplicatively. Exact implementation details depend on options such as decoupled decay and parameter groups, but the key distinction is:

- **L2 penalty in the objective:** \(J(\theta)=L(\theta)+\lambda\|\theta\|_2^2\), whose gradient adds a term proportional to \(\theta\).
- **Decoupled weight decay:** a separate shrinkage term, not passed through Adam's adaptive preconditioner.

For plain SGD, L2 regularization and weight decay can be expressed as equivalent updates under matching coefficient conventions. With adaptive optimizers such as Adam, coupling the penalty into the gradient and decoupling the decay generally produce different updates. In PyTorch, `torch.optim.Adam` and `torch.optim.AdamW` are distinct optimizers; do not call them interchangeable.

### G.7 Optimizer choice is empirical, not a theorem

The project discussed these broad starting points:
- SGD + momentum: useful for many CNN training setups and sometimes strong generalization.
- Adam: robust starting point for many smaller models and experiments.
- AdamW: common for transformers and modern architectures where decoupled decay is desired.

These are heuristics, not rules. Compare optimizers under controlled conditions and include learning rate, schedule, weight decay, data order, and training budget in the comparison.

## H. Regularization: L2, weight decay, dropout, early stopping

### H.1 L2 penalty derivation

If

\[
J(\theta)=L(\theta)+\lambda\sum_i\theta_i^2,
\]

then

\[
\nabla_\theta J=\nabla_\theta L+2\lambda\theta.
\]

A plain gradient-descent update is

\[
\theta \leftarrow \theta-\eta(\nabla_\theta L+2\lambda\theta)
=(1-2\eta\lambda)\theta-\eta\nabla_\theta L.
\]

The first term shrinks the parameter vector, which explains the phrase “weight decay.” Some libraries define the coefficient with a factor of \(1/2\), so the \(2\lambda\) factor depends on the convention. In the common PyTorch optimizer API, the exact relationship between `weight_decay` and an explicit L2 loss term should be understood in the context of the optimizer.

### H.2 What regularization is trying to control

Regularization discourages solutions that fit training data at the cost of poor generalization. It does not guarantee that every individual weight is small or that smaller weights always mean a better model. Parameter scale also depends on feature scaling, architecture, symmetries, and activation choices.

The Titanic experiment compared no decay with `weight_decay=3e-3` for an Adam optimizer. The reported curves suggested:
- without regularization, training loss continued to fall while validation/test loss eventually rose;
- with decay, training loss fell more slowly and the held-out loss curve was flatter;
- accuracy changed only modestly, while loss curves made the generalization difference clearer.

These were observations from a particular run, not guaranteed effects of that exact coefficient.

### H.3 L1 versus L2

- L1 penalty: \(\lambda\sum_i|\theta_i|\). It can encourage exact zeros and sparse parameter vectors; it is not differentiable at zero in the ordinary sense, though subgradients are defined and optimizers handle it.
- L2 penalty: \(\lambda\sum_i\theta_i^2\). It smoothly penalizes large magnitudes and typically shrinks weights rather than making them exactly zero.
- Neither is a substitute for sound validation, appropriate features, or a leakage-free pipeline.

### H.4 Dropout and early stopping

- Dropout randomly masks activations during training, reducing reliance on particular units/co-adaptations. In evaluation mode it is normally disabled and activations are appropriately scaled by the module's implementation.
- Early stopping chooses a checkpoint based on held-out validation performance, usually stopping after no improvement for a patience interval.
- Data augmentation changes the training examples in label-preserving ways and can improve robustness, but augmentation must not cross dataset boundaries in a way that leaks near-duplicates into validation/test.

## I. Activations, output heads, and initialization

### I.1 Activation functions in the project glossary

| Function | Definition | Key properties / common use |
|---|---|---|
| Identity / linear | \(f(x)=x\) | Common output for regression and logits. |
| Sigmoid | \(1/(1+e^{-x})\) | Range \((0,1)\); binary probability interpretation; saturates for large \(|x|\). |
| Tanh | \(\tanh x\) | Range \((-1,1)\); zero-centred but saturates. |
| ReLU | \(\max(0,x)\) | Cheap, sparse positive activations; derivative zero for negative inputs. |
| Leaky ReLU | \(\max(\alpha x,x)\) | Keeps a small negative-side slope. |
| GELU | \(x\Phi(x)\) (exact form) | Smooth activation used in many transformer architectures; implementations may use approximations. |
| Softmax | \(e^{z_i}/\sum_j e^{z_j}\) | Normalizes a vector of logits into class probabilities; typically paired implicitly with cross-entropy in PyTorch. |

A linear layer followed by another linear layer with no nonlinear activation can be collapsed into one linear/affine transformation. Nonlinear activations make deeper compositions more expressive.

### I.2 Initialization and symmetry breaking

If every hidden neuron starts with identical weights and biases, neurons receive identical gradients and remain symmetric, so they fail to learn distinct features. Random initialization breaks this symmetry. It should not be so large that activations/gradients saturate or explode, nor so small that signals vanish.

- Xavier/Glorot initialization is commonly paired with approximately linear/tanh-like activations and seeks to preserve activation/gradient variance across layers.
- He/Kaiming initialization is designed for ReLU-family activations and accounts for the fraction of activations set to zero.
- Initialization is separate from the optimizer: initialization sets starting parameters; the optimizer changes them during training.

## J. CNNs and spatial inductive bias

### J.1 Convolution dimensions

For a 2D convolution, output height is

\[
H_{\text{out}}=
\left\lfloor\frac{H_{\text{in}}+2P-D(K-1)-1}{S}+1\right\rfloor,
\]

and similarly for width. Here \(K\) is kernel size, \(S\) stride, \(P\) padding, and \(D\) dilation. For ordinary convolution with \(D=1\), this reduces to the familiar kernel/stride/padding formula.

A convolution layer has weight shape approximately

```text
(out_channels, in_channels / groups, kernel_height, kernel_width)
```

and parameter count

\[
C_{\text{out}}\left(\frac{C_{\text{in}}}{g}K_hK_w\right)+C_{\text{out}}
\]

when bias is enabled and the group count is \(g\).

### J.2 Why convolutions are parameter-efficient

A fully connected layer treats each input position as a separate connection. Convolution uses:
- **local receptive fields:** each filter sees a small spatial neighborhood;
- **weight sharing:** the same filter is applied at many spatial locations;
- **sparse connectivity:** each output position connects only to a local subset of input positions;
- **hierarchical feature composition:** later layers combine simpler local features into larger patterns.

Convolution is translation-equivariant under idealized conditions: shifting the input shifts the feature map correspondingly. This is not the same as translation invariance. Pooling, global aggregation, augmentation, and architecture design may increase robustness to shifts, but exact invariance should not be asserted casually.

### J.3 Stride, padding, dilation, pooling

- Stride greater than one reduces spatial resolution and computation.
- Padding controls border handling and can preserve spatial size.
- Dilation spaces kernel samples apart, expanding the receptive field without proportionally increasing parameter count.
- Pooling aggregates local responses and reduces resolution; modern architectures sometimes replace fixed pooling with strided convolution or global average pooling.
- Receptive field grows through successive layers, allowing deeper features to depend on larger portions of the image.

### J.4 Depthwise-separable and grouped convolution

- Grouped convolution splits channels into independent groups; each group learns its own filters.
- Depthwise convolution is the special case in which each input channel is convolved separately.
- Depthwise-separable convolution typically follows depthwise convolution with a 1×1 pointwise convolution that mixes channels. It can greatly reduce computation compared with a dense convolution, at potential accuracy/architecture trade-offs.
- These are modern glossary additions, not confirmed as fully taught modules in the source export.

## K. Residual networks: identity, skip path, and bottlenecks

### K.1 Residual block

The central form is

\[
y=F(x)+x.
\]

The residual branch \(F(x)\) learns a correction to the input rather than having to represent the entire desired mapping from scratch. If shapes differ, a projection \(W_s x\), often implemented with a 1×1 convolution (and possibly stride), aligns dimensions:

\[
y=F(x)+W_sx.
\]

The direct path gives gradients a shorter route through deep networks and makes it easier for a block to approximate the identity mapping.

### K.2 `nn.Identity` versus a skip connection

`nn.Identity()` is an identity module: it returns its input unchanged. It is often used as a placeholder when a projection is unnecessary. It does **not** itself add the residual branch to the input. The skip connection is the dataflow and addition in the block's `forward`, e.g.:

```python
identity = x
out = self.conv1(x)
out = self.relu(out)
out = self.conv2(out)
out += identity
out = self.relu(out)
```

A member named `drop_block` should not be assumed to implement a skip connection. If its `forward` path calls it and the configured module is `nn.Identity`, that call is a no-op. Determine behaviour from the actual instantiated module and dataflow, not from the variable name.

### K.3 ResNet-18-style basic block versus ResNet-50 bottleneck

A typical basic block uses two 3×3 convolutions in its residual branch. A bottleneck block typically uses:

```text
1×1 convolution: reduce channels
3×3 convolution: spatial processing
1×1 convolution: expand channels
```

The 1×1 convolutions control channel width and computational cost, allowing ResNet-50 to be deeper without making every 3×3 convolution operate on the full expanded channel count. Common bottleneck expansion factor is 4, though implementation details should be checked in the specific model.

## L. Transfer learning and fine-tuning: debugging checklist

A transfer-learning notebook `0600_transfer_learning_ft.ipynb` was reviewed after results were unexpected. One configuration using 32×32 inputs was chosen because the 224×224 setup was taking an estimated five hours. That performance observation is specific to the hardware, batch size, model, and pipeline at the time; do not assume it generalizes.

When continuing the review, establish each of the following from the actual notebook:

1. **Pretrained weights:** which checkpoint and which pretraining dataset?
2. **Input contract:** required image size, channel count, normalization mean/std, interpolation, and crop policy.
3. **Data transforms:** are train and validation transforms appropriate and consistent with the model's pretraining assumptions?
4. **Output head:** correct number of classes and correct output/loss pairing?
5. **Label mapping:** does the class-to-index mapping match training and evaluation?
6. **Frozen layers:** which parameters have `requires_grad=False`? Are intended layers actually trainable?
7. **Optimizer parameter groups:** does the optimizer include every trainable parameter, and exclude frozen ones as intended?
8. **Learning rates:** is the head trained at a suitable rate? Is the backbone learning rate lower during fine-tuning?
9. **Training/evaluation modes:** is the model in `train()` for training and `eval()` for validation? What happens to BatchNorm statistics in a frozen backbone?
10. **Data splits:** are samples, subjects, or source files separated before crops/windows/augmentation?
11. **Metrics:** is validation computed over the entire validation set, with correct aggregation?
12. **Compute budget:** does the image size make sense for the task and available hardware? Resizing 28×28 FashionMNIST to 224×224 does not create new image information but increases compute substantially.

Do not diagnose the result from the score alone. Reproduce a simple baseline and inspect the pipeline before changing multiple variables.

## M. MNIST, FashionMNIST, and image preprocessing

- In torchvision datasets such as MNIST, `train=True` selects the training split; `train=False` selects the test split.
- MNIST and FashionMNIST are grayscale 28×28 images in their original form.
- Converting grayscale to three channels by replication allows use of an RGB-pretrained backbone, but does not add new information.
- Resizing to 224×224 may be required by some pretrained architectures, but multiplies spatial positions dramatically. A 28×28 image has 784 pixels; a 224×224 image has 50,176 pixels, 64 times as many spatial positions.
- A noted CPU run on 224×224 inputs took roughly 3–4 minutes per epoch in one setup.
- Adaptive pooling lets a model produce a fixed spatial output size from varying input sizes, depending on where and how it is used. It does not automatically make an entire architecture valid for every input resolution; inspect downstream layer assumptions.
- Parameter count, FLOPs, input resolution, batch size, data-loading overhead, and hardware all affect runtime. Do not attribute a long epoch solely to model parameter count.

## N. Audio voice classifier: pipeline and leakage risk

The side project classified audio as one of two speakers:
- source recordings were M4A at 48 kHz;
- audio was converted to WAV at 16 kHz;
- audio was segmented into 2-second windows with a 1-second stride;
- spectrograms were generated and fed to a CNN;
- validation accuracy around 93.96% was reported in one run, with observed results varying roughly from 89% to 93%.

### N.1 The major experimental risk: correlated windows

Adjacent windows overlap by 50%. A window covering seconds 1–3 and another covering seconds 2–4 share one second of audio. Randomly splitting those windows across train and validation lets highly correlated audio appear on both sides, creating serious leakage and an overly optimistic validation score.

A safer procedure is:
1. split at the source-recording or recording-session level first;
2. segment each partition independently;
3. derive spectrograms within the relevant partition;
4. tune only on the validation recordings;
5. reserve a final test set from different source recordings/sessions.

Alphabetical sorting followed by an 80/20 split is not automatically safe: filenames may encode speaker, recording session, or collection order. Confirm that the split separates independent source material rather than merely separating neighboring filenames.

### N.2 Silence filtering

RMS energy over a window can flag very quiet segments:

\[
\operatorname{RMS}(x)=\sqrt{\frac1N\sum_{n=1}^N x_n^2}.
\]

A threshold can be used to skip near-silent windows before spectrogram generation. The threshold should be validated against actual recording levels: too high can discard quiet speech, and too low may retain noise. This is preprocessing, not a substitute for robust source-level splits.

### N.3 Resampling and spectrogram consistency

Converting 48 kHz audio to 16 kHz is resampling, not merely changing metadata. Proper resampling includes anti-alias filtering. If validation changes across conversion paths, check:
- that all recordings use the same resampler and target rate;
- that the waveform is genuinely resampled, not just relabelled;
- amplitude scaling and channel handling;
- spectrogram parameters (window, hop length, FFT size, power/log transform);
- segment boundaries and padding;
- train/validation normalization consistency.

A model trained and evaluated on spectrograms produced by different preprocessing pipelines may experience a distribution shift.

## O. Embeddings, representation learning, PCA, and autoencoders

The broader project has included PCA and autoencoder latent spaces. Keep these ideas distinct:

- **Feature engineering:** humans choose or construct useful input features.
- **Representation learning:** the model learns an internal representation from data to support the task.
- **Embedding:** a vector representation of a discrete or structured item, learned so that useful relationships can be represented geometrically.
- **Latent representation:** an internal compressed or transformed code, such as an autoencoder bottleneck.
- **PCA:** a linear, unsupervised dimensionality-reduction method that finds orthogonal directions of maximum variance; it is not a neural network and does not in general preserve task-relevant information.
- **Autoencoder:** an encoder maps input to a latent code; a decoder reconstructs the input. Reconstruction loss trains the representation, but good reconstruction does not guarantee that the latent space is optimal for classification or another downstream task.

A useful comparison is that PCA is a linear projection with a closed-form/eigendecomposition-based solution (or equivalent numerical methods), whereas an autoencoder learns a potentially nonlinear encoder/decoder by optimization. Their objectives and representations should not be treated as interchangeable.

## P. Glossary gaps and modern deep-learning vocabulary

The companion file “Teaching Understanding Questions.txt” compared the existing glossary with Michael Nielsen's *Neural Networks and Deep Learning*. It estimated roughly 85–90% coverage of core conceptual vocabulary, but that estimate was qualitative. The most useful additions for a technically durable glossary are:

### P.1 Foundations and backpropagation
- Perceptron
- Decision boundary
- Linear separability
- XOR problem
- Universal approximation theorem
- Computational graph
- Chain rule
- Jacobian
- Hadamard (element-wise) product
- Vectorization
- Broadcasting

### P.2 Optimization and training
- Batch gradient descent
- Mini-batch gradient descent
- Stochastic gradient descent
- Learning-rate schedule
- Annealing
- Plateau
- Vanishing gradients
- Exploding gradients
- Random initialization
- Symmetry breaking
- Xavier/Glorot initialization
- He/Kaiming initialization
- Random seed
- Checkpoint / best checkpoint
- Reproducibility

### P.3 Convolution and representations
- Channel / input channel / output channel
- Kernel size
- Stride / padding / dilation
- Local receptive field / receptive field
- Weight sharing
- Sparse connectivity
- Translation equivariance versus invariance
- Feature hierarchy
- Feature learning
- Representation learning
- Internal representation
- End-to-end learning
- Handcrafted versus learned features

### P.4 Probability, classification, and evaluation
- Likelihood
- Log-likelihood
- Maximum likelihood estimation
- Log probability
- One-hot label
- Argmax
- Decision threshold
- Calibration
- Baseline model
- Majority-class baseline
- Train loss / validation loss / test loss
- Learning curve

### P.5 Modern architecture and training terms
- Residual/skip connection
- Bottleneck block
- Depthwise-separable convolution
- Grouped convolution
- Layer normalization
- GELU
- AdamW
- Learning-rate warmup
- Cosine annealing
- Mixed precision
- Transfer learning
- Fine-tuning
- Embeddings
- Positional encoding
- Self-attention

RNN, LSTM, GRU, sequence models, language models, encoder/decoder, seq2seq, attention, and transformers were listed as concepts deliberately deferred in the first pass, not as permanently excluded topics.

## Q. Technical open items and unresolved state

These should be treated as continuation checkpoints, not silently declared complete:

1. **Transfer-learning fine-tuning notebook:** results were unexpected and the 224×224 configuration was expensive. Reinspect `0600_transfer_learning_ft.ipynb` and the actual output/logs before making changes.
2. **Untrained-model loss table:** use the conditional reference values in Section C.5; do not treat a single number as universal without specifying targets, logits, initialization, and reduction.
3. **`view` versus `permute`:** a separate discussion asked for the distinction. Core rule: `view` reshapes a tensor without reordering its logical elements and generally requires compatible strides/contiguity; `permute` reorders dimensions and usually returns a view with changed strides. `reshape` may return a view or make a copy. For image tensors, NCHW and NHWC are different dimension orders, not merely different shape labels.
4. **Softmax / log-softmax / cross-entropy / NLL:** preserve the API contract in Section C.2. In particular, `CrossEntropyLoss` expects logits; `NLLLoss` expects log-probabilities.
5. **Adam versus AdamW:** preserve the coupled-L2 versus decoupled-decay distinction in Section G.6.
6. **ResNet block inspection:** establish whether a module is `Identity`, a projection, DropBlock, or another module from the actual initialization and `forward` path. Names alone are insufficient.
7. **Metric edge cases:** if a confusion-matrix denominator is zero, state the mathematical undefinedness and separately report the library's convention.
8. **Audio evaluation:** source-level split before overlapping window generation is essential to avoid leakage.
9. **Titanic notebook versioning:** reported scores span different iterations; inspect the current code before citing a “best” result.
10. **CNN input sizes and transfer learning:** resizing to a pretrained network's preferred dimensions has a computational cost and does not increase source information.

## R. Source-of-truth policy for future continuation

Use this hierarchy:
1. Current notebook/code and its outputs for implementation state.
2. The original `teaching MLAI.md` export for what was actually discussed and which values appeared in a particular run.
3. This knowledge base for navigation, continuity, and consolidated equations.
4. General ML knowledge only to explain or verify a point, clearly distinguishing it from a decision or result documented in the source.

Do not merge numbers from different runs, treat suggestions as decisions, or infer that a proposed curriculum module has been completed. The knowledge base is intentionally more detailed than the original overview, but it remains a synthesis rather than a replacement for executable notebooks and raw conversation history.

# Technical Appendix II — Expanded Working Reference

This second appendix goes deeper than the first. It is designed for an AI or human continuing the actual technical work: reconstructing derivations, understanding why the code is shaped as it is, interpreting experimental evidence, and knowing what to inspect before changing an implementation.

**Evidence labels used here**
- **Recorded:** directly reflected in the project export, notebook excerpts, or documented experimental discussion.
- **Derived:** a mathematical consequence of the recorded setup.
- **General reference:** additional standard ML/PyTorch context included to make the notes usable. It should not be mistaken for a decision made in the project.

The conversation export includes more than one notebook iteration and some broader course-planning discussion. Reported numbers below are always tied to their described context where possible; they are not a single benchmark suite.

## T. The common mathematical skeleton of supervised learning

### T.1 Parameterized function

A supervised model can be represented as

\[
\hat y=f(x;\theta),
\]

where:
- \(x\) is an input example or feature vector;
- \(y\) is the observed target;
- \(\hat y\) is the model's prediction;
- \(\theta\) denotes all trainable parameters;
- \(f\) is the model architecture plus its forward computation.

For a dataset \(\mathcal D=\{(x_i,y_i)\}_{i=1}^N\), empirical risk minimization chooses parameters to minimize an average loss:

\[
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f(x_i;\theta),y_i).
\]

The familiar training loop is an implementation of this objective:

1. Compute \(f(x_i;\theta)\) — forward pass.
2. Compare predictions with targets using \(\ell\) — loss.
3. Differentiate the loss with respect to \(\theta\).
4. Update \(\theta\) with an optimization algorithm.
5. Repeat, then measure performance on data not used to fit \(\theta\).

This skeleton is shared by linear regression, logistic regression, multilayer perceptrons, CNNs, and transformers. The architecture changes the function family; the loss changes what counts as a good prediction; the optimizer changes how parameters are updated.

### T.2 Error, loss, cost, objective, and criterion

These terms are often used inconsistently, so record the convention in the notebook rather than assuming universal terminology.

- **Residual/error:** a per-example discrepancy such as \(e_i=y_i-\hat y_i\). In classification, “error” may mean an incorrect predicted class rather than a signed numerical residual.
- **Loss:** a scalar penalty for a prediction, often per example before reduction.
- **Cost:** often the aggregate loss over a dataset or batch.
- **Objective:** the complete quantity optimized; it may include data loss plus regularization or other terms.
- **Criterion:** common PyTorch code name for the loss module, e.g. `criterion = nn.MSELoss()`.

A regularized objective might be

\[
J(\theta)=\frac1N\sum_i\ell(f(x_i;\theta),y_i)+\lambda R(\theta).
\]

The regularizer \(R(\theta)\) is part of the objective even though it does not directly measure prediction error on an example.

### T.3 Empirical risk is not population risk

The training objective is calculated from a finite sample. The quantity of real interest is performance on the data-generating distribution, which is unknown. A model can minimize empirical loss while generalizing poorly if it exploits sample-specific patterns. This is why training loss alone cannot establish model quality, and why a held-out split must be independent of model selection to provide a credible final estimate.

## U. Deriving linear regression and its optimization landscape

### U.1 Vectorized linear regression

For \(N\) observations and \(d\) features, augment the feature matrix with a column of ones to absorb the intercept:

\[
\tilde X=
\begin{bmatrix}
x_1^\top & 1\\
\vdots & \vdots\\
x_N^\top & 1
\end{bmatrix},\qquad
\theta=
\begin{bmatrix}
w\\b
\end{bmatrix}.
\]

Then predictions are

\[
\hat y=\tilde X\theta.
\]

With residual vector \(r=\tilde X\theta-y\), the half-scaled least-squares objective is

\[
J(\theta)=\frac{1}{2N}\|\tilde X\theta-y\|_2^2.
\]

Its gradient is

\[
\nabla_\theta J=\frac1N\tilde X^\top(\tilde X\theta-y).
\]

The Hessian is

\[
\nabla^2_\theta J=\frac1N\tilde X^\top\tilde X.
\]

Since \(\tilde X^\top\tilde X\) is positive semidefinite, this objective is convex. If the design matrix has full column rank, the Hessian is positive definite and the minimizer is unique. If columns are linearly dependent, there may be multiple parameter vectors with identical predictions and loss.

The closed-form normal-equation solution, when the inverse exists, is

\[
\theta^*=(\tilde X^\top\tilde X)^{-1}\tilde X^\top y.
\]

In practice, one generally solves least squares using stable linear algebra rather than explicitly computing a matrix inverse. The normal equation is valuable conceptually; gradient descent is valuable for understanding iterative optimization and for models where a closed-form solution is unavailable or impractical.

### U.2 Why the two-parameter loss surface is bowl-shaped

For \(\hat y_i=mx_i+c\), MSE is

\[
L(m,c)=\frac1N\sum_i(y_i-mx_i-c)^2.
\]

It is a quadratic function of \(m,c\). Its level sets are ellipses (or degenerate ellipses) in parameter space, and the shape depends on the distribution of \(x\), its scale, and the coupling between slope and intercept. If \(x\) is not centered, changing \(m\) can be compensated partly by changing \(c\), tilting the contours. Centering \(x\) reduces this coupling.

This explains several visualization observations in the project:
- the trajectory descends toward the bottom of a bowl;
- contours need not be circles;
- the optimum can be tilted relative to the coordinate axes;
- the parameter path depends on the starting point and learning rate even though the convex objective has no spurious local minima.

### U.3 Gradient descent stability

For a quadratic objective with Hessian eigenvalues \(\lambda_i\), plain gradient descent behaves independently along each eigendirection. In the simplest positive-definite case, a learning rate satisfying

\[
0<\eta<\frac{2}{\lambda_{\max}}
\]

is required for convergence of the fixed-step linear iteration. If \(\eta\) is too large, the path can oscillate or diverge; if it is very small, convergence is slow. The condition number \(\kappa=\lambda_{\max}/\lambda_{\min}\) describes how stretched the bowl is and helps explain slow progress in narrow valleys.

This is why plotting only the loss value is not always enough. The parameter trajectory can reveal oscillation and poor conditioning that a loss curve hides.

### U.4 The project’s two different landscapes

**Recorded distinction:** the project had both:
1. an MSE surface computed from \(X\), \(y\), and \(\hat y=mx+c\); and
2. a synthetic non-convex scalar function, such as
   \[
   L(m,c)=m^2+c^2+2\sin(3m)\sin(3c).
   \]

The first illustrates the actual regression objective and is convex. The second is an artificial optimization landscape, not a regression model. It can have multiple local structures and is useful for demonstrating that gradient descent follows local derivatives rather than globally inspecting the entire surface.

For a trajectory visualization, store the parameters after each update. Be explicit about whether the initial point is included and whether the stored loss corresponds to parameters before or after the update; otherwise the plotted path and loss series may be offset by one step.

### U.5 Numerical gradient checking

A central finite-difference approximation for a scalar objective \(J(\theta)\) is

\[
\frac{\partial J}{\partial\theta_j}
\approx
\frac{J(\theta+\epsilon e_j)-J(\theta-\epsilon e_j)}{2\epsilon}.
\]

This is a useful check against autograd for a tiny model. Choose \(\epsilon\) neither so large that curvature dominates nor so small that floating-point rounding dominates. Compare relative error as well as absolute error, and use double precision for serious gradient checks. Finite differences become expensive in high-dimensional models because each parameter requires extra objective evaluations.

## V. The chain rule, computational graphs, and backpropagation

### V.1 Scalar chain rule

For \(u=g(x)\) and \(y=f(u)\),

\[
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}.
\]

For a neural network with successive layers,

\[
h_1=f_1(x),\quad h_2=f_2(h_1),\quad
\hat y=f_3(h_2),
\]

the derivative is a chain of local derivatives. Backpropagation computes these efficiently by reusing intermediate derivatives rather than differentiating every parameter independently from scratch.

### V.2 A one-hidden-layer example

Let

\[
z_1=W_1x+b_1,\qquad
h=\phi(z_1),\qquad
z_2=W_2h+b_2,\qquad
L=\ell(z_2,y).
\]

Backpropagation proceeds from the loss toward the input:

\[
\delta_2=\frac{\partial L}{\partial z_2},
\]
\[
\frac{\partial L}{\partial W_2}=\delta_2h^\top,\qquad
\frac{\partial L}{\partial b_2}=\delta_2,
\]
\[
\delta_1=(W_2^\top\delta_2)\odot\phi'(z_1),
\]
\[
\frac{\partial L}{\partial W_1}=\delta_1x^\top,\qquad
\frac{\partial L}{\partial b_1}=\delta_1.
\]

The Hadamard product \(\odot\) is element-wise multiplication. Matrix dimensions determine whether the products are valid; they are not decoration. With a batch, gradients sum or average over examples according to the loss reduction and implementation.

### V.3 Why autograd and backpropagation are related but not identical terms

- **Backpropagation** is the reverse-mode chain-rule algorithm used to compute derivatives efficiently for a scalar loss with respect to many parameters.
- **Automatic differentiation (autograd)** is the broader machinery that records or reconstructs computational dependencies and calculates derivatives.
- **Computational graph** describes the operations and dependencies used to calculate a result.
- PyTorch autograd usually builds a dynamic graph during the forward pass; `.backward()` traverses it in reverse to compute gradients.

Autograd is not symbolic algebra and does not generally simplify an expression into a human-readable closed form. It differentiates the operations actually executed. A wrong forward pass can produce perfectly consistent but wrong gradients.

### V.4 Leaf tensors, graph lifetime, and gradient accumulation

A tensor created directly with `requires_grad=True` is typically a leaf tensor. Its gradient is populated in `.grad` after backward. Intermediate non-leaf tensors do not retain `.grad` by default unless `retain_grad()` is called.

Important behaviours:
- Gradients accumulate into existing `.grad` buffers.
- `optimizer.zero_grad()` clears parameter gradients; some versions/options set gradients to `None` rather than filling with zeros.
- The same graph generally cannot be backpropagated through repeatedly after its saved intermediates are freed unless `retain_graph=True` is used. Retaining graphs unnecessarily consumes memory.
- In-place operations on tensors needed for gradient calculation can trigger autograd errors or invalidate assumptions.
- Use `torch.no_grad()` for parameter updates and evaluation computations that should not enter the graph.
- Avoid converting differentiable values to NumPy or Python numbers in the middle of a computation if gradients are still required.

## W. Tensor layout, shape, `view`, `reshape`, and `permute`

This section records a recurring source of confusion in the project.

### W.1 Shape is not layout

A tensor has a logical shape and a memory-stride description. Shape says how many elements each dimension represents. Strides describe how far memory moves when an index changes. Two tensors can have the same logical values and shape but different storage/layout characteristics.

### W.2 `view`

`x.view(new_shape)` creates a view of the same underlying storage, reinterpreting the elements under a new shape. It generally requires a stride-compatible layout; a non-contiguous tensor may not support the requested view.

Example:

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.view(2, 12)
```

This groups the same logical sequence of elements into a different shape. It does not transpose axes.

### W.3 `permute`

`x.permute(dims)` reorders the dimensions:

```python
# NCHW -> NHWC
x_nhwc = x_nchw.permute(0, 2, 3, 1)
```

For an image batch:
- NCHW means `(batch, channels, height, width)`;
- NHWC means `(batch, height, width, channels)`.

`permute` usually returns a view with changed strides, not a copy of reordered data. A subsequent `.view(...)` may fail if the strides are incompatible. `.contiguous()` can create a contiguous copy when required.

### W.4 `reshape`

`reshape` returns a view when possible and may allocate a copy otherwise. It is often more robust than `view` when the exact contiguity is uncertain, but the possibility of a copy matters for memory and performance. It does not substitute for `permute`: reshaping does not express a semantic reordering of axes.

### W.5 Practical diagnostic sequence

When a shape operation fails:
1. Print `x.shape`, `x.stride()`, and `x.is_contiguous()`.
2. Decide whether the goal is to merge/split dimensions (`view`/`reshape`) or reorder axes (`permute`/`transpose`).
3. If a contiguous layout is required, use `.contiguous()` deliberately and understand that it may copy.
4. Verify the semantic meaning of each axis, not only the final shape. A tensor can have the expected dimensions while the channels and spatial axes are incorrectly ordered.

## X. Batches, epochs, iterations, and loss reduction

### X.1 Vocabulary

- **Example/sample:** one input-target pair.
- **Batch:** examples processed together in one forward/backward pass.
- **Mini-batch:** a batch smaller than the full training set.
- **Iteration/step:** usually one optimizer update.
- **Epoch:** one pass through the entire training set.
- **Shuffle:** randomize example order before forming batches.

If the training set has \(N\) examples and batch size \(B\), one epoch contains approximately \(\lceil N/B\rceil\) updates when the final incomplete batch is retained. A “10,000 epochs” loop over a full tensor is 10,000 full-batch updates; a DataLoader loop may perform many updates per epoch.

### X.2 `reduction="mean"` versus `"sum"`

For a batch of \(B\) examples:
- `mean` typically averages element-wise loss values over the relevant output elements;
- `sum` adds them.

With `sum`, gradient magnitude usually grows with batch size. With `mean`, it is normalized by the number of elements used in the reduction. The exact denominator matters for losses with multiple output dimensions and masks. When comparing code or deriving gradients, state the reduction convention.

### X.3 Full-batch versus mini-batch optimization

Full-batch gradient descent uses the gradient of the complete training-set objective per update. Mini-batch gradients are estimates of that gradient and vary with batch sampling. Mini-batches reduce per-step compute and can provide useful stochasticity, but they also introduce variance in the optimization trajectory. A change from full-batch to mini-batch training is not merely a speed optimization: it changes the number and noise characteristics of updates per epoch.

## Y. Initialization, activations, and gradient propagation

### Y.1 Why hidden layers need nonlinearities

Composing affine maps alone does not create a more expressive family:

\[
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2).
\]

The composition is still affine. A nonlinear activation between layers prevents this collapse and permits nonlinear decision boundaries.

### Y.2 Sigmoid saturation

For sigmoid,

\[
\sigma'(z)=\sigma(z)(1-\sigma(z)).
\]

The derivative is at most \(1/4\) and approaches zero for large positive or negative \(z\). In deep compositions, repeatedly multiplying small derivatives can contribute to vanishing gradients. This is one reason sigmoid is common at a binary output but less common in hidden layers of modern deep feedforward networks.

### Y.3 ReLU and its limitations

\[
\operatorname{ReLU}(z)=\max(0,z).
\]

Its derivative is 1 on the positive side and 0 on the negative side (the derivative at exactly zero is conventionally assigned by the implementation). ReLU avoids sigmoid-style saturation on the positive side, but a unit that remains negative for all relevant inputs may receive no gradient through that activation; this is the “dead ReLU” concern. Leaky ReLU leaves a small negative slope.

### Y.4 Initialization scale

If activations grow or shrink systematically through layers, signal variance and gradient magnitudes can become unstable. Xavier/Glorot and He/Kaiming initialization are designed to control variance under different activation assumptions. They are not universal guarantees: residual connections, normalization layers, activation details, fan-in/fan-out, and architecture all matter.

### Y.5 Universal approximation is not a practical guarantee

A universal approximation theorem establishes that certain neural-network families can approximate broad classes of functions under stated conditions. It does not say:
- that a small network will learn the function from finite data;
- that gradient descent will find the right parameters;
- that the required width or data is practical;
- that approximation on observed data guarantees generalization.

The XOR example is a useful small illustration of representational limits: a single linear decision boundary cannot separate XOR labels in the original two-dimensional input space, while a network with a nonlinear hidden representation can.

## Z. Decision boundaries and logistic regression

### Z.1 Logistic regression is linear in log-odds

For binary logistic regression,

\[
z=w^\top x+b,\qquad
p(y=1\mid x)=\sigma(z).
\]

The log-odds are

\[
\log\frac{p}{1-p}=w^\top x+b.
\]

At a threshold \(p=0.5\), the boundary is \(w^\top x+b=0\), a hyperplane in the input feature space. Other probability thresholds shift the boundary to \(w^\top x+b=\log(\tau/(1-\tau))\).

The name “regression” refers to modelling a probability/log-odds function, not to predicting a continuous target. A hidden-layer network can form nonlinear boundaries by transforming the input before the output layer.

### Z.2 Threshold and probability are distinct

The output probability is a model score; the decision threshold converts it to a class. Moving the threshold changes precision and recall without changing the underlying logits. Threshold selection should be done on validation data, based on the costs of false positives and false negatives, and not tuned on the final test set.

## AA. Titanic experiment: more detailed audit record

### AA.1 Feature and preprocessing contract

**Recorded setup:** `sns.load_dataset("titanic")`, with columns:

```text
survived, pclass, sex, age, sibsp, parch, fare
```

The target is `survived`; the six input columns after encoding are `pclass`, `age`, `sibsp`, `parch`, `fare`, and one encoded `sex` indicator.

The discussed pipeline:
1. Select only these columns.
2. Impute missing `age`.
3. Encode `sex` with `pd.get_dummies(..., drop_first=True)`.
4. Split train/test with `test_size=0.2`, `random_state=42`, and stratification on `survived`.
5. Fit `StandardScaler` on training `age` and `fare`.
6. Transform both training and test columns using that fitted scaler.
7. Convert feature and target columns to float32 tensors.
8. Reshape binary target to `(N,1)`.
9. Train an MLP and compare train/test performance.

A later review specifically identified that imputation was initially performed before the split in one version. The median computed from all rows is a mild form of test-set leakage. The clean pipeline fits the imputer on training data only and applies that learned median to test data. The same train-only rule applies to scaling.

### AA.2 Exact tensor shapes in the documented split

For 891 rows, an 80/20 split with `test_size=0.2` produces 712 training rows and 179 test rows in the discussed scikit-learn configuration. The recorded training tensors were:
- `X`: `(712, 6)`
- `y` before reshape: `(712,)`
- `y` after `.view(-1, 1)`: `(712, 1)`

A `Linear(6,1)` model returns `(712,1)`, which aligns with `BCEWithLogitsLoss` targets of `(712,1)`.

### AA.3 Preprocessing subtleties

- **Median imputation:** median is robust to skew/outliers relative to the mean, but the choice is still a modelling decision. If the median is calculated before splitting, information from test data enters preprocessing.
- **`pclass`:** numeric values 1, 2, 3 imply an ordered representation and a particular geometry. One-hot encoding is an alternative when the equal-spacing assumption is undesirable.
- **`sibsp` and `parch`:** counts are discrete and often skewed. Standardizing them may be reasonable, but it changes interpretability; the project chose to scale continuous `age` and `fare` in the simpler teaching setup.
- **`fare`:** strongly skewed values can make standardization sensitive to outliers; a log transform could be tested later, but it should not be introduced while diagnosing a different bug.
- **`sex`:** one indicator suffices for two categories. The meaning of 0/1 depends on category ordering; check the resulting column name and mapping instead of assuming which sex equals one.
- **DataFrame column order:** model weights correspond to input positions. Persist the exact feature order for inference and ensure it is identical at training and prediction time.

### AA.4 Model size and reported results

Two discussed model capacities were:
- `6 → 4 → 1`: 33 parameters.
- `6 → 16 → 8 → 1`: 257 parameters.

In the 33-parameter architecture, the first layer computes four hidden features from six inputs, ReLU supplies nonlinearity, and the output layer produces one logit. It is deliberately small and is useful for showing the relationship between capacity and fit.

Reported observations across iterations included training accuracy around 86–87% and test accuracy around 75–82%, plus a later compact-model result near 87.77%. The source discussion moved through multiple fixes and model variants. Do not claim one is the final best result without reopening the actual notebook.

### AA.5 What the reported confusion matrix says

One recorded matrix was:

\[
\begin{bmatrix}
403 & 36\\
72 & 201
\end{bmatrix}
\]

Assuming rows are actual classes `[0,1]` and columns predicted classes `[0,1]`, this gives:
- TN = 403
- FP = 36
- FN = 72
- TP = 201
- total = 712
- accuracy \(=(403+201)/712\approx84.83\%\)
- precision for class 1 \(=201/(201+36)\approx84.81\%\)
- recall for class 1 \(=201/(201+72)\approx73.63\%\)
- F1 for class 1 \(\approx78.83\%\)

**Caveat:** this matrix's total equals 712, the documented training-set size, not the 179-row test split. It may therefore be a training-set evaluation or another split/context. Never label it as test performance without verifying the code that generated it. This is an important example of why a plausible-looking metric is not enough: the evaluated population matters.

### AA.6 Debugging sequence for the Titanic run

When results looked odd, the discussion narrowed the problem in this order:
1. Inspect the data and feature matrix shapes.
2. Verify target shape/dtype and output shape.
3. Confirm the loss expects logits rather than probabilities.
4. Confirm scaling was fitted on training data and applied to both sets.
5. Confirm evaluation uses `model.eval()` and `torch.no_grad()`.
6. Print loss at sensible intervals and inspect the learning curve.
7. Compare training and test performance.
8. Check number of epochs and learning rate.
9. Compare against a simpler baseline/logistic regression.
10. Change one variable at a time.

The source explicitly discouraged assuming the model architecture was at fault before verifying the data and evaluation pipeline.

## AB. Binary loss and output-head contracts in detail

### AB.1 `BCELoss` versus `BCEWithLogitsLoss`

`BCELoss` expects probabilities \(p\in[0,1]\). If the model returns logits \(z\), one would need a sigmoid first:

```python
probabilities = torch.sigmoid(logits)
loss = torch.nn.BCELoss()(probabilities, targets)
```

`BCEWithLogitsLoss` takes logits directly and combines sigmoid with BCE using a numerically stable formulation:

```python
loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
```

The second pattern is generally preferred. If sigmoid is applied in `forward()` and the result is then passed to `BCEWithLogitsLoss`, sigmoid is effectively applied twice, producing incorrect loss semantics.

### AB.2 Derivative intuition for binary cross-entropy with logits

For a single target \(y\) and logit \(z\), the sigmoid/BCE combination has the particularly clean derivative

\[
\frac{\partial L}{\partial z}=\sigma(z)-y.
\]

This is one reason logits are a natural output representation: the gradient directly compares predicted probability with the target. It also helps explain why the model can receive a useful gradient when it is confidently wrong, although extreme floating-point values still require stable implementations.

### AB.3 `pos_weight` and class imbalance

`BCEWithLogitsLoss` supports `pos_weight` to increase the contribution of positive examples. It changes the objective and often shifts the precision/recall trade-off; it is not a free correction that leaves the meaning of predicted probabilities unchanged. Select it based on the problem's error costs and validation evidence, and evaluate threshold/calibration behaviour after training.

### AB.4 Reduction and multi-label classification

For a multi-label task, output shape is often `(B, C)` with independent binary targets per class and BCE-with-logits. This differs from multiclass single-label classification, where classes are mutually exclusive and `CrossEntropyLoss` is typical. Do not choose a loss based solely on the number of output units; choose it based on the target semantics.

## AC. Evaluation: beyond accuracy

### AC.1 Confusion-matrix orientation must be documented

Libraries and plotting helpers differ in presentation conventions, although scikit-learn's standard `confusion_matrix(y_true, y_pred)` uses rows for true classes and columns for predicted classes. Before interpreting a matrix, state the class order and orientation. For binary labels ordered `[0,1]`, the usual layout is `[[TN, FP], [FN, TP]]`.

### AC.2 Precision, recall, specificity, and F1

\[
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall/Sensitivity}=\frac{TP}{TP+FN},
\]
\[
\text{Specificity}=\frac{TN}{TN+FP},\quad
F_1=\frac{2TP}{2TP+FP+FN}.
\]

- Precision matters when false positives are costly.
- Recall matters when missing a true positive is costly.
- Specificity measures how well actual negatives are rejected.
- F1 is the harmonic mean of precision and recall; it does not incorporate true negatives directly and should not be treated as a universal metric.
- Macro averaging gives each class equal weight; weighted averaging weights by class support; micro averaging aggregates counts before computing the metric. They answer different questions.

### AC.3 ROC and precision-recall curves

A ROC curve plots true-positive rate against false-positive rate as the threshold changes. ROC AUC summarizes ranking performance across thresholds but can look optimistic under severe class imbalance. A precision-recall curve focuses on the positive class and is often more informative when positives are rare. Neither curve identifies the operational threshold without costs or constraints.

### AC.4 Calibration

A classifier is calibrated when predictions assigned probability \(p\) correspond to approximately \(p\) empirical event frequency among comparable cases. Accuracy and ranking metrics do not guarantee calibration. If probabilities are used for decisions, risk estimation, or thresholding, calibration should be checked separately.

### AC.5 Undefined values are part of the metric specification

When a denominator is zero, the ratio is mathematically undefined. Libraries may map the result to zero, one, or `NaN`, or warn, depending on the metric and configuration. Store both:
- the mathematical condition (e.g. no predicted positives);
- the software policy used to report a number.

This is preferable to saying the metric “equals zero” without qualification.

## AD. Underfitting, overfitting, and model capacity

### AD.1 Training/validation patterns

| Training loss | Validation loss | Typical interpretation |
|---|---|---|
| High and still high | High | Underfitting, optimization failure, or poor features |
| Decreasing | Decreasing | Learning useful structure so far |
| Decreasing | Flat | Generalization improvement has stalled |
| Decreasing | Increasing | Overfitting may be emerging |
| Erratic | Erratic | Small/unstable validation set, excessive learning rate, noisy optimization, or pipeline problems |

These are diagnostic patterns, not proofs. Data leakage, distribution shift, inconsistent preprocessing, incorrect model mode, and noisy labels can produce misleading curves.

### AD.2 Capacity and regularization sequence in the Titanic notebook

The recorded narrative was:
1. a simple model underfit;
2. a model with hidden layers fit training data better;
3. validation/test behaviour exposed overfitting;
4. weight decay slowed fitting and improved the held-out loss curve in one run;
5. early stopping was proposed as the next concept to discover from the validation-loss curve.

The point is not that every model must follow this exact progression, but that the project deliberately used the same dataset to make model capacity and generalization visible.

### AD.3 Bias–variance framing

A high-bias model cannot express important structure; a high-variance model changes too much with the particular training sample. Increasing capacity can reduce approximation error while increasing sensitivity to the data. Regularization, more data, augmentation, and appropriate model selection can alter this trade-off. The familiar bias–variance decomposition is exact under particular assumptions (e.g. squared-error regression), so do not treat it as a universal literal decomposition of every classification error.

## AE. Optimizer details beyond the basic formulas

### AE.1 Gradient direction versus step magnitude

For Adam,

\[
\Delta\theta_t \approx -\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

The first-moment estimate \(\hat m_t\) smooths the signed gradient history. The second-moment estimate \(\hat v_t\) scales coordinates according to recent squared-gradient magnitude. It is useful to say that Adam smooths direction and adapts magnitude, but the two roles are coupled by coordinate-wise normalization; they are not a perfect orthogonal decomposition into direction and magnitude.

### AE.2 Epsilon placement matters

The common expression places \(\epsilon\) outside the square root:

\[
\sqrt{\hat v_t}+\epsilon.
\]

Different algorithms or implementations can use different placements or conventions. When explaining exact optimizer behaviour, use the specific algorithm/API formula rather than treating all adaptive methods as algebraically identical.

### AE.3 Weight decay and parameter groups

In real networks, weight decay is often not applied uniformly:
- biases may be excluded;
- normalization parameters may be excluded;
- weights may use one decay value while other parameters use another;
- pretrained backbone and newly initialized head may use different learning rates.

These are common implementation practices, not a project-wide settled rule. Record the optimizer parameter groups explicitly when comparing runs.

### AE.4 Schedulers and warmup

A learning-rate scheduler changes \(\eta\) over training:
- step decay lowers it at selected epochs;
- exponential decay multiplies it by a factor each step/epoch;
- cosine schedules reduce it smoothly;
- warmup begins at a low rate and increases toward the target rate over an initial interval.

Warmup can reduce instability early in training, especially with large models or large effective batch sizes. It does not compensate for incorrect labels, normalization, loss contracts, or a broken optimizer parameter group.

## AF. Convolution arithmetic and computational cost

### AF.1 Output-size formula

For one spatial dimension,

\[
L_{\text{out}}=
\left\lfloor
\frac{L_{\text{in}}+2P-D(K-1)-1}{S}+1
\right\rfloor.
\]

Use this for height and width separately. For a common 3×3 convolution with stride 1, padding 1, dilation 1, spatial size is preserved. For stride 2, spatial size is approximately halved depending on padding and input parity.

### AF.2 Parameters versus operations

Parameter count does not directly predict runtime:
- convolutional weights are reused at many spatial positions;
- activation tensors may be large even when parameter count is small;
- input resolution affects convolutional multiply-accumulate operations;
- batch size changes activation memory and throughput;
- memory bandwidth, kernel launch overhead, data loading, precision, and hardware also matter.

This distinction was relevant to the transfer-learning experiment: 224×224 inputs were much more expensive than 32×32 inputs, despite using the same backbone weights.

### AF.3 Why flattening an image into a dense layer is expensive

For a 224×224 RGB image, the raw flattened input has \(224\cdot224\cdot3=150{,}528\) values. A dense layer from this input to 512 units would require \(150{,}528\cdot512+512\) parameters—over 77 million—whereas a convolution shares a small kernel across locations. This is a mathematical illustration, not a layer that the project necessarily instantiated.

### AF.4 Global average pooling

Global average pooling averages each feature map over its spatial dimensions, converting `(B, C, H, W)` into `(B, C)` in a common implementation. It can replace a large flatten-and-dense head, reducing parameters and encouraging the channel activations to represent image-level evidence. It may reduce spatially detailed information, and the exact use depends on architecture.

## AG. ResNet internals and feature transfer

### AG.1 Residual addition requires compatible shapes

For \(y=F(x)+x\), the tensor shapes must match. If spatial size or channel count changes, the skip path uses a projection or another alignment operation. A 1×1 convolution can change channels; stride can change spatial resolution. This is a structural constraint, not an optional detail.

### AG.2 Basic block and bottleneck arithmetic

A basic block commonly has two 3×3 convolutions. A bottleneck commonly uses 1×1 reduction, 3×3 processing, and 1×1 expansion. The bottleneck's internal width is smaller than its output width, reducing the cost of the spatial convolution. ResNet-50 uses bottleneck blocks; ResNet-18 uses basic blocks.

### AG.3 Freezing a backbone

Setting `requires_grad=False` on backbone parameters prevents gradients from being accumulated for those parameters, but it does not automatically:
- switch BatchNorm layers to evaluation behaviour;
- disable Dropout if present;
- prevent the forward pass from constructing every possible graph node in all configurations;
- guarantee the feature distribution matches the new task.

For a frozen feature extractor, many implementations call the backbone in eval mode and train only the new head. Whether BatchNorm statistics should remain frozen depends on dataset size and domain shift; make the decision explicit.

### AG.4 Fine-tuning gradually

A common strategy is:
1. load pretrained weights;
2. replace the classification head;
3. train the new head with the backbone frozen;
4. unfreeze later/high-level backbone stages;
5. use a smaller learning rate for pretrained weights than for the new head;
6. validate whether the additional trainable capacity improves held-out performance.

This is a useful protocol, not proof that gradual unfreezing is always optimal. A fully frozen backbone can fail because its representation is mismatched to the task; fine-tuning can help, but can also overfit or destroy useful pretrained features if the learning rate is too large.

## AH. Transfer-learning experiment: recorded configuration and implications

### AH.1 Configuration captured in project context

The separate transfer-learning experiment used:
- a pretrained `timm` ResNet-18;
- Fashion-MNIST as the target dataset;
- grayscale images converted to three channels;
- input resize to 32×32;
- the original ImageNet classification head discarded;
- a new `Linear(512, 10)` classification head;
- the backbone initially frozen.

A critical missing preprocessing step was identified: ImageNet-style normalization. The expected channel-wise values for common ImageNet-pretrained models are

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The normalization must be applied after converting/resizing the image into the appropriate tensor representation. Check the specific model's preprocessing configuration rather than assuming every checkpoint uses these values.

### AH.2 Why three channels do not create RGB information

Replicating a grayscale channel produces three numerically identical channels. It satisfies an RGB-shaped input contract, but the channels are perfectly correlated and contain no additional chromatic information. The model's pretrained first-layer filters were learned on natural RGB images, so the representation mismatch remains even when the tensor shape is accepted.

### AH.3 Why 32×32 versus 224×224 is not a trivial choice

The pixel-area ratio is

\[
\frac{224^2}{32^2}=49.
\]

Thus, at the same batch size and comparable architecture, early spatial feature maps can be much more expensive at 224×224. The source discussion recorded approximately 4.5–5 hours per epoch for a 224×224 setup, while 32×32 was used to make the experiment feasible. That runtime is hardware/configuration-specific. A 32×32 input is valid only if the chosen architecture's stem and later layers support it; pretrained weights do not themselves require 224×224 as a mathematical law.

### AH.4 Recorded performance and the likely diagnosis

The frozen-backbone experiment produced roughly 70% train/test accuracy in one discussion. The missing ImageNet normalization was identified as a major pipeline problem. A subsequent proposed experiment unfroze `layer4` and trained the head with Adam at `lr=1e-4`; a one-epoch result was recorded as training loss about 0.9427, validation loss about 0.6456, and accuracy about 0.7926.

Do not infer a final conclusion from a single epoch. The next checks are:
- confirm preprocessing matches the checkpoint;
- verify train/eval modes and BatchNorm behaviour;
- verify the new head is trainable and included in the optimizer;
- inspect per-epoch train/validation curves;
- compare frozen and fine-tuned runs under identical splits;
- establish the class-index mapping and evaluation code.

### AH.5 Normalization is part of the model contract

Pretrained weights were optimized for a particular input distribution. Scaling pixels to `[0,1]` and then applying channel-wise standardization are separate operations. Using the wrong mean/std, skipping normalization, normalizing twice, or applying normalization in a different order shifts the input distribution. A shape-compatible tensor is not necessarily a semantically compatible input.

## AI. Instrumentation and TensorBoard: observed failure modes

This topic was recorded in project discussions and is useful because logging itself became a source of performance and diagnostic problems.

### AI.1 Logging overhead can dominate a small model

A small Iris network has only 123 trainable parameters. The discussion observed that per-epoch TensorBoard diagnostics could create roughly 70–100 logging operations, enough for instrumentation to dominate the actual training work. The recommended separation was:

- **FastLogger:** scalar metrics and essential values every epoch.
- **DiagnosticLogger:** heavier diagnostics every 25–100 epochs.
- **ArtifactLogger:** expensive artifacts infrequently.

The precise interval should be chosen according to model size, runtime, and diagnostic need. A tiny model may spend more time serializing/logging than computing gradients.

### AI.2 Histograms and event-file growth

A TensorBoard event file grew to approximately 260 MB in a run. The discussion attributed this primarily to excessive histogram logging. Histograms of weights, activations, or gradients can be useful for diagnosing saturation, dead units, or exploding values, but writing them every epoch across many tensors produces substantial I/O and storage.

A safe debugging sequence is:
1. create a clean run directory;
2. test scalar-only logging;
3. confirm steps are monotonic;
4. add one histogram family;
5. inspect file growth;
6. add more diagnostics only if they answer a concrete question.

### AI.3 Out-of-order steps and nested runs

TensorBoard showed out-of-order steps in connection with a nested `add_hparams()` run. Keep each run's event stream and global-step sequence coherent. `add_hparams()` can create an additional run/logging structure; inspect the generated directory hierarchy instead of assuming all tags share one simple step series. Do not mix runs in a way that makes the displayed curves look like a single continuous experiment.

### AI.4 Reproducibility metadata

At minimum, record:
- model and hyperparameters;
- dataset split/random seed;
- preprocessing configuration;
- optimizer and parameter groups;
- learning-rate schedule;
- batch size and epoch count;
- library/device versions where relevant;
- best checkpoint and selected epoch;
- metrics on train/validation/test;
- source code or commit identifier.

A plot without the configuration that generated it is difficult to interpret later.

## AJ. Dataset splits and leakage beyond the Titanic example

### AJ.1 Split the independent unit, not automatically the row

If rows are correlated—multiple windows from one recording, repeated measurements from one device, multiple images of the same object, or multiple records from the same person—the split should usually occur at the level of the independent source. Random row-level splitting can put near-duplicates into both train and validation.

### AJ.2 Split before generating overlapping windows

For audio windows of length 2 seconds and stride 1 second, adjacent windows share half their signal. If a recording is segmented first and windows are randomly divided, the model may see nearly the same acoustic material in training and validation. The project voice-classifier discussion explicitly identified this as a major leakage risk.

Correct order:

```text
Original recordings
       ↓
Split by recording/session/speaker as appropriate
       ↓
Segment each split independently
       ↓
Generate spectrograms
       ↓
Train / validate / test
```

The independent grouping depends on the intended deployment question. If the aim is to generalize to new recording sessions of known speakers, split by session. If the aim is to generalize to unseen speakers, split by speaker. The test design must match the claim.

### AJ.3 Leakage through preprocessing

Leakage can occur through:
- scaler or imputer statistics fit on all rows;
- feature selection on the full dataset;
- PCA fitted before splitting;
- oversampling before splitting;
- augmentation variants of the same source appearing in different splits;
- target-derived or future information in features;
- repeated hyperparameter tuning against the test set.

The remedy is not just “use train/test split.” It is to ensure every data-dependent choice is learned from training data or validation data only, with the final test set held back until selection is complete.

## AK. Audio preprocessing: more technical details

### AK.1 Sampling rate and Nyquist limit

A digital audio waveform sampled at \(f_s\) samples per second can represent frequencies below approximately \(f_s/2\), the Nyquist frequency, subject to filtering and real-world details. At 48 kHz the nominal Nyquist limit is 24 kHz; at 16 kHz it is 8 kHz. Downsampling therefore removes frequency content above the lower Nyquist limit and requires anti-alias filtering.

### AK.2 Windowing and overlap

For a 2-second segment at 16 kHz, each mono waveform window contains 32,000 samples. With a 1-second stride, the hop between windows is 16,000 samples and adjacent windows overlap by 16,000 samples. If stereo is present, decide whether to preserve channels or downmix to mono consistently.

### AK.3 Spectrogram parameters are part of the feature definition

A spectrogram is not uniquely defined by the word “spectrogram.” Important choices include:
- FFT/window length;
- hop length;
- window function;
- magnitude versus power;
- log magnitude/log power;
- frequency range;
- centering and padding;
- normalization;
- image resizing and channel conversion.

Changing these changes the model's input distribution. Store them as configuration alongside the model checkpoint.

### AK.4 Segment-level versus recording-level metrics

If many windows are extracted from one recording, computing accuracy across all windows weights long recordings more heavily. It can also give a misleading sense of sample size because windows are correlated. Depending on the deployment objective, aggregate predictions by recording or report both window-level and recording-level metrics. Confidence intervals should account for the grouping structure where feasible.

## AL. Classical ML comparison and baselines

### AL.1 Why compare an MLP with logistic regression?

The logistic-regression baseline has a direct linear decision boundary in the encoded feature space. An MLP with hidden nonlinear layers can express more complex boundaries, but adds parameters and optimization complexity. If the MLP does not outperform logistic regression on validation data, possible explanations include:
- the relationship is nearly linear;
- data size is too small for the added capacity;
- regularization/optimization is suboptimal;
- features are weak or noisy;
- the comparison is not controlled.

A more complex model is not automatically better.

### AL.2 Majority-class baseline

For a binary dataset with class proportion \(p\), always predicting the majority class achieves accuracy \(\max(p,1-p)\). This baseline can look strong on imbalanced data while having zero recall for the minority class. Report the class distribution and at least one baseline before celebrating a model's accuracy.

### AL.3 Baseline comparisons must share the pipeline

Models should use the same train/validation/test split and comparable feature information. If one model receives a differently processed dataset or has its threshold tuned on test data, the comparison is not fair.

## AM. Numerical precision, device, and runtime notes

### AM.1 Floating-point types

- `float32` is a common training dtype and balances precision, memory, and speed.
- `float64` can help with numerical checks and some sensitive calculations but is slower and more memory-intensive on many GPUs.
- Integer targets are appropriate for class indices in `CrossEntropyLoss`; binary BCE targets are usually floating-point values matching the output shape.
- Converting data to the wrong dtype can cause an error or silently alter semantics.

### AM.2 GPU availability does not guarantee acceleration

Small models and small datasets may run faster on CPU because transfer and kernel-launch overhead dominate. Large images, convolutions, and larger batches are more likely to benefit from a GPU. Benchmark a representative complete epoch, including data loading and preprocessing, rather than timing only the forward pass.

### AM.3 Reproducibility is layered

Setting `random_state` or a PyTorch seed is necessary for repeatable experiments but may not be sufficient. DataLoader workers, nondeterministic kernels, library versions, GPU architecture, and parallel reductions can affect exact results. Record the seed and environment, and distinguish “same experimental setup” from bitwise-identical results.

## AN. Model outputs and inference contracts

### AN.1 Training output versus user-facing output

A model's raw output is not always the final result:
- regression: output may be the prediction in the target's scale;
- binary classification: one logit, optionally converted to a probability;
- multiclass classification: \(K\) logits, optionally converted to probabilities;
- multi-label classification: one logit per independent label.

If the target was standardized or transformed, predictions may need an inverse transform before being reported in original units. If categorical values were encoded, inference must apply the same mapping as training.

### AN.2 Persist preprocessing with the model

Saving only model weights is insufficient if inference also depends on:
- feature order;
- categorical mapping;
- imputation values;
- scaler mean and scale;
- image normalization;
- audio resampling and spectrogram parameters;
- class-to-index mapping;
- decision threshold.

These are part of the model artifact. A model can load successfully and still give invalid predictions if the preprocessing contract is lost.

## AO. Research and architecture timeline: technical interpretation

The project's historical survey touched perceptrons, early neural-network limitations, backpropagation, convolutional networks, AlexNet-era deep learning, transformers, and LLMs. The useful technical distinctions are:

- A **perceptron** with a threshold activation learns a linear decision boundary; a single perceptron cannot solve non-linearly separable XOR.
- A **multilayer network** introduces intermediate representations and nonlinear composition, allowing nonlinear boundaries.
- **Backpropagation** made gradient calculation through layered compositions practical; it is an algorithm for derivatives, not by itself a guarantee of good optimization.
- **CNNs** exploit local structure and weight sharing in spatial data.
- **Residual connections** improve optimization in deep networks by providing identity-like paths.
- **Transformers** use attention and feed-forward blocks rather than recurrence as their primary sequence-mixing mechanism; positional information must be supplied or learned because plain self-attention alone is permutation-equivariant over token positions.
- **LLMs** are trained with large-scale objectives and data; next-token prediction is a core pretraining objective, but instruction tuning, preference optimization, retrieval, tool use, and inference-time decoding are separate components.

These statements are technical orientation, not a claim that every listed architecture has already been taught in the course.

## AP. Consolidated equations sheet

### Regression
\[
\hat y=wx+b
\]
\[
\operatorname{MSE}=\frac1N\sum_i(y_i-\hat y_i)^2
\]
\[
w\leftarrow w-\eta\frac{\partial L}{\partial w}
\]

### Linear layer
\[
Y=XW^\top+b,\qquad
P=n_{\text{in}}n_{\text{out}}+n_{\text{out}}
\]

### Binary classification
\[
p=\sigma(z)=\frac1{1+e^{-z}}
\]
\[
L_{\text{BCE}}=-[y\log p+(1-y)\log(1-p)]
\]
\[
\frac{\partial L}{\partial z}=\sigma(z)-y
\quad\text{(single example, unweighted BCE)}
\]

### Multiclass classification
\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\qquad
L=-\log p_y
\]

### Metrics
\[
P=\frac{TP}{TP+FP},\quad
R=\frac{TP}{TP+FN},\quad
F_1=\frac{2TP}{2TP+FP+FN}
\]

### L2 objective
\[
J(\theta)=L(\theta)+\lambda\|\theta\|_2^2,\qquad
\nabla J=\nabla L+2\lambda\theta
\]

### Momentum / Adam
\[
v_t=\beta v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t
\]
\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]
\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]
\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

### CNN output size
\[
L_{\text{out}}=
\left\lfloor\frac{L_{\text{in}}+2P-D(K-1)-1}{S}+1\right\rfloor
\]

### Residual block
\[
y=F(x)+x
\quad\text{or}\quad
y=F(x)+W_sx
\]

### Audio RMS
\[
\operatorname{RMS}(x)=\sqrt{\frac1N\sum_nx_n^2}
\]

## AQ. Practical pre-flight checklist before trusting a result

Before interpreting a metric or changing an architecture, verify:

1. **Data identity:** which exact dataset and rows/files were used?
2. **Split unit:** are correlated samples kept together?
3. **Leakage:** were all data-dependent preprocessing steps fit on training data only?
4. **Feature contract:** exact columns, order, encoding, units, missing-value treatment?
5. **Shape contract:** input, logits, and target dimensions match?
6. **Dtype contract:** float targets for BCE; integer class indices for ordinary cross-entropy?
7. **Output/loss contract:** logits versus probabilities correct?
8. **Optimizer contract:** all intended parameters included, correct learning rates and decay groups?
9. **Mode contract:** `train()` during training, `eval()` during evaluation?
10. **Gradient contract:** gradients cleared, backward called, optimizer stepped?
11. **Metric contract:** correct class mapping, matrix orientation, zero-denominator convention?
12. **Baseline:** majority class or simpler model reported?
13. **Selection contract:** validation used for tuning, test held back?
14. **Reproducibility:** split seed, model configuration, preprocessing, and checkpoint recorded?
15. **Runtime contract:** timing includes data loading and logging, not only model computation?
16. **Artifact contract:** weights and preprocessing configuration saved together?

## AR. Source provenance and confidence

The technical notes in the two appendices draw primarily from the project export and the specific experiment discussions it contains. The following distinctions should survive future edits:

- **Recorded configurations:** Iris `4→10→5→3`; Titanic six-feature binary MLPs; audio windows of 2 seconds with 1-second stride; pretrained `timm` ResNet-18 on Fashion-MNIST; TensorBoard logging overhead observations.
- **Recorded values from particular runs:** Titanic accuracy/loss/confusion-matrix examples; voice-classifier validation accuracy range; transfer-learning epoch timing and one-epoch metrics; TensorBoard event-file size. These are contextual observations, not general performance claims.
- **Derived values:** parameter counts, tensor shapes, metrics calculated from the given confusion matrix, and pixel-area ratios.
- **General reference additions:** finite-difference checks, Hessian conditioning, ROC/PR interpretation, calibration, `pos_weight`, scheduler details, convolution arithmetic, and several inference-contract recommendations. These are standard technical context and are not all documented as decisions in the project.
- **Unresolved:** the final state of the transfer-learning notebook, the definitive best Titanic run, and any implementation detail that depends on code after the export. Reopen the actual notebook rather than inferring a final answer from this document.

The original project files remain authoritative when this appendix and executable code disagree.
