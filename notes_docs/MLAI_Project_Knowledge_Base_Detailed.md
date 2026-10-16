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
| Huber | \(\frac12e^2\) for \(|e|\le\delta\); \(\delta(|e|-\frac12\delta)\) otherwise | Quadratic near zero, linear for large errors; balances smoothness and robustness. |
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

