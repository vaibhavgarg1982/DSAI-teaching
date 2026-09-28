# Timeline of Neural Networks and Deep Learning

* **1943 — McCulloch & Pitts**

  * Artificial neuron
  * Neural computation

* **1948 — Turing**

  * Machine intelligence
  * Early recurrent neural ideas

* **1956 — Dartmouth**

  * AI as a field
  * Symbolic AI vs connectionism

* **1957–60 — Rosenblatt**

  * Perceptron
  * Learning from examples
  * Decision boundary
  * Perceptron convergence theorem

* **1969 — Minsky & Papert**

  * Perceptron limitations
  * **XOR**
  * Linear separability
  * Single layer ≠ sufficient
  * Need hidden layers
  * But: how to train them?

* **1970s — First AI winter**

  * Early optimism → unmet expectations
  * Funding / interest decline
  * Perceptron limitations
  * Symbolic AI limitations

* **Late 1970s–1980s — Connectionist revival**

  * Multilayer networks
  * **Backpropagation**
  * Rumelhart, Hinton & Williams — 1986
  * Train the hidden layers
  * XOR becomes solvable

* **1980s — Universal approximation**

  * Cybenko — 1988/89
  * Multilayer networks → broad function approximation
  * Expressive power of nonlinear networks

* **1959–68 — Hubel & Wiesel**

  * Visual cortex
  * **Simple cells → edges / orientations**
  * **Complex cells → local feature invariance**
  * Hierarchical visual processing
  * Local receptive fields
  * Increasingly complex representations
  * **Edges → patterns → objects**

* **1980 — Fukushima**

  * Neocognitron
  * Early CNN-like architecture
  * Biological vision → computational architecture

* **1990s — LeCun**

  * Backprop + convolution
  * Handwritten digits
  * CNNs become practical
  * Local connectivity
  * Weight sharing
  * Pooling
  * Hierarchical features

* **1980s–90s — Second AI boom / winter**

  * Expert systems
  * Rule-based AI
  * Knowledge-engineering bottleneck
  * Brittleness
  * Expectations → reality gap
  * AI winter

* **1990s–2000s — Neural networks in the wilderness**

  * CNNs
  * RNNs
  * LSTM
  * Backpropagation through time
  * Vanishing gradients
  * Continued research, limited mainstream impact

* **2006 onward — Deep learning revival**

  * Hinton
  * Deep networks
  * Learned representations
  * Feature engineering → feature learning

* **2012 — AlexNet / ImageNet**

  * Deep CNN
  * Major performance jump
  * GPUs
  * Big data
  * ReLU
  * Mini-batch SGD
  * Deep learning becomes mainstream

* **2010s — Scaling the training**

  * Better initialization
  * SGD / mini-batches
  * Batch normalization
  * Dropout
  * Weight decay
  * Deeper networks

* **2015–16 — ResNet**

  * Depth creates optimization problems
  * **Residual / skip connections**
  * Very deep networks become trainable
  * Architecture as optimization aid

* **2010s — Overparameterization**

  * Huge models
  * Zero / near-zero training error
  * Still generalize
  * Classical overfitting intuition challenged
  * Implicit regularization

* **2017 onward — Transformers**

  * Attention
  * Sequence modelling
  * Scaling
  * Foundation models

* **Overall arc**

  * Artificial neuron
    → perceptron
    → **XOR**
    → hidden layers
    → backprop
    → universal approximation
    → biological vision
    → hierarchical features
    → CNN
    → AI winters
    → deep learning
    → scale
    → ResNet
    → Transformers


### From artificial neurons to deep learning

The story begins in the **1940s**, when McCulloch and Pitts proposed a mathematical model of a neuron. The basic idea was remarkably simple: perhaps complicated computation—and eventually intelligence—could emerge from networks of simple computational units.

In **1948**, Turing was already asking the broader question of whether machinery could exhibit intelligent behaviour. By the **1950s**, this had become the emerging field of artificial intelligence, with two broad approaches developing: **symbolic AI**, which attempted to explicitly represent knowledge and reasoning, and **connectionism**, which tried to obtain intelligent behaviour from networks of simple units.

In **1957**, Frank Rosenblatt popularized the **perceptron**. This was an important conceptual shift: rather than programming a decision rule explicitly, we could give a machine examples and allow it to learn the weights that determine its decisions. By **1960**, the perceptron convergence theorem provided a mathematical foundation for this learning process.

But then came a very important problem: **XOR**.

A perceptron can learn AND and OR because their classes can be separated by a straight line. XOR cannot: no single straight line can separate its two classes. This is the **linear separability** problem. The perceptron was not simply "bad at XOR"; the deeper point was that a **single layer did not have enough representational power**. We needed hidden layers.

The difficulty was that, at the time, there was no effective general method for training those hidden layers.

This limitation, combined with much broader difficulties in AI, contributed to the first **AI winter of the 1970s**. The early enthusiasm for AI had produced expectations that the technology could not yet meet. Funding and interest declined.

The story did not end there. During the **late 1970s and 1980s**, connectionism began to revive. The crucial development was **backpropagation**: a practical way of propagating the error through multiple layers and adjusting the weights of the hidden layers. The influential 1986 work of Rumelhart, Hinton and Williams helped establish backpropagation as the key mechanism for training multilayer neural networks.

The XOR problem now had a conceptual solution:

**perceptron → hidden layers → nonlinear representations → backpropagation**

And mathematical results in the **late 1980s**, particularly the universal approximation results, showed just how powerful multilayer networks could be: under suitable conditions, they could approximate broad classes of functions.

At roughly the same time, another line of research was asking a different question:

**What if the architecture itself reflected the structure of the problem?**

For vision, there was already a remarkable source of inspiration: the human visual system.

Between **1959 and 1968**, Hubel and Wiesel's experiments on the visual cortex identified neurons with different response properties. Some **simple cells** responded to local visual structures such as oriented edges. Other **complex cells** responded more flexibly and showed tolerance to small changes in the position of those features.

The important idea was **hierarchy**.

The visual system does not appear to treat an image as one enormous undifferentiated set of pixels. It can build increasingly complex representations from local features:

**edges → simple patterns → more complex patterns → higher-level visual structures**

This provided a striking conceptual analogy for what would eventually become the **convolutional neural network**:

**local receptive fields → feature detectors → pooling/invariance → hierarchical representations**

The analogy in the historical literature is particularly neat: convolutional outputs can be thought of as analogous to the responses of simple cells, while pooling provides some of the tolerance associated with complex cells.

Fukushima's **1980 neocognitron** was an early architecture inspired by this organization of visual processing. But the architecture alone was not enough. What was still needed was an effective way to learn its parameters.

That arrived when LeCun and collaborators demonstrated effective training of convolutional networks using backpropagation in the **1990s**, including the successful application of CNNs to handwritten-digit recognition.

So the two stories—**backpropagation** and **biological vision**—come together:

**perceptron**
→ discovers learning from examples
→ **XOR exposes the limitation of a single layer**
→ **hidden layers + backpropagation** make multilayer learning practical
→ neuroscience suggests **hierarchical visual representations**
→ **CNNs** incorporate locality, weight sharing and pooling
→ practical success on image recognition

But neural networks still did not immediately take over AI.

The **1980s** also saw a major boom in **expert systems**, where human expertise was encoded explicitly as rules. These systems became commercially important, but eventually encountered problems such as the cost and brittleness of knowledge engineering. As expectations again outran practical capability, AI entered another downturn—the **second AI winter of the late 1980s and 1990s**.

Meanwhile, neural networks continued to develop, but remained a relatively specialized approach. CNNs, RNNs, LSTMs and other ideas were being developed, but the ingredients needed for another major breakthrough were not yet in place.

Then, in the **2000s**, the emphasis increasingly shifted toward **deep representation learning**: instead of manually designing features and then giving those features to a machine-learning algorithm, let the network learn the representations itself.

Several developments converged:

* much larger datasets
* increasingly powerful GPUs
* deeper architectures
* ReLU activations
* improved initialization
* stochastic/mini-batch gradient descent
* better regularization
* later, batch normalization

The decisive public demonstration came in **2012**, when AlexNet achieved a dramatic result on ImageNet.

This was not one magical new algorithm. It was the point at which several decades of ideas finally became scalable:

**learned representations + deep networks + CNNs + GPUs + large datasets + improved optimization**

The result was the beginning of the modern deep-learning era.

The next problem was depth itself. If more layers give us more expressive representations, why not simply keep adding layers? In practice, very deep networks became difficult to optimize. **ResNet, 2015–2016**, addressed this with residual/skip connections, making much deeper networks trainable.

Again, the historical pattern repeats:

**new capability → new limitation → architectural innovation**

And the networks kept getting larger. During the **2010s**, overparameterized networks demonstrated that enormous models could fit their training data almost perfectly while still generalizing surprisingly well. This challenged the simple intuition that having more parameters must necessarily mean worse generalization.

Finally, from **2017 onward**, the Transformer introduced another major architectural shift. Attention-based models provided a powerful mechanism for learning relationships across sequences, and combined with massive datasets and computation, this eventually led to today's foundation-model era.

So the entire story can be seen as a succession of questions:

**1940s:**
Can simple mathematical units perform computation?

**1950s:**
Can machines learn from examples?

**1960s:**
What are the limitations of a single layer?
→ **XOR**

**1980s:**
Can we train multilayer networks?
→ **Backpropagation**

**1980s–90s:**
Can we build architectures that reflect the structure of the data?
→ **CNNs and hierarchical visual representations**

**1990s–2000s:**
Why aren't neural networks dominating despite all this promise?
→ limitations of computation, data and optimization

**2010s:**
What happens when we combine learned representations with enormous datasets and computation?
→ **Deep learning**

**Today:**
What happens when we scale these ideas to enormous models and diverse modalities?
→ **Foundation models**

The useful lesson is therefore not a chronology to memorize. It is a **problem-solving history**:

**Perceptron** exposed the need for representation.
**XOR** exposed the limitation of a single layer.
**Backpropagation** made multilayer learning practical.
**Neuroscience** inspired hierarchical visual architectures.
**CNNs** exploited the structure of images.
**AI winters** reminded the field that capability and expectations are not the same thing.
**Data + GPUs + better optimization** made deep networks practical at scale.
**ResNets and later architectures** removed new optimization barriers.
And the cycle continues.
