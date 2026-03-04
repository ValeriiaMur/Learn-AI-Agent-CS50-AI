# Neural Networks

## Overview

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (artificial neurons) organized in layers that learn to transform inputs into outputs by adjusting the strength of connections during training. Neural networks are the foundation of **deep learning** — the use of networks with multiple hidden layers — and power many of today's most impressive AI systems, from image recognition to language generation.

## The Artificial Neuron

The basic building block is the **artificial neuron** (or perceptron). It receives multiple inputs, each multiplied by a weight, sums them up, adds a bias term, and passes the result through an **activation function**:

output = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)

where xᵢ are inputs, wᵢ are weights, b is the bias, and f is the activation function.

A single neuron can learn a linear decision boundary. For instance, it can learn the logical AND or OR functions. However, a single neuron cannot learn non-linear functions like XOR — this limitation motivated the development of multi-layer networks.

## Activation Functions

The activation function introduces **non-linearity**, which is essential for learning complex patterns.

- **Step function**: outputs 0 or 1 based on a threshold. Simple but not differentiable, making gradient-based learning impossible.
- **Sigmoid**: σ(x) = 1 / (1 + e^(-x)). Outputs values between 0 and 1, interpretable as probabilities. Smooth and differentiable but suffers from the **vanishing gradient problem** for very large or very small inputs.
- **ReLU** (Rectified Linear Unit): f(x) = max(0, x). Simple, fast to compute, and alleviates the vanishing gradient problem for positive inputs. The most widely used activation in modern deep learning.
- **Softmax**: converts a vector of values into a probability distribution. Used in the output layer for multi-class classification: softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)

## Network Architecture

A neural network is organized into layers:

- **Input layer**: receives the raw features (one node per feature)
- **Hidden layers**: intermediate layers that learn internal representations. Each neuron in a hidden layer receives input from all neurons in the previous layer and sends output to all neurons in the next layer (in a **fully connected** or **dense** architecture)
- **Output layer**: produces the final prediction. For binary classification, a single sigmoid neuron. For multi-class classification, one softmax neuron per class. For regression, a single linear neuron.

The **depth** of a network (number of hidden layers) determines its capacity to learn hierarchical features. The **width** (number of neurons per layer) determines the richness of representations at each level.

## Training Neural Networks

### Forward Propagation

During **forward propagation**, input data flows through the network layer by layer. Each neuron computes its weighted sum, applies its activation function, and passes the result to the next layer. The final output is compared to the true label using a **loss function**.

### Loss Functions

- **Mean Squared Error (MSE)**: used for regression. MSE = (1/n) Σ(yᵢ - ŷᵢ)²
- **Binary Cross-Entropy**: used for binary classification. Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
- **Categorical Cross-Entropy**: used for multi-class classification. Loss = -Σᵢ yᵢ × log(ŷᵢ)

### Backpropagation

**Backpropagation** is the algorithm that computes how much each weight contributed to the error. Using the **chain rule** from calculus, it calculates the gradient of the loss function with respect to every weight in the network, working backward from the output layer to the input layer.

For a weight wᵢⱼ connecting neuron i to neuron j:

∂Loss/∂wᵢⱼ = ∂Loss/∂aⱼ × ∂aⱼ/∂zⱼ × ∂zⱼ/∂wᵢⱼ

where aⱼ is the activation of neuron j and zⱼ is its pre-activation (weighted sum).

### Gradient Descent

Once gradients are computed, **gradient descent** updates each weight:

w := w - α × ∂Loss/∂w

Variants include:
- **Batch gradient descent**: computes gradients over the entire training set. Stable but slow for large datasets.
- **Stochastic Gradient Descent (SGD)**: updates after each individual example. Noisy but fast and can escape local minima.
- **Mini-batch gradient descent**: a compromise — computes gradients over small batches (typically 32-256 examples). This is the most common approach in practice, balancing stability and speed.

Advanced optimizers like **Adam** (Adaptive Moment Estimation) adapt the learning rate for each parameter individually and maintain momentum terms, often converging faster than vanilla SGD.

## Preventing Overfitting

Deep neural networks have many parameters and can easily overfit. Key regularization techniques:

- **Dropout**: during training, randomly set a fraction of neuron outputs to zero. This prevents neurons from co-adapting and forces the network to learn more robust features. At test time, all neurons are active but outputs are scaled.
- **Early stopping**: monitor validation loss during training and stop when it begins to increase, even if training loss continues to decrease.
- **Weight decay** (L2 regularization): add a penalty proportional to the squared magnitude of weights to the loss function.
- **Data augmentation**: artificially increase training set size by applying transformations (rotations, flips, crops for images; synonym replacement for text).

## Convolutional Neural Networks (CNNs)

CNNs are specialized architectures designed for spatial data like images. Instead of fully connecting every neuron to every input, CNNs use **convolutional layers** with local connectivity.

### Convolutional Layers

A convolutional layer applies a set of small **filters** (kernels) that slide across the input. Each filter detects a specific local pattern — edges, textures, shapes — regardless of where it appears in the input. This **parameter sharing** dramatically reduces the number of weights compared to fully connected layers.

For a 2D input (like an image), each filter computes a dot product with a small patch of the input at each position, producing a **feature map**. Multiple filters produce multiple feature maps, each sensitive to different patterns.

### Pooling Layers

**Pooling layers** reduce the spatial dimensions of feature maps, creating invariance to small translations and reducing computation. **Max pooling** takes the maximum value in each local region, while **average pooling** takes the mean.

A typical CNN architecture alternates convolutional and pooling layers, progressively extracting higher-level features:
1. Early layers detect low-level features (edges, corners)
2. Middle layers detect mid-level features (textures, parts)
3. Deep layers detect high-level features (objects, faces)
4. Final fully connected layers perform classification

### Practical CNN Architectures

Landmark architectures include LeNet (handwritten digit recognition), AlexNet (which popularized deep learning for image classification), VGGNet (using very small 3×3 filters throughout), ResNet (introducing skip connections to train very deep networks), and more recently EfficientNet and Vision Transformers.

## Recurrent Neural Networks (RNNs)

While CNNs excel at spatial data, **Recurrent Neural Networks** are designed for sequential data — text, speech, time series, and any input where order matters.

### Architecture

An RNN maintains a **hidden state** that acts as memory. At each timestep, the network receives the current input and the previous hidden state, and produces a new hidden state and optionally an output:

hₜ = f(W_xh × xₜ + W_hh × hₜ₋₁ + b)

This recurrence allows information to flow across timesteps, enabling the network to maintain context.

### Vanishing Gradient Problem

Standard RNNs struggle to learn long-range dependencies because gradients can vanish (shrink to near-zero) or explode (grow extremely large) as they are propagated back through many timesteps.

### LSTM and GRU

**Long Short-Term Memory (LSTM)** networks solve the vanishing gradient problem by introducing a **cell state** that can carry information across many timesteps, regulated by three gates:
- **Forget gate**: decides what information to discard from the cell state
- **Input gate**: decides what new information to store
- **Output gate**: decides what to output from the cell state

**Gated Recurrent Units (GRUs)** are a simpler variant with two gates (reset and update) that often achieve comparable performance with fewer parameters.

## Key Takeaways

Neural networks learn hierarchical representations from data through layers of interconnected neurons. Backpropagation and gradient descent enable efficient training. CNNs specialize in spatial patterns through local connectivity and weight sharing, while RNNs handle sequential data through recurrent connections. Modern deep learning combines these building blocks with regularization techniques and advanced optimizers to achieve remarkable performance across a wide range of tasks.
