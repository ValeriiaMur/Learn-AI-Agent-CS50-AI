# Introduction to Neural Networks

## What is a Neural Network?

A neural network is a machine learning model inspired by the structure of biological brains. At its core, it's a function that takes inputs, processes them through layers of mathematical operations, and produces outputs. The "learning" happens by adjusting the internal parameters (weights and biases) to minimize prediction errors.

Think of it like a factory assembly line: raw materials (input data) enter, pass through multiple processing stations (layers), and a finished product (prediction) comes out the other end. Each station transforms the materials in a specific way, and the factory gets better over time as workers (weights) learn the best techniques.

## The Building Block: A Single Neuron

A single artificial neuron (also called a perceptron) performs three operations:
1. **Weighted Sum**: Multiply each input by a weight and sum them up: z = w1*x1 + w2*x2 + ... + wn*xn + b
2. **Bias Addition**: Add a bias term (b) that shifts the output
3. **Activation**: Pass the result through an activation function: output = f(z)

The weights determine how important each input is. The bias allows the neuron to shift its activation threshold. The activation function introduces non-linearity.

## Network Architecture

Neural networks are organized in layers:

### Input Layer
The input layer receives raw data. Each neuron represents one feature of the input. For an image of 28x28 pixels, the input layer would have 784 neurons (one per pixel).

### Hidden Layers
Hidden layers perform the actual computation. "Deep" learning refers to networks with many hidden layers. Each layer learns increasingly abstract representations: early layers might detect edges, middle layers detect shapes, and later layers detect objects.

### Output Layer
The output layer produces the final prediction. For binary classification, it typically has one neuron with sigmoid activation. For multi-class classification, it has one neuron per class with softmax activation. For regression, it has one neuron with no activation (linear output).

## How Neural Networks Learn

### Forward Pass
Data flows forward through the network, layer by layer, producing a prediction. Each layer transforms its input using weights, biases, and activation functions.

### Loss Calculation
The loss function compares the prediction to the actual target value. Higher loss means worse predictions. Common choices: MSE for regression, cross-entropy for classification.

### Backward Pass (Backpropagation)
The network computes how much each weight contributed to the error using the chain rule of calculus. Gradients flow backward from the output layer to the input layer.

### Weight Update
An optimizer (like SGD, Adam, or RMSprop) uses the gradients to update each weight in a direction that reduces the loss. The learning rate controls how big each update step is.

This cycle (forward → loss → backward → update) repeats for many iterations (epochs) until the model converges.

## Common Challenges

### Vanishing Gradients
In deep networks, gradients can become extremely small as they propagate backward, causing early layers to learn very slowly. Solutions include ReLU activation, residual connections (skip connections), and careful initialization.

### Exploding Gradients
The opposite problem — gradients become extremely large, causing unstable training. Solutions include gradient clipping, batch normalization, and LSTM/GRU architectures for sequential data.

### Choosing the Right Architecture
There's no universal architecture. CNNs work best for images, RNNs/Transformers for sequences, and feedforward networks for tabular data. Architecture search and experimentation are key skills.

## Practical Tips for Training Neural Networks

1. **Start simple**: Begin with a small network and increase complexity only if needed
2. **Normalize your data**: Scale inputs to similar ranges (e.g., 0-1 or mean 0, std 1)
3. **Use a validation set**: Monitor performance on held-out data to detect overfitting
4. **Learning rate matters most**: Try values like 0.001, 0.01, 0.1 and see what works
5. **Batch size tradeoff**: Smaller batches add noise (regularization) but train slower
6. **Save checkpoints**: Training can crash — save model weights periodically
7. **Visualize everything**: Plot loss curves, weight distributions, and predictions

## From Neural Networks to Deep Learning

Deep learning is simply neural networks with many layers. Key breakthroughs that enabled deep learning include better activation functions (ReLU), GPU computing, large datasets, and architectural innovations like residual networks (ResNets), attention mechanisms, and transformers.

The transformer architecture, introduced in 2017, revolutionized NLP and is now the foundation of large language models (LLMs) like GPT, BERT, and Claude. Transformers use self-attention to process all input tokens in parallel, overcoming the sequential bottleneck of RNNs.
