# Machine Learning Glossary

## Supervised Learning
Supervised learning is a type of machine learning where the model learns from labeled training data. Each training example consists of an input (features) and a desired output (label). The model learns to map inputs to outputs, then generalizes to make predictions on unseen data. Common algorithms include linear regression, logistic regression, decision trees, random forests, and support vector machines.

## Unsupervised Learning
Unsupervised learning works with unlabeled data. The model tries to find hidden patterns, structures, or groupings in the data without being told what to look for. Key techniques include clustering (K-means, DBSCAN, hierarchical clustering), dimensionality reduction (PCA, t-SNE, UMAP), and anomaly detection.

## Reinforcement Learning
Reinforcement learning (RL) is a paradigm where an agent learns by interacting with an environment. The agent takes actions, receives rewards or penalties, and learns a policy that maximizes cumulative reward over time. Key concepts include states, actions, rewards, policies, value functions, and the exploration-exploitation tradeoff. Famous algorithms include Q-learning, SARSA, policy gradient methods, and actor-critic methods.

## Neural Networks
A neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes (neurons) that process information. Each connection has a weight that is adjusted during training. A basic feedforward network has an input layer, one or more hidden layers, and an output layer. Each neuron applies a weighted sum of its inputs, adds a bias, and passes the result through an activation function.

## Backpropagation
Backpropagation is the algorithm used to train neural networks. It computes the gradient of the loss function with respect to each weight by applying the chain rule of calculus, propagating error backwards from the output layer to the input layer. These gradients tell us how to adjust each weight to reduce the loss. Combined with an optimizer like SGD or Adam, backpropagation enables the network to learn from data.

## Activation Functions
Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Without activation functions, a neural network would be equivalent to a linear model regardless of depth. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, tanh, and softmax. ReLU is the most widely used due to its simplicity and effectiveness at mitigating the vanishing gradient problem.

## Loss Functions
A loss function measures how far the model's predictions are from the actual values. It provides the signal that drives learning during training. Common loss functions include Mean Squared Error (MSE) for regression, Cross-Entropy Loss for classification, and Binary Cross-Entropy for binary classification. The choice of loss function depends on the task and affects how the model optimizes its parameters.

## Gradient Descent
Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize the loss function. It computes the gradient (direction of steepest increase) and takes a step in the opposite direction. Variants include batch gradient descent (uses all data), stochastic gradient descent (SGD, uses one sample), and mini-batch gradient descent (uses a subset). The learning rate controls step size and is a critical hyperparameter.

## Overfitting and Underfitting
Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, leading to poor generalization on new data. Signs include high training accuracy but low validation accuracy. Underfitting occurs when the model is too simple to capture the underlying patterns. Remedies for overfitting include regularization (L1, L2), dropout, early stopping, data augmentation, and using more training data.

## Convolutional Neural Networks (CNNs)
CNNs are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers that apply learnable filters to detect local patterns like edges, textures, and shapes. Key components include convolutional layers, pooling layers (which reduce spatial dimensions), and fully connected layers. CNNs leverage parameter sharing and local connectivity to efficiently process high-dimensional inputs.

## Recurrent Neural Networks (RNNs)
RNNs are designed for sequential data like text, time series, and audio. They maintain a hidden state that captures information from previous time steps, allowing them to model temporal dependencies. Standard RNNs suffer from vanishing and exploding gradients over long sequences. LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) architectures address this with gating mechanisms that control information flow.

## Regularization
Regularization techniques prevent overfitting by constraining the model's complexity. L1 regularization (Lasso) adds the absolute value of weights to the loss, encouraging sparsity. L2 regularization (Ridge) adds the squared weights, encouraging small weights. Dropout randomly deactivates neurons during training, forcing the network to be robust. Batch normalization normalizes layer inputs, improving training stability and acting as a mild regularizer.

## Transfer Learning
Transfer learning leverages knowledge from a model trained on one task to improve performance on a related task. Instead of training from scratch, you start with a pre-trained model and fine-tune it on your specific dataset. This is especially powerful when you have limited data. Common in computer vision (using pre-trained ImageNet models) and NLP (using pre-trained language models like BERT or GPT).
