# Machine Learning

## Overview

Machine learning is the subfield of AI concerned with algorithms that improve their performance through experience. Rather than being explicitly programmed with rules, a machine learning system learns patterns from data. This is especially powerful for tasks where writing explicit rules is impractical — recognizing faces, filtering spam, recommending products, or predicting stock prices.

Machine learning tasks generally fall into three categories: **supervised learning** (learning from labeled examples), **unsupervised learning** (discovering structure in unlabeled data), and **reinforcement learning** (learning through trial, error, and rewards).

## Supervised Learning

In supervised learning, the algorithm receives a training set of input-output pairs and learns a function that maps inputs to outputs. The goal is to generalize — to make accurate predictions on new, unseen inputs.

### Classification

Classification assigns inputs to discrete categories. Given features describing an input, the classifier outputs which class it belongs to.

#### k-Nearest Neighbors (kNN)

kNN is one of the simplest classification algorithms. To classify a new point, it finds the k closest training examples (by some distance metric, typically Euclidean distance) and assigns the most common class among those neighbors.

- When k=1, the classifier assigns the class of the single nearest neighbor
- Larger k values smooth out the decision boundary but may blur class boundaries
- The choice of k is a hyperparameter that can be tuned via cross-validation
- kNN is a **lazy learner** — it stores all training data and defers computation to prediction time

```python
def knn_classify(training_data, query_point, k):
    distances = [(euclidean(query_point, x), label)
                 for x, label in training_data]
    distances.sort(key=lambda d: d[0])
    k_nearest = distances[:k]
    labels = [label for _, label in k_nearest]
    return most_common(labels)
```

#### Support Vector Machines (SVMs)

SVMs find the **maximum-margin hyperplane** that separates classes. The hyperplane is positioned to maximize the distance to the nearest training points from each class (called **support vectors**).

For linearly separable data, the SVM finds a linear boundary w · x + b = 0. The margin is 2/||w||, and maximizing the margin improves generalization. For non-linearly separable data, SVMs use the **kernel trick** — mapping data into a higher-dimensional space where a linear separator exists. Common kernels include polynomial and radial basis function (RBF) kernels.

SVMs are effective in high-dimensional spaces and work well when the number of dimensions exceeds the number of samples.

### Regression

Regression predicts continuous values rather than discrete categories.

#### Linear Regression

Linear regression fits a line (or hyperplane) to the data by minimizing the sum of squared errors between predictions and actual values. For a single feature:

ŷ = w₁x + w₀

The weights are chosen to minimize the **cost function**:

Cost = (1/n) Σ (yᵢ - ŷᵢ)²

**Gradient descent** iteratively adjusts the weights in the direction that reduces the cost:

w := w - α × ∂Cost/∂w

where α is the **learning rate** — a hyperparameter controlling step size.

#### Regularization

To prevent **overfitting** (memorizing training data rather than learning general patterns), regularization adds a penalty for model complexity:

- **L1 regularization** (Lasso): adds λ × Σ|wᵢ| — encourages sparsity by driving some weights to zero
- **L2 regularization** (Ridge): adds λ × Σwᵢ² — discourages large weights, producing smoother models

The regularization parameter λ controls the trade-off between fitting the training data and keeping the model simple.

## Evaluating Models

### Loss Functions

A **loss function** measures how far predictions are from actual values:
- **0-1 loss**: counts misclassifications (for classification)
- **Mean squared error (MSE)**: average of squared differences (for regression)
- **Cross-entropy loss**: measures divergence between predicted and actual probability distributions

### Overfitting and Underfitting

- **Overfitting**: the model performs well on training data but poorly on new data — it has memorized noise rather than learned patterns
- **Underfitting**: the model is too simple to capture the underlying patterns in the data

The goal is to find the right balance of model complexity — the **bias-variance tradeoff**. High bias (underfitting) means the model makes strong assumptions. High variance (overfitting) means the model is too sensitive to training data.

### Train-Test Split and Cross-Validation

Never evaluate a model on the same data it was trained on. Standard practice:
- Split data into **training set** (typically 70-80%) and **test set** (20-30%)
- Train on the training set, evaluate on the test set
- **k-fold cross-validation**: divide data into k parts, train on k-1 parts, evaluate on the remaining part, rotate and repeat. This gives a more robust performance estimate.

## Unsupervised Learning

In unsupervised learning, there are no labels — the algorithm must discover structure in the data on its own.

### k-Means Clustering

k-Means groups data into k clusters by iteratively:
1. **Assigning** each data point to the nearest cluster centroid
2. **Updating** each centroid to be the mean of all points assigned to it
3. **Repeating** until assignments stabilize (convergence)

```python
def k_means(data, k, max_iterations=100):
    centroids = random.sample(data, k)
    for _ in range(max_iterations):
        # Assignment step
        clusters = [[] for _ in range(k)]
        for point in data:
            nearest = argmin(distance(point, c) for c in centroids)
            clusters[nearest].append(point)
        # Update step
        new_centroids = [mean(cluster) for cluster in clusters]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters
```

The choice of k is critical and can be guided by metrics like the **elbow method** (plotting within-cluster sum of squares vs. k) or the **silhouette score** (measuring how similar a point is to its own cluster vs. other clusters).

## Reinforcement Learning

In reinforcement learning, an **agent** learns by interacting with an **environment**. At each timestep, the agent observes a state, takes an action, receives a reward, and transitions to a new state. The goal is to learn a **policy** (a mapping from states to actions) that maximizes cumulative reward over time.

### Markov Decision Processes (MDPs)

Reinforcement learning problems are formalized as MDPs:
- **States** S: the possible configurations of the environment
- **Actions** A: the choices available to the agent
- **Transition model** P(s'|s, a): the probability of reaching state s' given state s and action a
- **Reward function** R(s, a, s'): the immediate reward for a transition
- **Discount factor** γ: a value between 0 and 1 that determines how much future rewards are worth compared to immediate rewards

### Q-Learning

Q-learning learns the value of taking action a in state s:

Q(s, a) ← Q(s, a) + α × [R + γ × max_a' Q(s', a') - Q(s, a)]

The agent updates its Q-values based on experienced rewards and its best estimate of future value. Over time, the Q-values converge to the optimal action-value function, and the agent can act greedily with respect to Q (always choosing the action with the highest Q-value).

The **exploration vs. exploitation** dilemma is central: the agent must balance trying new actions (exploration) to discover better strategies with choosing the best known action (exploitation) to maximize reward. The **epsilon-greedy** strategy explores with probability ε and exploits with probability 1-ε.

## Key Takeaways

Machine learning enables AI systems to learn from data rather than from explicit programming. Supervised learning handles labeled data through classification and regression, unsupervised learning discovers hidden structure, and reinforcement learning optimizes behavior through interaction. The key challenges — overfitting, choosing the right model complexity, and evaluating generalization — are addressed through regularization, cross-validation, and careful experimental design.
