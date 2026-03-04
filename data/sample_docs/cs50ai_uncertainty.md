# Uncertainty in Artificial Intelligence

## Overview

Real-world AI systems rarely have access to complete, certain information. Sensors are noisy, observations are incomplete, and outcomes are stochastic. Rather than dealing exclusively with facts that are definitely true or false, AI needs the ability to reason under uncertainty. Probability theory provides the mathematical framework for representing and manipulating uncertain beliefs.

## Probability Fundamentals

A **random variable** takes on values from a defined domain. For example, a Weather variable might have values {sunny, rainy, cloudy}. A **probability distribution** assigns a probability to each possible value, with all probabilities summing to 1.

Key concepts include:
- **Unconditional probability** P(A): the probability of event A regardless of any other information
- **Conditional probability** P(A|B): the probability of A given that B is known to be true, defined as P(A ∧ B) / P(B)
- **Joint probability** P(A, B): the probability that both A and B occur simultaneously

### Bayes' Rule

Bayes' Rule is one of the most important formulas in AI. It allows us to reverse conditional probabilities:

**P(A|B) = P(B|A) × P(A) / P(B)**

This is powerful because often we know P(B|A) (the likelihood) but want P(A|B) (the posterior). For example, given the probability of symptoms given a disease (likelihood), Bayes' Rule lets us compute the probability of the disease given observed symptoms (diagnosis).

The denominator P(B) acts as a normalizing constant and can be computed as:
P(B) = P(B|A) × P(A) + P(B|¬A) × P(¬A)

### Independence and Conditional Independence

Two events A and B are **independent** if knowing one gives no information about the other: P(A|B) = P(A). They are **conditionally independent** given C if P(A|B,C) = P(A|C). Conditional independence is crucial because it allows us to simplify complex joint distributions, making inference tractable.

## Bayesian Networks

A **Bayesian Network** is a directed acyclic graph (DAG) where each node represents a random variable, directed edges represent direct probabilistic dependencies, and each node has a conditional probability table (CPT) specifying the probability of each value given its parents.

For example, a simple weather model might have:
- Rain → Traffic (rain causes more traffic)
- Rain → Wet Grass (rain makes grass wet)
- Sprinkler → Wet Grass (sprinklers also make grass wet)

The power of Bayesian Networks is that the full joint distribution can be factored as a product of conditional probabilities:

P(X₁, X₂, ..., Xₙ) = ∏ P(Xᵢ | Parents(Xᵢ))

This factored representation is much more compact than storing the full joint distribution table. A network with n binary variables needs far fewer parameters than the 2^n entries in the full joint.

### Inference in Bayesian Networks

Given observed evidence, we want to compute the probability of query variables. This involves summing over (marginalizing) all unobserved variables:

P(Query | Evidence) = α × Σ_hidden P(Query, hidden, Evidence)

where α is a normalization constant.

**Exact inference** computes these sums directly but can be exponential in the number of hidden variables. For larger networks, **approximate inference** techniques are used.

## Markov Models

When reasoning about sequences of events over time, **Markov Models** capture temporal dependencies.

### Markov Chains

A **Markov Chain** models a sequence of states where the probability of the next state depends only on the current state — this is the **Markov property** (memorylessness):

P(Xₜ | X₀, X₁, ..., Xₜ₋₁) = P(Xₜ | Xₜ₋₁)

A Markov Chain is defined by a **transition model** — a matrix where entry (i, j) gives the probability of moving from state i to state j. For example, today's weather might depend only on yesterday's weather, not on the entire weather history.

### Hidden Markov Models (HMMs)

Often the true state of the world is not directly observable. A **Hidden Markov Model** extends Markov Chains by distinguishing between hidden states (which we cannot observe directly) and observations (which we can see).

An HMM has:
- A **transition model**: P(Xₜ | Xₜ₋₁) — how hidden states evolve
- A **sensor model** (emission model): P(Eₜ | Xₜ) — the probability of an observation given the hidden state

Key tasks in HMMs:
- **Filtering**: computing P(Xₜ | E₁, ..., Eₜ) — the current state given all observations so far
- **Prediction**: computing P(Xₜ₊ₖ | E₁, ..., Eₜ) — future states
- **Smoothing**: computing P(Xₖ | E₁, ..., Eₜ) for k < t — past states given all evidence
- **Most likely explanation**: finding the most probable sequence of hidden states (the Viterbi algorithm)

## Sampling Methods

When exact inference is too expensive, **sampling** provides approximate answers by generating random samples from the probability distribution.

### Direct Sampling

Generate samples according to the Bayesian network's probability distributions, starting from root nodes (no parents) and proceeding to child nodes. Count how many samples match the query to estimate probabilities.

### Rejection Sampling

To estimate P(Query | Evidence), generate samples from the full network, reject those that don't match the evidence, and count query matches among the remaining samples. This is simple but inefficient when evidence is unlikely — most samples get rejected.

### Likelihood Weighting

An improvement over rejection sampling: fix the evidence variables to their observed values and weight each sample by the likelihood of the evidence given the sampled values of other variables. This ensures no samples are wasted, making estimation more efficient.

```python
def likelihood_weighting(network, query, evidence, n_samples):
    counts = defaultdict(float)
    for _ in range(n_samples):
        sample, weight = weighted_sample(network, evidence)
        counts[sample[query]] += weight
    # Normalize
    total = sum(counts.values())
    return {val: count / total for val, count in counts.items()}
```

## Key Takeaways

Probability provides the mathematical language for reasoning under uncertainty. Bayes' Rule lets us update beliefs given new evidence. Bayesian Networks compactly represent complex joint distributions. Markov Models handle temporal sequences, and Hidden Markov Models deal with partially observable states. When exact computation is infeasible, sampling methods provide practical approximations.
