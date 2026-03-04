# Optimization in Artificial Intelligence

## Overview

Many AI problems can be framed as optimization: finding the best solution from a set of possible solutions according to some objective function. Unlike search problems where we seek a path from start to goal, optimization problems focus on finding the state (or configuration) that maximizes or minimizes a given measure of quality.

## Local Search

Local search algorithms operate by maintaining a single current state and iteratively moving to a neighboring state. They don't track the path — only the current position matters. This makes them memory-efficient and suitable for problems where the path is irrelevant and only the final configuration matters.

### State-Space Landscape

We can visualize the optimization problem as a landscape where each state has a height proportional to its objective value. The goal is to find the highest peak (maximization) or the lowest valley (minimization). The landscape may contain:
- **Global maximum/minimum**: the best solution overall
- **Local maxima/minima**: solutions better than all neighbors but not globally optimal
- **Plateaus**: flat regions where all neighbors have the same value
- **Shoulders**: flat regions that eventually lead uphill

### Hill Climbing

Hill climbing is the most basic local search strategy. Starting from a random state, it repeatedly moves to the neighbor with the highest (or lowest) value, stopping when no neighbor improves the current state.

```python
def hill_climbing(problem):
    current = problem.random_state()
    while True:
        neighbor = best_neighbor(current, problem)
        if problem.value(neighbor) <= problem.value(current):
            return current  # Local maximum reached
        current = neighbor
```

Hill climbing is fast and simple but suffers from getting stuck at local optima, plateaus, and ridges. It only finds the global optimum if the landscape is convex (has a single peak).

### Hill Climbing Variants

Several modifications help hill climbing escape local optima:

- **Steepest-ascent hill climbing**: evaluates all neighbors and picks the best one (standard version)
- **Stochastic hill climbing**: randomly selects among uphill neighbors, with preference for steeper moves
- **Random-restart hill climbing**: runs hill climbing multiple times from random starting positions and keeps the best result. With enough restarts, this is likely to find the global optimum
- **First-choice hill climbing**: generates random neighbors one at a time and accepts the first one that improves the current state. Useful when there are many neighbors

### Simulated Annealing

**Simulated annealing** draws inspiration from the metallurgical process of annealing, where metals are heated and slowly cooled to reach a low-energy crystalline state. The algorithm introduces controlled randomness that decreases over time.

At each step, a random neighbor is selected. If the neighbor is better, it is always accepted. If it is worse, it is accepted with a probability that depends on how much worse it is and on a "temperature" parameter that decreases over time:

P(accept worse) = e^(ΔE / T)

where ΔE is the change in value and T is the current temperature.

Early in the search (high temperature), the algorithm freely explores and can escape local optima. As the temperature cools, it becomes increasingly greedy and converges toward an optimum. With an appropriate cooling schedule, simulated annealing is guaranteed to find the global optimum in the limit.

```python
def simulated_annealing(problem, schedule):
    current = problem.random_state()
    for t in range(1, MAX_STEPS):
        T = schedule(t)
        if T == 0:
            return current
        neighbor = random_neighbor(current, problem)
        delta_e = problem.value(neighbor) - problem.value(current)
        if delta_e > 0 or random() < exp(delta_e / T):
            current = neighbor
    return current
```

## Linear Programming

**Linear programming** optimizes a linear objective function subject to linear constraints (equalities and inequalities). Problems are expressed as:

Minimize c^T × x, subject to A × x ≤ b and x ≥ 0

where c is the cost vector, x is the vector of decision variables, A is the constraint matrix, and b is the constraint vector.

Linear programming has efficient polynomial-time algorithms (like the interior point method) and is widely used in resource allocation, scheduling, logistics, and economics. The **Simplex algorithm**, while worst-case exponential, is extremely efficient in practice.

## Constraint Satisfaction Problems (CSPs)

A **Constraint Satisfaction Problem** defines a set of variables, each with a domain of possible values, and a set of constraints that specify which combinations of values are allowed. The goal is to find an assignment of values to all variables that satisfies every constraint.

### Formal Definition

- **Variables**: X₁, X₂, ..., Xₙ
- **Domains**: D₁, D₂, ..., Dₙ (possible values for each variable)
- **Constraints**: restrictions on combinations of variable values

### Types of Constraints

- **Unary constraints**: involve a single variable (e.g., X₁ ≠ Monday)
- **Binary constraints**: involve two variables (e.g., X₁ ≠ X₂)
- **Higher-order constraints**: involve three or more variables

A classic CSP is **graph coloring**: given a map of regions, assign colors to each region such that no two adjacent regions share the same color. The four-color theorem guarantees a solution with at most four colors for any planar graph.

### Node Consistency and Arc Consistency

**Node consistency** ensures every value in a variable's domain satisfies all unary constraints on that variable.

**Arc consistency** is a stronger property: variable X is arc-consistent with respect to variable Y if for every value in X's domain, there exists at least one value in Y's domain that satisfies the constraint between X and Y. The **AC-3 algorithm** enforces arc consistency by maintaining a queue of arcs to check:

```python
def ac3(csp):
    queue = list(csp.all_arcs())
    while queue:
        (Xi, Xj) = queue.pop(0)
        if revise(csp, Xi, Xj):
            if len(csp.domain[Xi]) == 0:
                return False  # No solution
            for Xk in csp.neighbors(Xi) - {Xj}:
                queue.append((Xk, Xi))
    return True

def revise(csp, Xi, Xj):
    revised = False
    for x in csp.domain[Xi][:]:
        if not any(csp.satisfies(x, y, Xi, Xj) for y in csp.domain[Xj]):
            csp.domain[Xi].remove(x)
            revised = True
    return revised
```

### Backtracking Search

CSPs are solved using **backtracking search** — a systematic depth-first search that assigns one variable at a time and backtracks when a constraint is violated.

Key optimizations:
- **Minimum Remaining Values (MRV)**: choose the variable with the fewest legal values remaining — this is the variable most likely to cause a failure, so addressing it first prunes the search tree
- **Degree heuristic**: among tied MRV candidates, choose the variable involved in the most constraints with unassigned variables
- **Least Constraining Value (LCV)**: for the selected variable, try values that rule out the fewest options for neighboring variables — this maximizes flexibility for future assignments
- **Maintaining Arc Consistency (MAC)**: after each assignment, enforce arc consistency to prune domains early

## Key Takeaways

Optimization pervades AI: hill climbing and simulated annealing solve continuous and combinatorial problems through iterative improvement, linear programming handles problems with linear structure, and CSPs model problems with discrete variables and constraints. The key challenge across all optimization is avoiding local optima and efficiently navigating large solution spaces. Techniques like random restarts, annealing schedules, and constraint propagation help address these challenges.
