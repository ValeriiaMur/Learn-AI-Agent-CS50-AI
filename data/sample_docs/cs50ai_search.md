# Search Algorithms in Artificial Intelligence

## Overview

Search is one of the foundational concepts in artificial intelligence. At its core, an AI agent often needs to find a path from an initial state to a goal state through a space of possible configurations. Whether navigating a maze, playing a game, or planning a sequence of actions, search algorithms provide the backbone of intelligent decision-making.

A **search problem** is formally defined by an initial state, a set of actions available in each state, a transition model describing how actions change states, a goal test to check whether a given state is the goal, and a path cost function that assigns a numerical cost to each path.

## Uninformed Search

Uninformed (or blind) search strategies have no additional information about states beyond the problem definition. They explore the state space systematically without guidance.

### Breadth-First Search (BFS)

Breadth-First Search explores all nodes at the current depth before moving to nodes at the next depth level. It uses a **queue** (FIFO data structure) as its frontier.

BFS is **complete** — it will always find a solution if one exists — and **optimal** when all step costs are equal, since it finds the shallowest goal node first. However, it can be memory-intensive because it must store all nodes at the current depth level. The time and space complexity are both O(b^d), where b is the branching factor and d is the depth of the shallowest solution.

```python
def bfs(initial_state, goal_test, get_neighbors):
    frontier = Queue()
    frontier.add(initial_state)
    explored = set()

    while not frontier.empty():
        state = frontier.remove()
        if goal_test(state):
            return state
        explored.add(state)
        for neighbor in get_neighbors(state):
            if neighbor not in explored and neighbor not in frontier:
                frontier.add(neighbor)
    return None  # No solution
```

### Depth-First Search (DFS)

Depth-First Search explores as far as possible along each branch before backtracking. It uses a **stack** (LIFO data structure) as its frontier.

DFS uses much less memory than BFS — O(b × m) where m is the maximum depth — but it is neither complete (it can get stuck in infinite branches) nor optimal (it may find a deeper solution before a shallower one). DFS is useful when memory is a constraint or when any solution is acceptable regardless of optimality.

## Informed Search

Informed search strategies use problem-specific knowledge — typically in the form of a **heuristic function** h(n) — to guide the search toward the goal more efficiently.

### Greedy Best-First Search

Greedy best-first search expands the node that appears to be closest to the goal based on the heuristic function h(n). It selects the node with the lowest heuristic value from the frontier. While often fast in practice, it is neither complete nor optimal because it can be misled by heuristics into dead ends or suboptimal paths.

### A* Search

A* search combines the strengths of uniform-cost search and greedy best-first search. It evaluates nodes using the function:

**f(n) = g(n) + h(n)**

where g(n) is the actual cost to reach node n from the start, and h(n) is the estimated cost from n to the goal.

A* is both **complete** and **optimal** provided the heuristic is:
- **Admissible**: h(n) never overestimates the true cost to reach the goal
- **Consistent** (monotonic): for every node n and successor n', h(n) ≤ cost(n, n') + h(n')

The power of A* lies in choosing good heuristics. For grid-based pathfinding, **Manhattan distance** (sum of absolute differences in x and y coordinates) is a common admissible heuristic. The better the heuristic approximates actual cost, the fewer nodes A* needs to explore.

```python
def a_star(initial_state, goal_test, get_neighbors, h):
    frontier = PriorityQueue()
    frontier.add(initial_state, priority=h(initial_state))
    came_from = {initial_state: None}
    cost_so_far = {initial_state: 0}

    while not frontier.empty():
        current = frontier.remove()
        if goal_test(current):
            return reconstruct_path(came_from, current)

        for neighbor, step_cost in get_neighbors(current):
            new_cost = cost_so_far[current] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + h(neighbor)
                frontier.add(neighbor, priority=priority)
                came_from[neighbor] = current
    return None
```

## Adversarial Search

In many AI applications — especially games — the agent must contend with an **adversary** who is actively working against it. Adversarial search addresses this by modeling the interaction between competing agents.

### Minimax

The Minimax algorithm is designed for two-player, zero-sum games (where one player's gain is the other's loss). It models the game as a tree where:
- The **maximizing player** (e.g., "X" in tic-tac-toe) chooses the move with the highest value
- The **minimizing player** (e.g., "O") chooses the move with the lowest value

Each player plays optimally, assuming the opponent also plays optimally. Terminal states are assigned utility values (e.g., +1 for a win, -1 for a loss, 0 for a draw), and these values propagate upward through the tree.

```python
def minimax(state, is_maximizing):
    if is_terminal(state):
        return utility(state)

    if is_maximizing:
        best = -infinity
        for action in actions(state):
            value = minimax(result(state, action), False)
            best = max(best, value)
        return best
    else:
        best = +infinity
        for action in actions(state):
            value = minimax(result(state, action), True)
            best = min(best, value)
        return best
```

### Alpha-Beta Pruning

The minimax algorithm can be expensive because it explores every possible game state. **Alpha-beta pruning** optimizes minimax by skipping branches that cannot possibly influence the final decision.

It maintains two values:
- **Alpha**: the best value the maximizer can guarantee (starts at −∞)
- **Beta**: the best value the minimizer can guarantee (starts at +∞)

When alpha ≥ beta, the current branch can be pruned because the opponent would never allow this position to be reached. In the best case, alpha-beta pruning reduces the effective branching factor from b to √b, allowing search to go twice as deep in the same time.

### Depth-Limited Minimax

For complex games like chess, exploring the entire game tree is infeasible. **Depth-limited minimax** stops the search at a specified depth and uses an **evaluation function** to estimate the utility of non-terminal states. This evaluation function encodes domain knowledge — for chess, it might consider material advantage, king safety, pawn structure, and piece mobility.

## Key Takeaways

Search algorithms form the foundation of AI problem-solving. BFS and DFS provide systematic exploration, A* adds intelligence through heuristics, and minimax handles adversarial scenarios. The choice of algorithm depends on the problem structure: whether the search space is finite or infinite, whether costs are uniform, whether there is an adversary, and what computational resources are available.
