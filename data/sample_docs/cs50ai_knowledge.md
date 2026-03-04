# Knowledge Representation and Reasoning

## Overview

For an AI to reason about the world, it needs a way to represent what it knows and draw conclusions from that knowledge. Knowledge representation is the study of how to encode facts, rules, and relationships in a form that a computer can manipulate. Logical reasoning then allows the AI to derive new knowledge from existing facts using formal rules of inference.

## Propositional Logic

Propositional logic is the simplest form of formal logic. It deals with **propositions** — statements that are either true or false. Propositions are represented by symbols (e.g., P, Q, R) and combined using logical connectives.

### Logical Connectives

- **NOT (¬)**: Negation. ¬P is true when P is false.
- **AND (∧)**: Conjunction. P ∧ Q is true only when both P and Q are true.
- **OR (∨)**: Disjunction. P ∨ Q is true when at least one of P or Q is true.
- **IMPLICATION (→)**: P → Q is false only when P is true and Q is false. It means "if P, then Q."
- **BICONDITIONAL (↔)**: P ↔ Q is true when P and Q have the same truth value.

### Knowledge Base and Entailment

A **knowledge base** (KB) is a set of sentences (propositions and their combinations) that the AI knows to be true. **Entailment** (KB ⊨ α) means that in every model (truth assignment) where the knowledge base is true, sentence α is also true. In other words, α follows logically from the KB.

The goal of a reasoning system is to determine whether the KB entails some query sentence — this process is called **inference**.

## Inference Methods

### Model Checking

The most straightforward inference method is **model checking**: enumerate every possible assignment of truth values to all propositional symbols and check whether the query is true in every model where the KB is true.

```python
def model_check(knowledge, query, symbols, model):
    if not symbols:
        if knowledge.evaluate(model):
            return query.evaluate(model)
        return True  # KB is false, so entailment holds vacuously

    p = symbols[0]
    rest = symbols[1:]

    # Try both True and False for this symbol
    model_true = {**model, p: True}
    model_false = {**model, p: False}

    return (model_check(knowledge, query, rest, model_true) and
            model_check(knowledge, query, rest, model_false))
```

Model checking is **sound** (it only proves things that are true) and **complete** (it can prove everything that is true), but it is computationally expensive. With n symbols, there are 2^n possible models to check.

### Inference Rules

More efficient reasoning uses **inference rules** — patterns that produce new sentences from existing ones. Key rules include:

- **Modus Ponens**: From P → Q and P, conclude Q
- **And Elimination**: From P ∧ Q, conclude P (or Q)
- **Double Negation**: From ¬(¬P), conclude P
- **Implication Elimination**: P → Q is equivalent to ¬P ∨ Q
- **De Morgan's Laws**: ¬(P ∧ Q) ≡ ¬P ∨ ¬Q and ¬(P ∨ Q) ≡ ¬P ∧ ¬Q
- **Resolution**: From (P ∨ Q) and (¬P ∨ R), conclude (Q ∨ R)

### Resolution

Resolution is a particularly powerful inference rule. To prove that KB ⊨ α, we can use **proof by contradiction**: assume ¬α, add it to the KB, convert everything to **Conjunctive Normal Form** (CNF) — a conjunction of disjunctions — and repeatedly apply the resolution rule. If we derive the **empty clause** (a contradiction), then the original query α must be true.

CNF conversion follows these steps:
1. Eliminate biconditionals: P ↔ Q becomes (P → Q) ∧ (Q → P)
2. Eliminate implications: P → Q becomes ¬P ∨ Q
3. Push negations inward using De Morgan's laws
4. Distribute OR over AND to get clauses

## First-Order Logic

Propositional logic has limitations — it cannot easily express statements about objects and their relationships. **First-order logic** (FOL) extends propositional logic with:

- **Constants**: specific objects (e.g., Harry, Hagrid, Hogwarts)
- **Predicates**: properties or relations (e.g., Person(x), Teaches(x, y))
- **Quantifiers**:
  - **Universal (∀)**: "for all" — ∀x: Person(x) → Mortal(x) means "all persons are mortal"
  - **Existential (∃)**: "there exists" — ∃x: Person(x) ∧ Wizard(x) means "some person is a wizard"
- **Functions**: mappings from objects to objects (e.g., MotherOf(x))

First-order logic is far more expressive than propositional logic. Many AI knowledge representation tasks — from database querying to natural language understanding — rely on FOL or its fragments.

## Knowledge Engineering

Building a knowledge-based AI system involves **knowledge engineering**: the process of encoding domain expertise into formal logical statements. This requires:

1. **Identifying relevant propositions**: What facts matter for the problem?
2. **Encoding relationships**: How do facts relate to each other?
3. **Defining rules**: What inference rules capture the domain logic?

A classic example is encoding the rules of a puzzle like Clue (Cluedo), where the AI must deduce the murderer, weapon, and room from a set of constraints and observations. Each piece of evidence eliminates possibilities, and logical inference narrows down the solution.

## Applications

Knowledge representation and reasoning appear throughout AI: expert systems use rule-based knowledge bases to diagnose diseases or configure equipment, planning systems represent actions and their preconditions to find sequences of steps achieving a goal, and semantic web technologies use formal ontologies to enable machines to understand web content. The ability to represent knowledge formally and reason about it systematically remains one of the pillars of artificial intelligence.

## Key Takeaways

Propositional logic provides a foundation for representing and reasoning about boolean facts. First-order logic extends this to objects and relationships. Inference methods — from brute-force model checking to efficient resolution — allow AI systems to derive new knowledge from what they already know. The challenge lies in encoding knowledge accurately and performing inference efficiently as the knowledge base grows.
