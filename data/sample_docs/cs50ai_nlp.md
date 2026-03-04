# Natural Language Processing

## Overview

Natural Language Processing (NLP) is the subfield of AI focused on enabling computers to understand, interpret, and generate human language. Language is one of the most complex and nuanced forms of communication — it is ambiguous, context-dependent, and constantly evolving. NLP techniques have progressed from simple rule-based systems through statistical methods to modern deep learning approaches, culminating in the powerful language models we have today.

## Text Representation

Before a machine learning model can process text, words must be converted to numerical representations.

### Bag of Words

The simplest approach is the **bag of words** model: represent each document as a vector where each dimension corresponds to a word in the vocabulary, and the value is the word count (or binary presence). This ignores word order entirely — "dog bites man" and "man bites dog" have identical representations.

### TF-IDF

**Term Frequency–Inverse Document Frequency** improves on raw counts by weighting words based on their importance:

- **Term Frequency (TF)**: how often a word appears in a document
- **Inverse Document Frequency (IDF)**: log(N/df), where N is total documents and df is the number of documents containing the term

TF-IDF(t, d) = TF(t, d) × IDF(t)

Words that appear frequently in a specific document but rarely across documents get high TF-IDF scores, making them good discriminators. Common words like "the" get low scores.

### Word Embeddings

**Word embeddings** represent words as dense, low-dimensional vectors (typically 100-300 dimensions) where semantically similar words are close together in the vector space.

**Word2Vec** learns embeddings by training a neural network on a simple task: predicting a word from its context (Skip-gram) or predicting context from a word (CBOW). The trained weight matrix becomes the embedding.

Remarkable property: word embeddings capture semantic relationships through vector arithmetic. For example, the vector for "king" minus "man" plus "woman" yields a vector close to "queen."

**GloVe** (Global Vectors) produces similar embeddings by factorizing a co-occurrence matrix that counts how often words appear near each other across a corpus.

## Tokenization

**Tokenization** breaks text into units (tokens) for processing. Approaches include:

- **Word tokenization**: splitting on whitespace and punctuation. Simple but struggles with out-of-vocabulary words.
- **Subword tokenization**: breaking words into meaningful subunits. For example, "unhappiness" might become ["un", "happi", "ness"]. This handles rare and unseen words by composing them from known subword units.
- **Byte Pair Encoding (BPE)**: a common subword algorithm that starts with individual characters and iteratively merges the most frequent pairs. Used by GPT and many modern models.
- **SentencePiece**: another subword tokenizer that operates directly on raw text without pre-tokenization, treating the input as a stream of Unicode characters.

## Sequence Models for NLP

### RNNs for Text

Recurrent Neural Networks process text sequentially, maintaining a hidden state that accumulates information from each token. This makes them natural for tasks like sentiment analysis (classify the overall sentiment of a review) and language modeling (predict the next word given previous words).

However, standard RNNs struggle with long-range dependencies. **LSTM** and **GRU** variants address this through gating mechanisms that control information flow, as discussed in the Neural Networks chapter.

### Sequence-to-Sequence Models

For tasks where both input and output are sequences (like machine translation), **encoder-decoder** architectures process the input with an encoder RNN and generate the output with a decoder RNN. The encoder produces a context vector summarizing the input, and the decoder uses this context to generate the output one token at a time.

The bottleneck of compressing the entire input into a single fixed-size vector motivated the development of the **attention mechanism**.

## The Attention Mechanism

**Attention** was introduced to allow the decoder to "look back" at all encoder hidden states when generating each output token, rather than relying on a single summary vector.

For each decoder timestep, attention computes:
1. **Scores**: how relevant each encoder hidden state is to the current decoder state
2. **Weights**: softmax of the scores, forming a probability distribution over encoder states
3. **Context vector**: weighted sum of encoder hidden states

```python
# Simplified attention computation
scores = [dot(decoder_state, encoder_state_i) for encoder_state_i in encoder_states]
weights = softmax(scores)
context = sum(w * h for w, h in zip(weights, encoder_states))
```

This allows the model to focus on different parts of the input when generating each output token. For translating a sentence, the model might attend to the subject when generating the subject in the target language, and to the verb when generating the verb.

## Transformers

The **Transformer** architecture, introduced in the landmark 2017 paper "Attention Is All You Need," replaced recurrence entirely with **self-attention**, enabling parallel processing of entire sequences.

### Self-Attention

In self-attention, each token attends to every other token in the sequence. For each token, the model computes three vectors from the input embedding:

- **Query (Q)**: what this token is looking for
- **Key (K)**: what this token contains
- **Value (V)**: the information this token provides

Attention scores are computed as:

Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

where d_k is the dimensionality of the key vectors (the scaling factor prevents dot products from growing too large).

### Multi-Head Attention

Rather than computing a single attention function, transformers use **multi-head attention**: multiple attention "heads" that each learn different types of relationships. One head might attend to syntactic dependencies, another to semantic similarities, and another to positional relationships. The outputs of all heads are concatenated and projected:

MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W_O

where each headᵢ = Attention(Q × Wᵢ_Q, K × Wᵢ_K, V × Wᵢ_V)

### Positional Encoding

Since self-attention has no inherent notion of word order (unlike RNNs), transformers add **positional encodings** to the input embeddings to inject information about each token's position in the sequence. The original paper used sinusoidal functions of different frequencies, though learned positional embeddings are also common.

### Transformer Architecture

A full transformer consists of:
- **Encoder**: a stack of layers, each containing multi-head self-attention and a feed-forward network, with residual connections and layer normalization
- **Decoder**: similar to the encoder but with an additional cross-attention layer that attends to the encoder output, and masked self-attention that prevents tokens from attending to future positions

## Large Language Models

Modern language models are built on the transformer architecture and trained on massive text corpora.

### GPT (Generative Pre-trained Transformer)

GPT uses only the transformer **decoder** stack. It is trained with a simple objective: predict the next token given all previous tokens (causal language modeling). Despite this simple training objective, large GPT models exhibit remarkable capabilities including text generation, summarization, translation, question answering, and even coding.

### BERT (Bidirectional Encoder Representations from Transformers)

BERT uses only the transformer **encoder** stack. It is trained with two objectives: masked language modeling (predict randomly masked tokens from context on both sides) and next sentence prediction. BERT's bidirectional context makes it excellent for understanding tasks like classification, named entity recognition, and question answering.

### Training at Scale

Modern LLMs are trained on billions of tokens of text using hundreds or thousands of GPUs. Key insights enabling scale include efficient attention implementations, mixed-precision training, and parallelism strategies (data, tensor, and pipeline parallelism).

**Transfer learning** is central to the LLM paradigm: a large model is **pre-trained** on a general corpus, then **fine-tuned** on specific tasks with much less data. This allows the model's general language understanding to transfer to specialized domains.

## Key Takeaways

NLP has evolved from rule-based systems through statistical methods to deep learning. Word embeddings capture semantic meaning in dense vectors. Attention mechanisms allow models to focus on relevant context. The Transformer architecture, built entirely on attention, enables parallel processing and has become the foundation of modern language AI. Large language models demonstrate that scaling up transformers with more data and parameters leads to emergent capabilities that continue to surprise researchers and practitioners alike.
