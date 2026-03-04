# Transformers and Large Language Models

## The Attention Mechanism

Attention is the core innovation behind transformers. It allows the model to focus on different parts of the input when producing each part of the output. Instead of compressing an entire sequence into a fixed-size vector (as RNNs do), attention lets the model look at all input positions and decide which ones are most relevant.

The attention mechanism computes three vectors for each input token: Query (Q), Key (K), and Value (V). The attention score between two tokens is the dot product of the query of one with the key of the other. These scores are scaled and passed through softmax to get attention weights, which are used to compute a weighted sum of the value vectors.

Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

## Self-Attention

Self-attention (or intra-attention) is when a sequence attends to itself. Each token computes attention scores with every other token in the same sequence. This allows the model to capture dependencies regardless of distance — a word at position 1 can directly attend to a word at position 100.

Multi-head attention runs multiple attention operations in parallel with different learned projections. Each "head" can learn to attend to different types of relationships (syntactic, semantic, positional). The outputs are concatenated and projected.

## The Transformer Architecture

The original transformer (Vaswani et al., 2017, "Attention Is All You Need") consists of an encoder and decoder, each made of stacked layers.

### Encoder
Each encoder layer has two sub-layers: multi-head self-attention and a feedforward network. Residual connections and layer normalization are applied around each sub-layer.

### Decoder
Each decoder layer adds a third sub-layer: cross-attention over the encoder output. The self-attention in the decoder is masked to prevent positions from attending to future tokens (autoregressive property).

### Positional Encoding
Since transformers process all tokens in parallel (no sequential order), positional encodings are added to the input embeddings to provide position information. These can be sinusoidal (fixed) or learned.

## Large Language Models (LLMs)

LLMs are transformer-based models trained on massive text corpora. They learn to predict the next token in a sequence (autoregressive language modeling). Key characteristics:

### Scale
Modern LLMs have billions of parameters. GPT-3 has 175 billion, GPT-4 is estimated to be much larger. Scale tends to improve capabilities through emergent behaviors — abilities that appear at certain model sizes but not below.

### Pre-training and Fine-tuning
LLMs are first pre-trained on huge general text corpora (books, websites, code) to learn language patterns. They can then be fine-tuned on specific tasks or aligned with human preferences through RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization).

### Tokenization
LLMs don't process raw text — they break it into tokens using algorithms like BPE (Byte-Pair Encoding) or SentencePiece. Common words become single tokens while rare words are split into subword pieces. Understanding tokenization is crucial for prompt engineering and cost estimation.

### Context Window
The context window is the maximum number of tokens an LLM can process at once. GPT-4 supports 128K tokens, Claude supports 200K tokens. Longer context allows processing entire documents but increases computational cost quadratically with sequence length.

## Retrieval-Augmented Generation (RAG)

RAG combines LLMs with external knowledge retrieval to overcome limitations like knowledge cutoffs and hallucination.

### How RAG Works
1. **Index**: Documents are split into chunks, embedded into vectors, and stored in a vector database
2. **Retrieve**: When a user asks a question, the query is embedded and similar document chunks are retrieved
3. **Generate**: Retrieved chunks are added to the LLM's prompt as context, grounding the response in actual documents

### Vector Databases
Vector databases (Pinecone, Weaviate, ChromaDB, Milvus, pgvector) are optimized for storing and searching high-dimensional vectors. They use approximate nearest neighbor (ANN) algorithms like HNSW for fast similarity search.

### Embedding Models
Embedding models convert text into dense vectors that capture semantic meaning. Similar texts have vectors close together in the embedding space. Models like OpenAI's text-embedding-3-small or open-source alternatives like BGE and E5 are commonly used.

### Chunking Strategies
How you split documents affects retrieval quality. Common strategies include fixed-size chunks with overlap, recursive splitting on natural boundaries (paragraphs, sentences), semantic chunking (grouping related content), and parent-child chunking for hierarchical retrieval.

## Prompt Engineering

Prompt engineering is the practice of designing effective inputs to get desired outputs from LLMs. Key techniques include:

- **Zero-shot**: Ask the model directly without examples
- **Few-shot**: Provide examples of the desired input-output pattern
- **Chain-of-thought**: Ask the model to reason step by step
- **System prompts**: Set the model's role, constraints, and behavior
- **Temperature**: Controls randomness (0 = deterministic, 1 = creative)

## AI Agents

AI agents are systems where LLMs can take actions, use tools, and make decisions in a loop. Key components:

- **Planning**: The agent breaks down complex tasks into steps
- **Tool Use**: The agent calls external APIs, runs code, or searches databases
- **Memory**: Short-term (conversation context) and long-term (persistent storage)
- **Reflection**: The agent evaluates its own outputs and corrects mistakes

Frameworks like LangGraph, CrewAI, and AutoGen enable building multi-agent systems where specialized agents collaborate on complex tasks.

## Fine-tuning vs RAG

When to use each approach:

**Use RAG when:**
- You need up-to-date information
- The knowledge base changes frequently
- You want source attribution and traceability
- You have limited compute budget

**Use Fine-tuning when:**
- You need to change the model's behavior or style
- The task requires specialized reasoning patterns
- You need consistent output formatting
- Latency is critical (no retrieval step)

In practice, many production systems combine both: fine-tune for behavior and use RAG for knowledge.
