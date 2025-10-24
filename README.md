# Transformer-Based English-Italian Translation Model with Attention Visualization

This project implements a Transformer model from scratch using PyTorch to perform English-to-Italian translation. The key highlight of this project is a deep dive into the self-attention and cross-attention mechanisms, with interactive visualizations to help understand how the model learns relationships between tokens across different layers and heads.

## 📌 Project Highlights

- ✅ Built a full Transformer model using PyTorch (Encoder–Decoder architecture)
- 🧠 Implemented Greedy Decoding for inference
- 📊 Visualized attention matrices using **Altair** for:
  - Encoder self-attention
  - Decoder self-attention
  - Encoder-Decoder (cross) attention
- 📝 Tokenized and decoded sequences using source and target vocabularies
- 🔍 Performed layer-wise and head-wise inspection of attention behavior

## 🏗️ Key Components

### 1. Data Preprocessing
- Tokenized English and Italian sentence pairs
- Batched inputs with padding
- Computed input/output lengths for cleaner visualization

### 2. Model Architecture
- Custom-built Transformer including:
  - Multi-head attention layers
  - Positional encodings
  - Layer normalization
  - Feed-forward networks
- Tracked and stored attention weights (`self.attention_scores`) for visualization

### 3. Training & Inference
- Trained using teacher forcing
- Inference via greedy decoding (predict next token with highest probability)

### 4. Attention Visualization
- Built helper functions to extract attention weights from specific layers and heads
- Displayed attention maps using **Altair** in three modes:
  - Encoder Self-Attention: input tokens attending to themselves
  - Decoder Self-Attention: predicted tokens attending to previous ones
  - Cross-Attention: decoder tokens attending to encoder inputs

## 🛠 Libraries Used

- `torch` – core deep learning framework
- `altair` – interactive chart visualizations
- `numpy`, `pandas` – tensor and data manipulation
- `matplotlib` (optional) – alternative visualization

## 🖼 Sample Visualizations

- Layer-wise heatmaps showing which source tokens contribute most to translated words
- Diagonal dominance in self-attention heads (token attends to itself)
- Distinct patterns across heads showing syntactic or semantic mappings

## 📌 Limitations

- Not trained on large corpora – intended for educational/demo purposes
- Greedy decoding may not always give best translation (vs. beam search)

## 📚 Learnings

- Deeper understanding of attention layers in Transformers
- Practical experience with sequence-to-sequence learning
- Model interpretability through visualization
