# Transformer-Machine-Translation
This project implements a **Transformer-based machine translation model** built entirely from scratch using **PyTorch**. Inspired by the original “Attention is All You Need” paper, this repository covers the **end-to-end training pipeline**, including **tokenization**, **multi-head attention**, **encoder-decoder architecture**, **greedy decoding**, and **attention score visualization** using **Altair**.

## 🔍 Overview

- Implements a custom Transformer model for language translation
- Encoder-Decoder with **multi-head self-attention** and **positional encodings**
- Trained using CrossEntropyLoss on a parallel language dataset (e.g., English → Italian)
- Inference via **Greedy Decoding**
- Visualizes:
  - **Encoder Self-Attention**
  - **Decoder Self-Attention**
  - **Cross Attention (Encoder ↔ Decoder)**

## 🛠️ Tech Stack

- Python 🐍
- PyTorch ⚙️
- Altair 📊 (for visualizing attention scores)
- Tokenizers from TorchText or custom vocabularies
- Jupyter Notebooks for modular experimentation
