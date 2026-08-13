# 🧠 GPT From Scratch

A GPT-style Large Language Model built from scratch using **Python and PyTorch**.

The goal of this project is to understand how modern language models work internally by implementing the major components step by step, starting with a character-level tokenizer and gradually progressing toward a Transformer-based GPT architecture.

---

## 🎯 Project Goal

This project is a hands-on implementation of a language model from the ground up.

Instead of using pre-trained models or high-level LLM libraries, the project progressively implements:

- Tokenization
- Dataset preparation
- Language modeling
- Embeddings
- Loss functions
- Gradient descent
- Text generation
- Self-attention
- Multi-head attention
- Transformer blocks
- GPT architecture

The project is also structured as a professional Python/PyTorch codebase rather than a single notebook.

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Git**
- **GitHub**

---

## 📂 Project Structure

```text
GPT-From-Scratch/
│
├── data/
│   └── wizard_of_oz.txt
│
├── models/
│
├── outputs/
│
├── src/
│   ├── attention.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore