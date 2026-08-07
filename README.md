# 🧠 AI-LLM-From-Scratch

A GPT-style Large Language Model built completely from scratch using **Python** and **PyTorch**.

This project focuses on understanding how modern Large Language Models work internally by implementing every major component step by step—from a character-level tokenizer to a complete GPT-style Transformer.

---

# 📚 Project Goals

- Learn how LLMs work internally
- Implement every component instead of using high-level libraries
- Follow professional software engineering practices
- Build a modular, maintainable codebase
- Document the learning process through Git commits

---

# 🚀 Tech Stack

- Python 3
- PyTorch
- NumPy
- Matplotlib
- tqdm
- Git & GitHub

---

# 📂 Project Structure

```text
AI-LLM-From-Scratch/
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
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ✅ Features Implemented

### Data Pipeline

- Character-level tokenizer
- Character vocabulary generation
- Character ↔ Integer encoding
- Integer ↔ Character decoding
- Dataset loading
- PyTorch tensor conversion
- Train / Validation split
- Random batch generation

---

# 📈 Current Progress

| Stage | Status |
|-------|--------|
| Project Setup | ✅ |
| Dataset Loading | ✅ |
| Character Tokenizer | ✅ |
| Encode / Decode | ✅ |
| PyTorch Tensor Conversion | ✅ |
| Dataset Pipeline | ✅ |
| Train / Validation Split | ✅ |
| Batch Generation | ✅ |
| Bigram Language Model | ⏳ |
| Embedding Layer | ⏳ |
| Training Loop | ⏳ |
| Self-Attention | ⏳ |
| Multi-Head Attention | ⏳ |
| Transformer Blocks | ⏳ |
| GPT Language Model | ⏳ |
| Text Generation | ⏳ |

---

# 📖 Dataset

**The Wonderful Wizard of Oz**

Source: Project Gutenberg

The project begins by training on a character-level representation of this public-domain novel before moving toward larger datasets.

---

# ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/AmitojOberoi/llm-from-scratch.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train.py
```

---

# 🛣️ Roadmap

- [x] Character-level tokenizer
- [x] Dataset pipeline
- [x] Batch generation
- [ ] Bigram Language Model
- [ ] Embedding layer
- [ ] Training loop
- [ ] Self-attention
- [ ] Multi-head attention
- [ ] Transformer blocks
- [ ] GPT Language Model
- [ ] Model checkpoints
- [ ] Text generation

---

# 🎯 Learning Objectives

This repository is intended as a hands-on implementation of the core concepts behind transformer-based language models, including:

- Tokenization
- Language modeling
- Embeddings
- Self-attention
- Multi-head attention
- Transformer architecture
- Autoregressive text generation

---

# 📄 License

This project is intended for educational and learning purposes.