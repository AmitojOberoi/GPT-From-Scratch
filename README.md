# AI-LLM-From-Scratch

A GPT-style Large Language Model built from scratch using **Python** and **PyTorch**, following the fundamental concepts behind modern transformer-based language models.

---

## 📚 Project Goal

The goal of this project is to understand how Large Language Models work internally by implementing every major component from scratch instead of relying on high-level libraries.

The project starts with a simple character-level tokenizer and gradually progresses toward a complete GPT-style transformer capable of generating text.

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- NumPy
- Matplotlib
- tqdm

---

## 📂 Project Structure

```
AI-LLM-From-Scratch/
│
├── data/
│   └── wizard_of_oz.txt
│
├── outputs/
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── attention.py
│   ├── model.py
│   ├── generate.py
│   ├── config.py
│   ├── train.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ✅ Features Implemented

- Character-level tokenizer
- Character ↔ Integer encoding
- Integer ↔ Character decoding
- Vocabulary generation
- Dataset loading
- PyTorch tensor conversion
- Train / Validation split
- Modular project structure

---

## 🚧 Current Progress

- [x] Project setup
- [x] Load training corpus
- [x] Character-level tokenizer
- [x] Encode / Decode pipeline
- [x] Convert dataset to PyTorch tensors
- [x] Train / Validation split
- [ ] Batch generation
- [ ] Bigram language model
- [ ] Embedding layer
- [ ] Training loop
- [ ] Self-attention
- [ ] Multi-head attention
- [ ] Transformer blocks
- [ ] GPT Language Model
- [ ] Text generation
- [ ] Model checkpoints

---

## 📖 Dataset

**The Wonderful Wizard of Oz**

Source: Project Gutenberg

The initial implementation uses this public-domain book to build and train the first character-level language model.

---

## 🚀 Future Improvements

- OpenWebText pretraining
- Model checkpoint saving/loading
- Command-line inference
- Better tokenizer
- Byte Pair Encoding (BPE)
- Fine-tuning support

---

## 📜 License

This project is intended for educational purposes while learning how transformer-based language models work.