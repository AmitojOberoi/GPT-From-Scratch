# GPT From Scratch 🧠

A small **GPT-style character-level language model built from scratch using Python and PyTorch**.

The goal of this project is to understand how GPT-style language models work internally by implementing the major components of a Transformer rather than using a pre-trained model.

---

## 🚀 Features

- Character-level tokenizer
- Train/validation dataset pipeline
- Bigram language model baseline
- Self-attention
- Query / Key / Value projections
- Causal masking
- Multi-head attention
- Feed-forward network
- Layer normalization
- Residual connections
- Positional embeddings
- Transformer blocks
- GPT language model
- Training and validation
- Autoregressive text generation
- Model checkpointing
- Model evaluation
- Standalone inference

---

## 🏗️ Architecture

```text
Raw Text
   ↓
Character Tokenizer
   ↓
Token Embeddings + Positional Embeddings
   ↓
Transformer Blocks
   ├── Multi-Head Self-Attention
   ├── Feed-Forward Network
   ├── Layer Normalization
   └── Residual Connections
   ↓
Language Model Head
   ↓
Next-Token Prediction
```

---

## 📊 Model Configuration

| Configuration | Value |
|---|---:|
| Vocabulary Size | 89 |
| Embedding Size | 32 |
| Attention Heads | 4 |
| Transformer Blocks | 2 |
| Context Length | 8 |
| Parameters | 31,321 |
| Training Tokens | 204,397 |
| Validation Tokens | 22,711 |

---

## 📚 Dataset

The initial model is trained on **The Wonderful Wizard of Oz**.

A character-level tokenizer converts the text into a vocabulary of 89 unique characters.

Example:

```text
hello
↓
[63, 60, 67, 67, 70]
↓
hello
```

The dataset is split into:

```text
90% Training
10% Validation
```

---

## 📈 Training Results

Example training run:

```text
Step    0 | Train Loss: 4.7120 | Val Loss: 4.7304
Step  100 | Train Loss: 3.0971 | Val Loss: 3.2237
Step  500 | Train Loss: 2.5087 | Val Loss: 3.0298
Step  900 | Train Loss: 2.3807 | Val Loss: 2.9922
```

Final evaluation:

```text
Train Loss:      2.4033
Validation Loss: 2.9766
```

The model learns character-level patterns from the training corpus and can generate novel text.

---

## ✍️ Text Generation

Train the model:

```powershell
python src\train.py
```

Generate text using the saved checkpoint:

```powershell
python src\generate.py
```

Generate from a custom prompt:

```powershell
python src\generate.py --prompt "Dorothy" --tokens 300
```

Evaluate the trained model:

```powershell
python src\evaluate.py
```

---

## 💾 Checkpointing

After training, the model is saved as:

```text
models/gpt_wizard_of_oz.pth
```

The checkpoint contains the trained model parameters and configuration required to reconstruct the model.

The checkpoint is excluded from Git version control.

---

## 📂 Project Structure

```text
GPT-From-Scratch/
│
├── data/
│   └── wizard_of_oz.txt
│
├── models/
│   └── gpt_wizard_of_oz.pth
│
├── src/
│   ├── attention.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Git**
- **GitHub**

---

## 🔮 Future Improvements

Possible future versions may include:

- Larger Transformer architecture
- Longer context windows
- BPE/subword tokenization
- Larger training corpora
- Temperature sampling
- Top-k / Top-p sampling
- GPU training
- Mixed precision
- Learning-rate scheduling
- Larger-scale experiments

---

## 📌 Project Status

### **v1.0 — Complete ✅**

The project successfully implements a small GPT-style language model from:

```text
Tokenization
    ↓
Dataset
    ↓
Attention
    ↓
Transformer
    ↓
GPT
    ↓
Training
    ↓
Checkpointing
    ↓
Evaluation
    ↓
Inference
```

Built to understand **GPT from the inside out**.