# 🧠 GPT From Scratch

A GPT-style Large Language Model built from scratch using **Python and PyTorch**.

The goal of this project is to understand how modern Transformer-based Large Language Models work internally by implementing the architecture step by step — starting from raw text and a character-level tokenizer and progressing toward a complete GPT-style model.

---

# 🎯 Project Goal

This is a hands-on implementation of a language model without relying on pre-trained LLMs or high-level language-model frameworks.

The project progressively implements:

- Character-level tokenization
- Dataset preparation
- Language modeling
- Embeddings
- Loss functions
- Gradient descent
- Autoregressive text generation
- Self-attention
- Query / Key / Value projections
- Scaled dot-product attention
- Multi-head attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Transformer blocks
- GPT architecture

The codebase is organized as a modular Python/PyTorch project rather than a single notebook.

---

# 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Git**
- **GitHub**

---

# 📂 Project Structure

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
```

---

# 📚 Dataset

The initial training corpus is:

**The Wonderful Wizard of Oz**

The text is sourced from **Project Gutenberg** and is used as the initial corpus for experimenting with character-level language modeling.

Current dataset statistics:

```text
Training Tokens   : 204,397
Validation Tokens : 22,711
Vocabulary Size   : 89 characters
```

---

# 🔤 Character-Level Tokenization

The project currently uses a character-level tokenizer.

For example:

```text
hello
```

is converted into integer token IDs:

```text
[63, 60, 67, 67, 70]
```

The tokenizer can also decode the IDs back into text:

```text
[63, 60, 67, 67, 70]
        ↓
hello
```

---

# 🧠 Bigram Language Model

The first language model implemented is a **character-level Bigram Language Model**.

The model predicts the next character based on the current character.

Conceptually:

```text
Current Character
       ↓
Bigram Model
       ↓
Next Character
```

The model uses a PyTorch embedding table:

```text
Embedding(89, 89)
```

The Bigram model was trained using:

- Cross-Entropy Loss
- Backpropagation
- AdamW optimizer
- Train/validation evaluation

---

# ✍️ Autoregressive Text Generation

The Bigram model can generate text one character at a time.

Starting from:

```text
D
```

the model repeatedly predicts and samples the next character:

```text
D
 ↓
next character
 ↓
next character
 ↓
next character
 ↓
...
```

Example output from the current model:

```text
DOZl,?E#Nk”5yHWuGHXOv?’FwlsE...
```

The output is intentionally included as a demonstration of the current model's limitations.

Because the Bigram model only considers the current character, it cannot maintain meaningful long-range context.

This limitation motivates the transition toward Transformer-based language modeling.

---

# 🧠 Self-Attention

A basic self-attention implementation was first created to understand the underlying mechanism.

The attention pipeline is:

```text
Input Representations
        ↓
Attention Scores
        ↓
Softmax
        ↓
Attention Weights
        ↓
Weighted Values
        ↓
Output Representations
```

---

# 🔑 Query, Key and Value

The attention mechanism was then extended to use learned projections:

```text
Q = XWq
K = XWk
V = XWv
```

Attention scores are calculated using:

```text
QKᵀ
```

and scaled using:

```text
QKᵀ
─────
 √dk
```

The resulting scores are passed through Softmax and multiplied by the Value representations.

---

# 🔒 Causal Attention

The model uses a **causal attention mask**.

This prevents a token from accessing future tokens during autoregressive language modeling.

The attention pattern is therefore:

```text
Token 1 → Token 1

Token 2 → Token 1, Token 2

Token 3 → Token 1, Token 2, Token 3

Token 4 → Token 1, Token 2, Token 3, Token 4
```

This ensures that the model cannot see information from future positions when predicting the next token.

---

# 🧩 Multi-Head Attention

Multiple attention heads were implemented so that different heads can learn different relationships between tokens.

Current test configuration:

```text
Embedding Size : 32
Number of Heads: 4
Head Size      : 8
```

Conceptually:

```text
              Input
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
     Head 1   Head 2   Head 3 ... Head N
       │        │        │
       └────────┼────────┘
                ↓
          Concatenation
                ↓
        Linear Projection
                ↓
             Output
```

---

# 🏗️ Transformer Block

The individual components have now been combined into a Transformer block.

The current architecture is:

```text
Input
  ↓
Layer Normalization
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Layer Normalization
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output
```

The current test configuration uses:

```text
Batch Size     : 4
Sequence Length: 8
Embedding Size : 32
Attention Heads: 4
```

The Transformer block preserves the input/output representation size:

```text
Input:
[4, 8, 32]

       ↓

Transformer Block

       ↓

Output:
[4, 8, 32]
```

Current test parameter count:

```text
12,608 parameters
```

---

# 📊 Current Progress

| Component | Status |
|---|---|
| Project Setup | ✅ |
| Git / GitHub Integration | ✅ |
| Wizard of Oz Dataset | ✅ |
| Character Tokenizer | ✅ |
| Encode / Decode | ✅ |
| PyTorch Tensor Conversion | ✅ |
| Train / Validation Split | ✅ |
| Batch Generation | ✅ |
| Bigram Language Model | ✅ |
| Embedding Layer | ✅ |
| Forward Pass | ✅ |
| Cross-Entropy Loss | ✅ |
| AdamW Optimizer | ✅ |
| Backpropagation | ✅ |
| Training Loop | ✅ |
| Train / Validation Evaluation | ✅ |
| Autoregressive Generation | ✅ |
| Basic Self-Attention | ✅ |
| Query / Key / Value | ✅ |
| Scaled Dot-Product Attention | ✅ |
| Causal Attention Mask | ✅ |
| Multi-Head Attention | ✅ |
| Feed-Forward Network | ✅ |
| Layer Normalization | ✅ |
| Residual Connections | ✅ |
| Transformer Block | ✅ |
| Positional Encoding | ⏳ |
| GPT Language Model | ⏳ |
| Transformer Integration | ⏳ |
| Model Training | ⏳ |
| Model Checkpoints | ⏳ |
| Larger Training Corpus | ⏳ |
| Command-Line Interface | ⏳ |

---

# 🛣️ Roadmap

## Phase 1 — Language Model Fundamentals

- [x] Character tokenizer
- [x] Dataset pipeline
- [x] Train/validation split
- [x] Batch generation
- [x] Bigram model
- [x] Cross-entropy loss
- [x] Gradient descent
- [x] Training loop
- [x] Text generation

## Phase 2 — Transformer Fundamentals

- [x] Self-attention
- [x] Query / Key / Value
- [x] Scaled dot-product attention
- [x] Causal masking
- [x] Multi-head attention
- [x] Feed-forward network
- [x] Layer normalization
- [x] Residual connections
- [x] Transformer block
- [ ] Positional encoding

## Phase 3 — GPT

- [ ] Token embedding + positional embedding
- [ ] Stack multiple Transformer blocks
- [ ] GPT language model
- [ ] Language-modeling head
- [ ] Forward pass
- [ ] Loss evaluation
- [ ] Autoregressive generation
- [ ] Model training
- [ ] Model checkpoints
- [ ] Model loading

## Phase 4 — Scaling & Engineering

- [ ] Larger corpus
- [ ] Efficient data loading
- [ ] GPU training
- [ ] Training checkpoints
- [ ] Command-line interface
- [ ] Performance optimization
- [ ] Improved text generation
- [ ] Final documentation

---

# ▶️ Running the Project

Clone the repository:

```bash
git clone <repository-url>
```

Enter the project directory:

```bash
cd GPT-From-Scratch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the language-model training pipeline:

```powershell
python src\train.py
```

Run the Transformer attention experiments:

```powershell
python src\attention.py
```

---

# 🧪 Current Limitations

The current implementation is intentionally educational and relatively small.

### Character-Level Tokenization

The tokenizer operates at the character level rather than using modern subword tokenization.

### Bigram Model

The current trained language model is still a Bigram model and therefore has extremely limited context.

### Small Training Run

The current Bigram experiment uses a small number of training iterations.

### Small Corpus

The initial training corpus consists of a single book.

### Transformer Not Yet Integrated

The attention and Transformer components have been implemented and tested independently, but they have not yet been integrated into the final GPT language model.

These limitations will be addressed as the project progresses.

---

# 📖 Learning Objectives

This project is designed to build practical understanding of:

- PyTorch
- Neural network architecture
- Embeddings
- Language modeling
- Cross-entropy loss
- Gradient descent
- Optimization
- Autoregressive generation
- Self-attention
- Query / Key / Value
- Multi-head attention
- Layer normalization
- Residual connections
- Feed-forward networks
- Transformer architecture
- GPT architecture
- Model training
- Model evaluation

---

# 🚀 Final Goal

The final objective is to build a small GPT-style language model **from scratch**, understand each component of its architecture, train it on a text corpus, and generate text without relying on a pre-trained LLM.

The project is evolving through:

```text
Raw Text
   ↓
Character Tokenizer
   ↓
Bigram Language Model
   ↓
Self-Attention
   ↓
Multi-Head Attention
   ↓
Transformer Block
   ↓
Positional Encoding
   ↓
GPT
   ↓
Text Generation
```

---

# 📌 Current Status

**Completed:** Bigram Language Model, Text Generation, Self-Attention, Multi-Head Attention, and Transformer Block.

**Next:** Positional Encoding and integration of the Transformer components into the GPT language model.