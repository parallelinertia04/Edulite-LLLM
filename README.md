# EduLiteLLM

A ~15M-parameter transformer language model trained from scratch on OpenStax
educational textbooks, using PyTorch and a SentencePiece BPE tokenizer.

Built as a complete, end-to-end NLP pipeline: PDF ingestion -> text cleaning ->
tokenizer training -> sequence creation -> training -> checkpointing.

---

## Results

Trained for 4 logged epochs (~17,000 steps each, batch size 64).

| Epoch | Total Loss | Avg Loss/Step | Final Step Loss | Perplexity |
| :---- | ---------: | ------------: | --------------: | ---------: |
| 1     | 44,430.10  |         2.61  |         1.5476  |       4.70 |
| 2     | 22,914.81  |         1.35  |         1.1901  |       3.29 |
| 3     | 19,196.41  |         1.13  |         1.0379  |       2.82 |
| 4     |     —      |           —   | 1.0144 (step 13k) |       2.76 |

**Key outcomes**

- **56.8% reduction in average loss** between epoch 1 (2.61) and epoch 3 (1.13)
- Perplexity improved from **4.70 -> 2.76**
- Training was stable: no loss spikes or divergence across 4 epochs
- Most of the gain landed in epoch 1 -> 2, with continued refinement through epoch 4

> Perplexity is `exp(loss)`, computed from the final logged step loss of each epoch.
> Epoch 4 was still in progress when logs were captured. The training script is
> configured for 5 epochs.

### Training logs

Raw terminal captures are committed under
![Epoch 1](./results%20of%20model%20training/epoch%201.jpeg)
![Epoch 2](./results%20of%20model%20training/epoch%202.jpeg)
![Epoch 3](./results%20of%20model%20training/epoch%203.jpeg)
![Epoch 4](./results%20of%20model%20training/epoch%204.jpeg)
one screenshot per epoch.

---

## Model Architecture

A decoder-style transformer implemented from scratch in PyTorch, with no
`transformers` dependency.

| Hyperparameter   | Value       |
| :--------------- | :---------- |
| Parameters       | 14,999,072 (~15.0M) |
| Vocabulary size  | 20,000      |
| Embedding dim    | 256         |
| Layers           | 6           |
| Attention heads  | 4 (see Limitations) |
| Context length   | 128 tokens  |
| FFN expansion    | 4x          |
| Optimizer        | Adam, lr 3e-4 |
| Loss function    | Cross-entropy |

### Parameter breakdown

| Component                  | Parameters |
| :------------------------- | ---------: |
| Token embedding            |  5,120,000 |
| 6 x Transformer blocks     |  4,738,560 |
| Output projection (256 -> 20,000) |  5,140,000 |
| Layer norms                |         512 |
| **Total**                  | **14,999,072** |

### Components

- `TokenEmbedding` — token lookup table
- `SelfAttention` — query / key / value / output projections, scaled dot-product
  scores, softmax attention, output projection
- `FeedForward` — two-layer MLP (256 -> 1024 -> 256) with ReLU
- `TransformerBlock` — pre-norm residual block combining attention and FFN
- `EduLiteLLM` — embedding, 6 stacked blocks, final LayerNorm, linear LM head

---

## Dataset

Two OpenStax open-access textbooks, ingested from PDF:

| Source                            | PDF size  | Extracted text |
| :-------------------------------- | --------: | -------------: |
| Biology 2e                        | 378.66 MB |       3.88 MB  |
| Organic Chemistry                 | 193.05 MB |       1.96 MB  |
| **Combined, cleaned**             |           | **5.65 MB**   |

- Lowercased, whitespace-normalised, and split into sentences
- Sentences shorter than 20 characters discarded
- Tokenized to approximately **1.09M tokens** with SentencePiece BPE
  (`character_coverage=1.0`, 20,000-piece vocabulary)
- Converted into ~**1.09M training sequences** of length 128 using a sliding
  window with a one-token shift, yielding ~**139M next-token predictions**

---

## Pipeline

| Stage | Script                    | Description                                          |
| :---- | :------------------------ | :--------------------------------------------------- |
| 1     | `convert_pdf.py`          | Extract text from OpenStax PDFs using PyMuPDF        |
| 2     | `clean_text.py`           | Lowercase, normalise whitespace, sentence-split      |
| 3     | `train_tokenizer.py`      | Train SentencePiece BPE tokenizer (20K vocab)         |
| 4     | `test_tokenizer.py`       | Verify encode/decode round-trip                      |
| 5     | `tokenize_dataset.py`     | Encode cleaned corpus to token IDs                   |
| 6     | `create_sequences.py`     | Build 128-token sliding-window input/target pairs    |
| 7     | `train_model.py`          | Train the transformer and save the checkpoint        |

---


Generated artefacts (`*.npy`, `*.pt`, source PDFs) are excluded via `.gitignore`
to keep the repository lightweight. Regenerate them by running the pipeline.

---
## Setup
```bash
git clone https://github.com/parallelinertia04/Edulite-LLLM.git
cd Edulite-LLLM

pip install torch numpy sentencepiece pymupdf
```

## Usage

Run the stages in order:

```bash
python scripts/convert_pdf.py
python scripts/clean_text.py
python scripts/train_tokenizer.py
python scripts/tokenize_dataset.py
python scripts/create_sequences.py
python scripts/train_model.py
```

`train_model.py` automatically selects CUDA when available, otherwise CPU.

---

## Limitations

This is a compact educational language model. The following are known and
deliberately out of scope for the current version:

- **No positional encoding.** The model has no representation of token order,
  which is the single largest architectural limitation.
- **No causal masking.** Attention is bidirectional, so this behaves as an
  encoder-style model rather than a left-to-right decoder.
- **Multi-head attention is not fully wired up.** `head_dim` is computed but
  query/key/value are not reshaped into heads, so the attention currently runs
  as single-head over the full 256-dimensional projection.
- **Small context window** (128 tokens) and a small corpus limit fluency.
- **No generation script yet.** A sampling/greedy decode loop is the next step.
- **No evaluation harness.** Perplexity is tracked from training loss only;
  there is no held-out validation split.

### Planned improvements

1. Add learned or sinusoidal positional embeddings
2. Implement proper multi-head attention and causal masking
3. Add a train/validation split with held-out evaluation
4. Increase context length and layer count
5. Add a text-generation script for qualitative evaluation

---

## Tech Stack

Python · PyTorch · NumPy · SentencePiece · PyMuPDF

---

## License

Code is released under the [MIT License](LICENSE).

The bundled dataset is derived from OpenStax textbooks, which are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution to
OpenStax is required if you redistribute the extracted text or derivatives of it.
