"""
Transformer toy translation - single-file runnable script

Files included (this single script contains everything):
- Model implementation: MultiHeadAttention, FeedForward, EncoderLayer, DecoderLayer, Encoder, Decoder, TransformerSeq2Seq
- Synthetic dataset: random token sequences -> reversed sequences as targets (easy to learn; good for attention visualization)
- Training loop with teacher forcing
- Evaluation and BLEU computation (using nltk)
- Visualizations saved to runs/mt/: loss curves, attention heatmaps, masks demo, decodes table, bleu report

How to run (after saving as transformer_toy_translation.py):
1. Create a virtualenv and install requirements from requirements.txt (shown at bottom of file)
2. python transformer_toy_translation.py --epochs 20 --batch-size 64

Notes:
- This script is intentionally self-contained so it can be copied to VS Code and run directly.
- It uses a small synthetic dataset; training is quick on CPU but GPU will be faster. Set --device cuda if available.

"""

import os
import math
import time
import random
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# BLEU from nltk
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# -----------------------------
# Utility: Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Synthetic Parallel Corpus
# -----------------------------
class ReversedDataset(Dataset):
    """Generate random token sequences and targets = reversed(source)"""
    def __init__(self, n_samples=5000, min_len=3, max_len=7, vocab_size=30):
        self.n_samples = n_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.SOS = '<sos>'
        self.EOS = '<eos>'
        self.PAD = '<pad>'
        self.UNK = '<unk>'

        # create vocab tokens as words t0, t1, ...
        tokens = [f't{i}' for i in range(vocab_size)]
        self.itos = [self.PAD, self.SOS, self.EOS, self.UNK] + tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        self.data = []
        for _ in range(n_samples):
            L = random.randint(min_len, max_len)
            seq = [random.choice(tokens) for _ in range(L)]
            src = seq
            trg = list(reversed(seq))
            self.data.append((src, trg))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def encode(self, tokens):
        return [self.stoi.get(t, self.stoi[self.UNK]) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def collate_fn(self, batch):
        # batch: list of (src_tokens, trg_tokens)
        src_seqs = [self.encode(x[0]) for x in batch]
        trg_seqs = [self.encode(x[1]) for x in batch]

        # add SOS/EOS
        src_seqs = [[self.stoi[self.SOS]] + s + [self.stoi[self.EOS]] for s in src_seqs]
        trg_seqs = [[self.stoi[self.SOS]] + s + [self.stoi[self.EOS]] for s in trg_seqs]

        src_lens = [len(s) for s in src_seqs]
        trg_lens = [len(s) for s in trg_seqs]
        max_src = max(src_lens)
        max_trg = max(trg_lens)

        PAD = self.stoi[self.PAD]
        src_padded = [s + [PAD] * (max_src - len(s)) for s in src_seqs]
        trg_padded = [s + [PAD] * (max_trg - len(s)) for s in trg_seqs]

        src_tensor = torch.LongTensor(src_padded)
        trg_tensor = torch.LongTensor(trg_padded)
        src_mask = (src_tensor != PAD).unsqueeze(1)  # (B,1,SrcLen)
        trg_mask = (trg_tensor != PAD).unsqueeze(1)  # (B,1,TrgLen)

        return src_tensor, trg_tensor, src_mask, trg_mask

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, SeqLen, D)
        x = x + self.pe[:, : x.size(1), :]
        return x

# -----------------------------
# Multi-head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, return_attn=False):
        # q,k,v: (B, SeqLen, D)
        B = q.size(0)
        Q = self.q_lin(q).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, Lq, d_head)
        K = self.k_lin(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, Lk, d_head)
        V = self.v_lin(v).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, Lv, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, Lq, Lk)
        if mask is not None:
            # mask expected shape: (B, 1, 1, Lk) or (B, 1, Lq, Lk)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, Lq, d_head)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B, Lq, D)
        out = self.out_lin(out)
        if return_attn:
            return out, attn
        return out

# -----------------------------
# Feed-forward
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Encoder / Decoder Layers
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: (B, S, D)
        _src = src
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(_src + self.dropout(src2))
        _src = src
        src2 = self.ff(src)
        src = self.norm2(_src + self.dropout(src2))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, return_attn=False):
        _tgt = tgt
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(_tgt + self.dropout(tgt2))

        _tgt = tgt
        # cross-attention
        if return_attn:
            tgt2, attn = self.cross_attn(tgt, memory, memory, mask=memory_mask, return_attn=True)
        else:
            tgt2 = self.cross_attn(tgt, memory, memory, mask=memory_mask)
            attn = None
        tgt = self.norm2(_tgt + self.dropout(tgt2))
        _tgt = tgt
        tgt2 = self.ff(tgt)
        tgt = self.norm3(_tgt + self.dropout(tgt2))
        return tgt, attn

# -----------------------------
# Encoder and Decoder stacks
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads, d_ff, max_len=100, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_embed(src) * math.sqrt(self.tok_embed.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads, d_ff, max_len=100, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, return_attn_layers=False):
        attn_layers = []
        x = self.tok_embed(tgt) * math.sqrt(self.tok_embed.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x, attn = layer(x, memory, tgt_mask, memory_mask, return_attn=return_attn_layers)
            if return_attn_layers:
                attn_layers.append(attn)
        x = self.norm(x)
        if return_attn_layers:
            return x, attn_layers
        return x

# -----------------------------
# Full Seq2Seq Transformer
# -----------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model=128, N=3, n_heads=4, d_ff=256, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, n_heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def make_src_mask(self, src, pad_idx):
        # src: (B, SrcLen)
        # mask for attention over keys: (B, 1, 1, SrcLen)
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt, pad_idx):
        # tgt: (B, T)
        B, T = tgt.size()
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        # subsequent mask
        subsequent = torch.tril(torch.ones((T, T), device=tgt.device)).bool()
        subsequent = subsequent.unsqueeze(0).unsqueeze(1)  # (1,1,T,T)
        mask = pad_mask & subsequent
        return mask

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx, return_attn=False):
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)
        memory = self.encoder(src, src_mask)
        if return_attn:
            dec_out, attn_layers = self.decoder(tgt, memory, tgt_mask, src_mask, return_attn_layers=True)
        else:
            dec_out = self.decoder(tgt, memory, tgt_mask, src_mask)
            attn_layers = None
        out = self.out(dec_out)
        return out, attn_layers

# -----------------------------
# Training and Evaluation
# -----------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx, clip=1.0):
    model.train()
    total_loss = 0
    for src, trg, src_mask, trg_mask in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        # input to decoder is trg[:, :-1], predict trg[:,1:]
        inp = trg[:, :-1]
        target = trg[:, 1:]
        outputs, _ = model(src, inp, pad_idx, pad_idx)
        B, T, V = outputs.size()
        outputs = outputs.reshape(-1, V)
        target = target.reshape(-1)
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg, src_mask, trg_mask in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            inp = trg[:, :-1]
            target = trg[:, 1:]
            outputs, _ = model(src, inp, pad_idx, pad_idx)
            B, T, V = outputs.size()
            outputs = outputs.reshape(-1, V)
            target = target.reshape(-1)
            loss = criterion(outputs, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Greedy decode for inference

def greedy_decode(model, src, src_pad_idx, tgt_sos_idx, tgt_eos_idx, max_len=30, device='cpu'):
    model.eval()
    src = src.unsqueeze(0).to(device)  # make batch 1
    src_mask = model.make_src_mask(src, src_pad_idx)
    memory = model.encoder(src, src_mask)
    ys = torch.LongTensor([[tgt_sos_idx]]).to(device)
    attentions = []
    with torch.no_grad():
        for i in range(max_len):
            out, attn_layers = model.decoder(ys, memory, model.make_tgt_mask(ys, src_pad_idx), src_mask, return_attn_layers=True)
            prob = model.out(out[:, -1, :])  # last step
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.LongTensor([[next_word]]).to(device)], dim=1)
            attentions.append(attn_layers)  # list of layer attns (each: (B,H,1,SrcLen))
            if next_word == tgt_eos_idx:
                break
    return ys.squeeze(0).cpu().tolist(), attentions

# Compute BLEU on a dataset using greedy decode

def compute_bleu(model, dataset, dataloader, device):
    refs = []
    hyps = []
    pad_idx = dataset.stoi[dataset.PAD]
    sos_idx = dataset.stoi[dataset.SOS]
    eos_idx = dataset.stoi[dataset.EOS]
    for src, trg, src_mask, trg_mask in dataloader:
        for i in range(src.size(0)):
            s = src[i]
            decoded_ids, _ = greedy_decode(model, s, pad_idx, sos_idx, eos_idx, max_len=30, device=device)
            # remove sos
            if decoded_ids and decoded_ids[0] == sos_idx:
                decoded_ids = decoded_ids[1:]
            # trim at eos
            if eos_idx in decoded_ids:
                decoded_ids = decoded_ids[:decoded_ids.index(eos_idx)]
            hyp = [dataset.itos[x] for x in decoded_ids]
            trg_ids = trg[i].cpu().tolist()
            # remove sos/eos/pad
            if trg_ids and trg_ids[0] == sos_idx:
                trg_ids = trg_ids[1:]
            if dataset.stoi[dataset.EOS] in trg_ids:
                trg_ids = trg_ids[:trg_ids.index(dataset.stoi[dataset.EOS])]
            ref = [dataset.itos[x] for x in trg_ids]
            # append
            hyps.append(hyp)
            refs.append([ref])
    # compute BLEU
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie) * 100
    return bleu, refs, hyps

# Visualization helpers

def plot_curves(train_vals, valid_vals, ylabel, savepath):
    plt.figure()
    plt.plot(train_vals, label='train')
    plt.plot(valid_vals, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def save_attention_heatmap(attn, src_tokens, title, savepath):
    # attn: (H, SrcLen) averaged over heads if needed
    plt.figure(figsize=(6, 3))
    plt.imshow(attn, aspect='auto')
    plt.yticks(range(attn.shape[0]), [f'h{i}' for i in range(attn.shape[0])])
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=90)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

# -----------------------------
# Main runner
# -----------------------------

def main(args):
    out_dir = 'runs/mt'
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    print('Using device:', device)

    # Dataset
    train_ds = ReversedDataset(n_samples=args.train_samples, min_len=3, max_len=8, vocab_size=args.vocab_size)
    valid_ds = ReversedDataset(n_samples=args.valid_samples, min_len=3, max_len=8, vocab_size=args.vocab_size)
    # Share same vocab indices by copying stoi/itos
    valid_ds.itos = train_ds.itos
    valid_ds.stoi = train_ds.stoi

    pad_idx = train_ds.stoi[train_ds.PAD]
    sos_idx = train_ds.stoi[train_ds.SOS]
    eos_idx = train_ds.stoi[train_ds.EOS]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=valid_ds.collate_fn)

    # Model
    model = TransformerSeq2Seq(src_vocab=len(train_ds.itos), trg_vocab=len(train_ds.itos), d_model=args.d_model, N=args.layers, n_heads=args.heads, d_ff=args.d_ff, max_len=100, dropout=args.dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses, valid_losses = [], []
    best_valid = float('inf')
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_idx)
        valid_loss = evaluate(model, valid_loader, criterion, device, pad_idx)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        t1 = time.time()
        print(f'Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} | Val Loss {valid_loss:.4f} | Time {t1-t0:.1f}s')

        # save best
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))

    total_time = time.time() - start_time
    print('Training finished, total time {:.1f}s'.format(total_time))

    # Plot curves
    plot_curves(train_losses, valid_losses, 'Loss', os.path.join(out_dir, 'curves_mt.png'))

    # Compute BLEU on validation set
    bleu, refs, hyps = compute_bleu(model, valid_ds, valid_loader, device)
    print(f'Validation BLEU: {bleu:.2f}')

    # Save BLEU report image (simple text)
    plt.figure(figsize=(4,2))
    plt.text(0.1, 0.5, f'Corpus BLEU: {bleu:.2f}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bleu_report.png'))
    plt.close()

    # Masks demo: visualize src mask and tgt causal mask for one batch
    for src, trg, src_mask, trg_mask in train_loader:
        src = src[0:1]
        trg = trg[0:1]
        break
    src_mask_example = model.make_src_mask(src, pad_idx).squeeze(0).squeeze(0).cpu().numpy()  # (SrcLen)
    tgt_mask_example = model.make_tgt_mask(trg, pad_idx).squeeze(0).squeeze(0).cpu().numpy()  # (TgtLen, TgtLen)
    plt.figure(figsize=(6,2))
    plt.imshow(src_mask_example, aspect='auto')
    plt.title('Source mask (1=not-pad)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'masks_demo_src.png'))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.imshow(tgt_mask_example, aspect='auto')
    plt.title('Target causal & pad mask (1=allowed)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'masks_demo_tgt.png'))
    plt.close()

    # Decodes table: 10 examples
    n_examples = min(10, args.valid_samples)
    examples = []
    model.eval()
    for i in range(n_examples):
        src_tokens = valid_ds.data[i][0]
        trg_tokens = valid_ds.data[i][1]
        src_ids = torch.LongTensor([valid_ds.stoi[valid_ds.SOS]] + valid_ds.encode(src_tokens) + [valid_ds.stoi[valid_ds.EOS]])
        decoded_ids, attns = greedy_decode(model, src_ids, pad_idx, sos_idx, eos_idx, max_len=30, device=device)
        # convert
        if decoded_ids and decoded_ids[0] == sos_idx:
            decoded_ids = decoded_ids[1:]
        if eos_idx in decoded_ids:
            decoded_ids = decoded_ids[:decoded_ids.index(eos_idx)]
        hyp_tokens = [valid_ds.itos[x] for x in decoded_ids]
        examples.append((src_tokens, trg_tokens, hyp_tokens, attns))

    # Save decodes_table.png
    fig, axs = plt.subplots(n_examples, 1, figsize=(8, n_examples*0.8))
    if n_examples == 1:
        axs = [axs]
    for i, (src, trg, hyp, _) in enumerate(examples):
        txt = f'SRC: {" ".join(src)}  |  TRG: {" ".join(trg)}  |  HYP: {" ".join(hyp)}'
        axs[i].text(0, 0.5, txt, fontsize=10)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'decodes_table.png'))
    plt.close()

    # Attention heatmaps: use last example's attentions
    # attns is a list per decoding step; each element is list of layer attention tensors shape (B,H,1,SrcLen)
    if examples and examples[-1][3]:
        _, _, _, attns = examples[-1]
        # average across decoding steps then heads
        # attns: list of length T (dec steps); each element: list of L layers, each (B,H,1,SrcLen)
        # We'll take the last step
        last_step = attns[-1]
        # last_step is list of layer attn tensors
        for l_idx, layer_attn in enumerate(last_step):
            # layer_attn: (B,H,1,SrcLen)
            at = layer_attn[0].mean(axis=1)  # average across heads -> (H, SrcLen) after mean? careful
            # layer_attn[0] shape: (H,1,SrcLen)
            at = layer_attn[0].squeeze(1).cpu().numpy()  # (H, SrcLen)
            src_tokens = examples[-1][0]
            attn_mean = at.mean(0)  # shape (target_len, source_len)
            save_attention_heatmap(
                attn_mean,
                ['SOS'] + src_tokens + ['EOS'],
                f'Layer{l_idx} cross-attn',
                os.path.join(out_dir, f'attention_layer{l_idx}_headmean.png')
            )


    print('Saved visual artifacts in', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-samples', type=int, default=2000)
    parser.add_argument('--valid-samples', type=int, default=500)
    parser.add_argument('--vocab-size', type=int, default=30)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--d-ff', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    main(args)


# -----------------------------
# requirements.txt (copy these into requirements.txt file)
# -----------------------------
# torch and torchvision versions depend on the system. A safe set:
# torch>=1.12
# torchvision>=0.13
# numpy
# matplotlib
# nltk



