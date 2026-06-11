"""
Shared utilities for Deep Learning Major Assignment.
Unified data loading, evaluation, and benchmarking functions.
"""
import math, random, time, re, json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────
KAGGLE_TRAIN = Path('/kaggle/input/datasets/neitng/mind-small-rs-kh02/train')
KAGGLE_DEV   = Path('/kaggle/input/datasets/neitng/mind-small-rs-kh02/dev')
LOCAL_TRAIN  = Path('../../news-recommendation-system/data/raw/train')
LOCAL_DEV    = Path('../../news-recommendation-system/data/raw/dev')

TRAIN_DIR = KAGGLE_TRAIN if KAGGLE_TRAIN.exists() else LOCAL_TRAIN
DEV_DIR   = KAGGLE_DEV   if KAGGLE_DEV.exists()   else LOCAL_DEV

WORK_DIR  = Path('/kaggle/working/dl_results') if Path('/kaggle/working').exists() else Path('../results')
MODEL_DIR = WORK_DIR / 'models'
WORK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NEWS_COLS = ['news_id','category','subcategory','title','abstract',
             'url','title_entities','abstract_entities']
BEH_COLS  = ['impression_id','user_id','time','history','impressions']

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── Data Loading ───────────────────────────────────────────────────────
def load_news(path):
    df = pd.read_csv(path/'news.tsv', sep='\t', header=None,
                     names=NEWS_COLS, dtype=str, keep_default_na=False)
    return df.drop_duplicates('news_id').reset_index(drop=True)

def load_behaviors(path):
    return pd.read_csv(path/'behaviors.tsv', sep='\t', header=None,
                       names=BEH_COLS, dtype=str, keep_default_na=False)

def parse_impressions(s):
    out = []
    if not isinstance(s, str) or not s.strip():
        return out
    for tok in s.split():
        if '-' not in tok:
            continue
        i = tok.rfind('-')
        nid = tok[:i]
        try:
            lbl = int(tok[i+1:])
        except ValueError:
            continue
        out.append((nid, lbl))
    return out

# ── Tokenizer (word-level) ─────────────────────────────────────────────
def simple_tokenize(text, max_len=30):
    """Lowercase word tokenization, truncated to max_len."""
    tokens = re.findall(r'[a-z0-9]+', (text or '').lower())
    return tokens[:max_len]

def build_vocab(news_df, min_freq=2, max_len=30):
    """Build word→idx vocab from title+abstract. idx=0 is <PAD>."""
    counter = Counter()
    for _, row in news_df.iterrows():
        text = (row['title'] or '') + ' ' + (row['abstract'] or '')
        counter.update(simple_tokenize(text, max_len=9999))
    vocab = {'<PAD>': 0}
    idx = 1
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

def tokenize_to_ids(text, vocab, max_len=30):
    """Convert text to padded token-id array."""
    tokens = simple_tokenize(text, max_len)
    ids = [vocab.get(t, 0) for t in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# ── Dataset ────────────────────────────────────────────────────────────
class MINDTrainDataset(Dataset):
    """Yields (history_news_ids, pos_news_id, [neg_news_ids])."""
    def __init__(self, beh_df, nid2idx, max_hist=50, neg_k=4, max_rows=None):
        self.samples = []
        self.nid2idx = nid2idx
        self.max_hist = max_hist
        self.neg_k = neg_k
        for _, row in beh_df.iterrows():
            hist_raw = row['history'].split() if row['history'].strip() else []
            hist = [nid2idx[h] for h in hist_raw if h in nid2idx][-max_hist:]
            imps = parse_impressions(row['impressions'])
            pos = [nid2idx[n] for n, l in imps if l == 1 and n in nid2idx]
            neg = [nid2idx[n] for n, l in imps if l == 0 and n in nid2idx]
            if not hist or not pos or not neg:
                continue
            for p in pos[:2]:
                negs = random.choices(neg, k=neg_k)
                self.samples.append((hist, p, negs))
                if max_rows and len(self.samples) >= max_rows:
                    break
            if max_rows and len(self.samples) >= max_rows:
                break

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_train(batch, max_hist=50):
    """Collate: pad history, stack pos and neg candidates."""
    hists, poss, negs_list = zip(*batch)
    B = len(hists)
    H = np.zeros((B, max_hist), dtype=np.int64)
    for i, seq in enumerate(hists):
        seq = seq[-max_hist:]
        H[i, -len(seq):] = np.array(seq, dtype=np.int64)
    P = np.array(poss, dtype=np.int64)
    neg_k = len(negs_list[0])
    N = np.array(negs_list, dtype=np.int64)  # (B, neg_k)
    return torch.from_numpy(H), torch.tensor(P), torch.from_numpy(N)

# ── Unified Evaluation ─────────────────────────────────────────────────
def compute_ranking_metrics(labels, scores):
    """Compute AUC, MRR, nDCG@5, nDCG@10 for a single impression."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    y = labels[order]
    pos = labels.sum()

    # AUC
    if 0 < pos < len(labels):
        pos_s = scores[labels == 1]
        neg_s = scores[labels == 0]
        auc = np.mean(np.array([(p > n) + 0.5*(p == n)
                                for p in pos_s for n in neg_s]))
    else:
        return None  # skip single-class impressions

    # MRR
    rr = 0.0
    for i, v in enumerate(y, start=1):
        if v == 1:
            rr = 1.0 / i
            break

    # nDCG@k
    def ndcg_at_k(y_sorted, k, n_pos):
        dcg  = sum(v / math.log2(i+2) for i, v in enumerate(y_sorted[:k]))
        idcg = sum(1.0 / math.log2(i+2) for i in range(min(int(n_pos), k)))
        return dcg / idcg if idcg > 0 else 0.0

    return {
        'AUC':     float(auc),
        'MRR':     float(rr),
        'nDCG@5':  float(ndcg_at_k(y, 5, pos)),
        'nDCG@10': float(ndcg_at_k(y, 10, pos)),
    }

@torch.no_grad()
def evaluate_model(model, beh_dev, nid2idx, news_token_ids,
                   max_hist=50, device='cpu', max_rows=None):
    """
    Evaluate model on dev behaviors.
    model must have: encode_user(hist_tokens), encode_news(cand_tokens), score_from_vecs(u, c)
    news_token_ids: dict nid_idx -> token_id_array
    """
    model.eval()
    all_metrics = []
    rows = beh_dev if max_rows is None else beh_dev.head(max_rows)

    for _, row in rows.iterrows():
        hist_raw = row['history'].split() if row['history'].strip() else []
        hist = [nid2idx[h] for h in hist_raw if h in nid2idx][-max_hist:]
        imps = parse_impressions(row['impressions'])
        if not hist or len(imps) < 2:
            continue

        cand_nids = [(nid2idx[n], l) for n, l in imps if n in nid2idx]
        if len(cand_nids) < 2:
            continue
        labels = [l for _, l in cand_nids]
        if len(set(labels)) < 2:
            continue

        # Build history tensor
        H = np.zeros((1, max_hist), dtype=np.int64)
        H[0, -len(hist):] = np.array(hist)
        H_t = torch.from_numpy(H).to(device)

        # Build candidate tensor
        cand_idxs = [idx for idx, _ in cand_nids]
        C_t = torch.tensor(cand_idxs, dtype=torch.long, device=device)

        # Score
        scores = model.score_candidates(H_t, C_t).cpu().numpy()
        m = compute_ranking_metrics(labels, scores)
        if m is not None:
            all_metrics.append(m)

    if not all_metrics:
        return {}
    return {k: float(np.mean([x[k] for x in all_metrics]))
            for k in all_metrics[0]}

# ── Benchmarking ───────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_vram(model, input_fn, device='cuda', warmup=3, repeats=10):
    """Measure peak VRAM during forward pass."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    for _ in range(warmup):
        with torch.no_grad():
            input_fn(model)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        with torch.no_grad():
            input_fn(model)
    peak = torch.cuda.max_memory_allocated(device)
    return peak / (1024**2)  # MB

def measure_latency(model, input_fn, device='cuda', warmup=5, repeats=50):
    """Measure inference latency in ms using CUDA events."""
    model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            input_fn(model)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start_event.record()
        with torch.no_grad():
            input_fn(model)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return float(np.mean(times)), float(np.std(times))
