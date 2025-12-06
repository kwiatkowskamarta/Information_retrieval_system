import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================
# Configuration
# =========================
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
SEED = 42
FT_SAVE_SUBDIR = "final_model"
set_seed(SEED)

DEFAULT_TOPIC = """Polyphenols are natural compounds synthesized by plants as a defense mechanism against
sunlight, pests, and other environmental stressors. They are widely distributed in fruits
such as grapes, apples, berries, and pomegranates; in vegetables such as spinach, broccoli,
and onions; as well as in commonly consumed beverages such as tea, coffee, red wine,
and cocoa. They are also present in olive oil, legumes, and a variety of spices. Their
relevance stems from their strong antioxidant activity, which helps protect cells against
oxidative stress. In addition, polyphenols contribute to cardiovascular health, reduce
inflammation, and support the balance of the intestinal microbiota. A common example
is the intense coloration of red wine or blueberries, which is attributed to polyphenols and
simultaneously illustrates their potential health benefits.
"""

# =========================
# Data loading & utils
# =========================
def read_and_label(relevant_csv: str, nonrelevant_csv: str) -> pd.DataFrame:
    """Read CSVs and assign labels: 1 for relevant, 0 for non-relevant."""
    rel = pd.read_csv(relevant_csv)
    nrel = pd.read_csv(nonrelevant_csv)

    for df in (rel, nrel):
        if "Title" not in df.columns:
            alt = [c for c in df.columns if c.lower() == "title"]
            df["Title"] = df[alt[0]] if alt else ""
        if "Abstract" not in df.columns:
            alt = [c for c in df.columns if c.lower() == "abstract"]
            df["Abstract"] = df[alt[0]] if alt else ""
        df["Title"] = df["Title"].fillna("")
        df["Abstract"] = df["Abstract"].fillna("")

    rel["label"] = 1
    nrel["label"] = 0

    # Preserve an ID column if present on relevant side
    id_col = next((c for c in ["PMID", "pmid", "Id", "id"] if c in rel.columns), None)
    if id_col and id_col not in nrel.columns:
        nrel[id_col] = np.nan

    return pd.concat([rel, nrel], ignore_index=True)

def make_doc_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate title and abstract into a single string."""
    return (df["Title"].astype(str).fillna("") + ". " + df["Abstract"].astype(str).fillna("")).str.strip()

def metrics_from_scores(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Compute metrics from positive-class probabilities using a fixed threshold."""
    pred = (proba >= thr).astype(int)
    acc = accuracy_score(y_true, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, proba)
        pr = average_precision_score(y_true, proba)
    except Exception:
        roc, pr = None, None
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": roc, "pr_auc": pr}

def save_json(obj: Dict, path: str):
    """Write a JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, grid: int = 200) -> float:
    """Select the threshold that maximizes F1 on validation scores."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    cands = np.linspace(scores.min(), scores.max(), num=grid)
    best_thr, best_f1 = 0.5, -1.0
    for t in cands:
        pred = (scores >= t).astype(int)
        f1 = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)[2]
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)

# A: TF-IDF + Logistic Regression
def run_tfidf_lr(
    X_train_text: List[str], y_train: np.ndarray,
    X_val_text: List[str], y_val: np.ndarray,
    X_test_text: List[str], y_test: np.ndarray,
    out_dir: str
) -> Dict[str, Dict]:
    """
    Query-aware representation by concatenating:
      'TOPIC: {topic} [SEP] DOC: {title + abstract}'
    Vectorize with TF-IDF and fit Logistic Regression.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
    pipe.fit(X_train_text, y_train)

    results = {}
    for split_name, X, y in [("val", X_val_text, y_val), ("test", X_test_text, y_test)]:
        proba = pipe.predict_proba(X)[:, 1]
        mets = metrics_from_scores(y, proba, 0.5)
        results[split_name] = mets

    save_json(results, os.path.join(out_dir, "tfidf_lr_metrics.json"))
    return results

# B: BioBERT embeddings + Logistic Regression
class PairTextDataset(Dataset):
    """Dataset yielding tokenized (topic, doc) pairs for encoding."""
    def __init__(self, topic: str, docs: List[str], tokenizer, max_length: int = 512):
        self.topic = topic
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.docs)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.topic, self.docs[idx],
            truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

def embed_pairs(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Encode pairs and return pooled embeddings."""
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)
            outs.append(pooled.cpu().numpy())
    return np.vstack(outs)

def run_frozen_biobert_lr(
    topic: str,
    X_train_docs: List[str], y_train: np.ndarray,
    X_val_docs: List[str],   y_val: np.ndarray,
    X_test_docs: List[str],  y_test: np.ndarray,
    out_dir: str, max_length: int = 512, batch_size: int = 16
) -> Dict[str, Dict]:
    """Encode (topic, doc) with a frozen encoder and train Logistic Regression on embeddings."""
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    def build_loader(docs: List[str]) -> DataLoader:
        ds = PairTextDataset(topic, docs, tokenizer, max_length=max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    train_emb = embed_pairs(encoder, build_loader(X_train_docs), device)
    val_emb   = embed_pairs(encoder, build_loader(X_val_docs), device)
    test_emb  = embed_pairs(encoder, build_loader(X_test_docs), device)

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(train_emb, y_train)

    results = {}
    for split_name, X, y in [("val", val_emb, y_val), ("test", test_emb, y_test)]:
        proba = clf.predict_proba(X)[:, 1]
        mets = metrics_from_scores(y, proba, 0.5)
        results[split_name] = mets

    save_json(results, os.path.join(out_dir, "biobert_lr_metrics.json"))
    return results

# B2: BioBERT + Cosine(topic, doc)
def embed_texts_separately(topic: str, docs: List[str], max_length: int = 512, batch_size: int = 16):
    """Encode topic and documents separately with a frozen encoder and return pooled vectors."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        t = tokenizer(topic, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
        t_out = model(**t)
        topic_emb = (t_out.pooler_output if getattr(t_out, "pooler_output", None) is not None
                     else t_out.last_hidden_state.mean(dim=1)).detach().cpu().numpy()

    embs = []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            pooled = (out.pooler_output if getattr(out, "pooler_output", None) is not None
                      else out.last_hidden_state.mean(dim=1))
            embs.append(pooled.detach().cpu().numpy())
    doc_embs = np.vstack(embs)
    return topic_emb, doc_embs

def run_frozen_biobert_cosine(
    topic: str,
    train_docs: List[str], y_train: np.ndarray,
    val_docs: List[str],   y_val: np.ndarray,
    test_docs: List[str],  y_test: np.ndarray,
    out_dir: str, max_length: int = 512, batch_size: int = 16
) -> Dict[str, Dict]:
    """Compute cosine similarity between topic and doc embeddings; threshold chosen on validation."""
    os.makedirs(out_dir, exist_ok=True)

    t_train, train_embs = embed_texts_separately(topic, train_docs, max_length, batch_size)
    t_val,   val_embs   = embed_texts_separately(topic, val_docs,   max_length, batch_size)
    t_test,  test_embs  = embed_texts_separately(topic, test_docs,  max_length, batch_size)

    val_scores  = cosine_similarity(val_embs,  t_val).ravel()
    test_scores = cosine_similarity(test_embs, t_test).ravel()

    thr = find_best_threshold(y_val, val_scores)

    results = {}
    for split_name, scores, y in [("val", val_scores, y_val), ("test", test_scores, y_test)]:
        mets = metrics_from_scores(y, scores, thr)
        results[split_name] = {**mets, "decision_threshold": float(thr)}

    save_json(results, os.path.join(out_dir, "biobert_cosine_metrics.json"))
    return results


# C: Fine-tuned BioBERT
class TopicPairForFT(Dataset):
    """Dataset for fine-tuning: tokenized (topic, doc) with labels."""
    def __init__(self, df: pd.DataFrame, tokenizer, topic_text: str, max_length: int = 512):
        self.labels = df["label"].astype(int).to_numpy()
        self.enc = tokenizer(
            [topic_text] * len(df),
            (df["Title"].fillna("") + ". " + df["Abstract"].fillna("")).tolist(),
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def _compute_ft_metrics(eval_pred):
    """HF Trainer metric wrapper."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    return metrics_from_scores(np.array(labels), probs, 0.5)

def finetune_biobert_and_save(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
    topic: str, out_dir: str, epochs: int, batch_size: int, lr: float, max_length: int
) -> Dict[str, Dict]:
    """
    Fine-tune BioBERT and SAVE to out_dir/final_model/ (model + tokenizer).
    Return metrics (val/test) and write them to finetuned_biobert_metrics.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_dir = os.path.join(out_dir, FT_SAVE_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    ds_train = TopicPairForFT(df_train, tokenizer, topic, max_length)
    ds_val   = TopicPairForFT(df_val, tokenizer, topic, max_length)
    ds_test  = TopicPairForFT(df_test, tokenizer, topic, max_length)

    # Training setup 
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        load_best_model_at_end=False,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to=[]
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, tokenizer=tokenizer)
    trainer.train()

    # Evaluation on val/test
    results = {}
    for split_name, ds in [("val", ds_val), ("test", ds_test)]:
        pred = trainer.predict(ds)
        probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()[:, 1]
        mets = metrics_from_scores(ds.labels, probs, 0.5)
        results[split_name] = mets

    # Persist model + tokenizer
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    save_json(results, os.path.join(out_dir, "finetuned_biobert_metrics.json"))
    return results

def evaluate_loaded_finetuned_biobert(
    finetuned_dir: str,
    df_val: pd.DataFrame, df_test: pd.DataFrame,
    topic: str, max_length: int, out_dir: str
) -> Dict[str, Dict]:
    """
    Load a saved fine-tuned model from finetuned_dir (folder with config.json/tokenizer files/pytorch_model.bin)
    and evaluate on val/test. Save metrics to finetuned_biobert_metrics.json.
    """
    if not (os.path.isdir(finetuned_dir) and os.path.isfile(os.path.join(finetuned_dir, "config.json"))):
        raise FileNotFoundError(f"Saved model not found at: {finetuned_dir}")

    tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_dir)

    ds_val  = TopicPairForFT(df_val,  tokenizer, topic, max_length)
    ds_test = TopicPairForFT(df_test, tokenizer, topic, max_length)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "tmp_eval"),
        per_device_eval_batch_size=32,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to=[]
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)

    results = {}
    for split_name, ds in [("val", ds_val), ("test", ds_test)]:
        pred = trainer.predict(ds)
        probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()[:, 1]
        mets = metrics_from_scores(ds.labels, probs, 0.5)
        results[split_name] = mets

    save_json(results, os.path.join(os.path.dirname(finetuned_dir), "finetuned_biobert_metrics.json"))
    return results

# Main pipeline
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = read_and_label(args.relevant_csv, args.nonrelevant_csv)

    # Splits
    df_train, df_tmp = train_test_split(df, test_size=args.val_size + args.test_size,
                                        random_state=SEED, stratify=df["label"])
    df_val, df_test = train_test_split(df_tmp, test_size=args.test_size/(args.val_size+args.test_size),
                                       random_state=SEED, stratify=df_tmp["label"])

    # Texts
    train_docs = make_doc_text(df_train).tolist()
    val_docs   = make_doc_text(df_val).tolist()
    test_docs  = make_doc_text(df_test).tolist()

    def qa_strings(docs: List[str]) -> List[str]:
        return [f"TOPIC: {args.topic_text} [SEP] DOC: {d}" for d in docs]

    # A) TF-IDF + LR
    tfidf_dir = os.path.join(args.output_dir, "tfidf_lr")
    tfidf_res = run_tfidf_lr(
        qa_strings(train_docs), df_train["label"].to_numpy(),
        qa_strings(val_docs),   df_val["label"].to_numpy(),
        qa_strings(test_docs),  df_test["label"].to_numpy(),
        tfidf_dir
    )

    # B) BioBERT + LR
    frozen_dir = os.path.join(args.output_dir, "biobert_lr")
    frozen_res = run_frozen_biobert_lr(
        args.topic_text,
        train_docs, df_train["label"].to_numpy(),
        val_docs,   df_val["label"].to_numpy(),
        test_docs,  df_test["label"].to_numpy(),
        frozen_dir, max_length=args.max_length, batch_size=args.batch_size*2
    )

    # B2) BioBERT + Cosine
    frozen_cos_dir = os.path.join(args.output_dir, "biobert_cosine")
    frozen_cos_res = run_frozen_biobert_cosine(
        args.topic_text,
        train_docs, df_train["label"].to_numpy(),
        val_docs,   df_val["label"].to_numpy(),
        test_docs,  df_test["label"].to_numpy(),
        frozen_cos_dir, max_length=args.max_length, batch_size=args.batch_size*2
    )

    # C) Fine-tuned BioBERT: train or load
    ft_res = {}
    ft_root = os.path.join(args.output_dir, "finetuned_biobert")
    ft_model_dir_default = os.path.join(ft_root, FT_SAVE_SUBDIR)
    finetuned_dir = args.finetuned_dir or ft_model_dir_default

    if args.skip_finetune:
        # Load and evaluate an already fine-tuned model (if present)
        if os.path.isdir(finetuned_dir):
            ft_res = evaluate_loaded_finetuned_biobert(
                finetuned_dir=finetuned_dir,
                df_val=df_val, df_test=df_test,
                topic=args.topic_text, max_length=args.max_length,
                out_dir=ft_root
            )
            print(f"Loaded fine-tuned model from: {finetuned_dir}")
        else:
            print(f"[INFO] --skip_finetune set, but no model found at: {finetuned_dir}. Skipping fine-tune section.")
    else:
        # Train and save
        ft_res = finetune_biobert_and_save(
            df_train, df_val, df_test,
            args.topic_text, ft_root,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_length=args.max_length
        )
        print(f"Fine-tuned model saved to: {ft_model_dir_default}")

    # Aggregate comparison (JSON only)
    def flat(prefix: str, d: Dict[str, Dict]) -> Dict[str, float]:
        out = {}
        for split in d:
            for k, v in d[split].items():
                out[f"{prefix}_{split}_{k}"] = v
        return out

    comp = {
        **flat("tfidf_lr", tfidf_res),
        **flat("biobert_lr", frozen_res),
        **flat("biobert_cosine", frozen_cos_res),
    }
    if ft_res:
        comp.update(flat("finetuned_biobert", ft_res))

    save_json(comp, os.path.join(args.output_dir, "model_comparison.json"))

    print("\n=== COMPARISON (keys -> value) ===")
    for k, v in comp.items():
        print(f"{k}: {v}")
    print(f"\nSaved JSON comparison to: {os.path.join(args.output_dir, 'model_comparison.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevant_csv", type=str, default="./pubmed_abstracts.csv")
    parser.add_argument("--nonrelevant_csv", type=str, default="./non_relevant.csv")
    parser.add_argument("--output_dir", type=str, default="./ir_comparison")
    parser.add_argument("--topic_text", type=str, default=DEFAULT_TOPIC)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument(
        "--skip_finetune",
        action="store_true",
        help="If set: skip training and attempt to load a saved model from --finetuned_dir (if it exists)."
    )
    parser.add_argument(
        "--finetuned_dir",
        type=str,
        default="",
        help="Path to a saved fine-tuned model (folder with config.json/tokenizer files/pytorch_model.bin). "
             "Default: <output_dir>/finetuned_biobert/final_model"
    )
    args = parser.parse_args()
    main(args)