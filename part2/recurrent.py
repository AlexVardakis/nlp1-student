"""
Task 4: LSTM encoder experiments on IMDB (frozen vs fine-tuned vs random embeddings).

Reuses the tutorial code wherever it fits:
- clslib.LSTMEncoder and clslib.SoftmaxRegressor (the model),
- experiment.LabelledCorpus, experiment.collate_labelled_texts, experiment.training_one_epoch (the data and the epoch loop),
- experiment.load (to reload a saved fold model from model.json + model.pt).

What this module adds, because a controlled A/B/C comparison needs it and experiment.train lacks it:
- TruncatedTokenizer: keep the first `max_tokens` tokens of every review (train, early-stopping and evaluation alike).
- corpus_embeddings: keep only the fastText rows for word types that occur in IMDB, so fine-tuning updates
  ~10^5 rows instead of 10^6 (the full table with AdamW would need ~5 GB). Tokenisation is unchanged (checked).
- random_embedding_matrix: same vocabulary, same shape and same average squared norm as fastText, so only
  "pretrained or not" differs between experiments B and C.
- train_lstm: early stopping on validation loss with patience, then restores the best checkpoint.
- run_cv_system: runs one system on the Project 1 cross-validation folds and saves results.json in the same
  schema as the Task 3 systems, so the Bayesian correlated t-test code in the notebook works unchanged.
"""
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from . import clslib
from . import experiment


# ---------------------------------------------------------------------------
# Tokenisation and embeddings
# ---------------------------------------------------------------------------

class TruncatedTokenizer(clslib.Tokenizer):
    """Wraps a tokenizer and keeps only the first `max_tokens` tokens of each text."""

    def __init__(self, base: clslib.Tokenizer, max_tokens: int):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.base = base
        self.max_tokens = max_tokens
        self.vocab = base.vocab

    def tokens(self, text: str) -> list:
        return list(self.base.tokens(text)[:self.max_tokens])

    def ids(self, text: str) -> list:
        return list(self.base.ids(text)[:self.max_tokens])


def truncation_report(base_tokenizer: clslib.Tokenizer, texts, max_tokens: int) -> dict:
    """How much of each review a `max_tokens` cut removes (lengths are in tokenizer ids, i.e. what the LSTM sees)."""
    lengths = np.array([len(base_tokenizer.ids(t)) for t in texts])
    kept = np.minimum(lengths, max_tokens)
    return {
        "num_texts": int(len(lengths)),
        "max_tokens": int(max_tokens),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "p90_length": float(np.percentile(lengths, 90)),
        "max_length": int(lengths.max()),
        "pct_truncated": float(100 * np.mean(lengths > max_tokens)),
        "pct_tokens_kept": float(100 * kept.sum() / lengths.sum()),
    }


def _embeddings_from_rows(words, vectors, pad_token, unk_token) -> clslib.PretrainedEmbeddings:
    """Build a PretrainedEmbeddings object from a word list and a matching matrix (no file parsing)."""
    emb = object.__new__(clslib.PretrainedEmbeddings)
    emb.vocab = clslib.Vocabulary(init=list(words), default=unk_token, frozen=True)
    emb.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    emb.num_dims = int(emb.vectors.shape[1])
    emb.unk_token = unk_token
    emb.pad_token = pad_token
    assert emb.vectors.shape[0] == len(emb.vocab), "Mismatch between vocab and vectors"
    return emb


def corpus_embeddings(emb_model: clslib.PretrainedEmbeddings, tokenizer: clslib.Tokenizer, texts) -> clslib.PretrainedEmbeddings:
    """
    Restrict `emb_model` to <pad>, <unk> and the vocabulary words that `tokenizer` (built on emb_model.vocab)
    produces on `texts`. Pass every text the model will ever see (train, validation, test): this uses no labels,
    and it keeps the pretrained vector of every word that the full table would have given.
    Row order follows the original table; <pad> stays id 0 and <unk> id 1 (its vector stays the full-table mean).
    """
    words = emb_model.vocab.words
    if words[0] != emb_model.pad_token or words[1] != emb_model.unk_token:
        raise ValueError("Expected <pad> at id 0 and <unk> at id 1, as built by clslib.PretrainedEmbeddings")
    seen = set()
    for text in texts:
        seen.update(tokenizer.tokens(text))
    keep_ids = [0, 1] + [i for i in range(2, len(words)) if words[i] in seen]
    return _embeddings_from_rows(
        words=[words[i] for i in keep_ids],
        vectors=emb_model.vectors[keep_ids],
        pad_token=emb_model.pad_token,
        unk_token=emb_model.unk_token,
    )


def check_same_tokenization(full_tokenizer, full_emb, small_tokenizer, small_emb, texts) -> int:
    """
    Assert that the reduced table gives every text the same word sequence and the same vectors as the full table.
    Returns the number of texts checked.
    """
    n = 0
    for text in texts:
        full_ids, small_ids = full_tokenizer.ids(text), small_tokenizer.ids(text)
        full_words = [full_emb.vocab.word(i) for i in full_ids]
        small_words = [small_emb.vocab.word(i) for i in small_ids]
        if full_words != small_words:
            raise AssertionError(f"Tokenisation differs on: {text[:200]!r}")
        if not np.array_equal(full_emb.vectors[full_ids], small_emb.vectors[small_ids]):
            raise AssertionError(f"Vectors differ on: {text[:200]!r}")
        n += 1
    return n


def random_embedding_matrix(pretrained_vectors: np.ndarray, seed: int, pad_idx: int = 0) -> np.ndarray:
    """
    Random table with the same vocabulary and shape as `pretrained_vectors`.
    Entries are N(0, s^2) with s = root-mean-square of the pretrained word vectors (rows 2+, i.e. not <pad>/<unk>),
    so random and pretrained vectors have the same average squared norm: only the pretraining differs, not the scale.
    The <pad> row is zero, as in the pretrained table.
    """
    words_part = pretrained_vectors[2:].astype(np.float64)
    s = float(np.sqrt(np.mean(words_part ** 2)))
    rng = np.random.default_rng(seed)
    matrix = (rng.standard_normal(pretrained_vectors.shape) * s).astype(np.float32)
    matrix[pad_idx] = 0.0
    return matrix


# ---------------------------------------------------------------------------
# Training with early stopping and best-checkpoint restore
# ---------------------------------------------------------------------------

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(ids, labels, batch_size, pad_id, shuffle, seed=None):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        experiment.LabelledCorpus(ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: experiment.collate_labelled_texts(batch, padding_value=pad_id),
    )


def predict_logprobs_ids(model, ids, batch_size, pad_id, device) -> np.ndarray:
    """Log-probabilities [num_texts, num_classes] for already-tokenised texts, in input order."""
    loader = _make_loader(ids, [0] * len(ids), batch_size, pad_id, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for X_batch, _ in loader:
            out.append(model(X_batch.to(device)).cpu())
    return torch.cat(out).numpy()


def _scores(logprobs: np.ndarray, labels) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    # NLLLoss with mean reduction, as in training
    loss = float(-logprobs[np.arange(len(labels)), labels].astype(np.float64).mean())
    preds = logprobs.argmax(axis=1)
    return {"loss": loss, "accuracy": float(np.mean(preds == labels)), "predictions": preds}


def build_model(embedding_matrix: np.ndarray, freeze_embeddings: bool, pad_id: int, num_classes: int,
                enc_kwargs: dict, regressor_kwargs: dict, init_seed: int) -> clslib.SoftmaxRegressor:
    """
    LSTM encoder + MLP classifier from the tutorial classes.
    The embedding matrix is copied, so fine-tuning never changes the caller's array (or another fold's model).
    Seeding right before construction gives the LSTM and the classifier the same initial weights in A, B and C
    (nn.Embedding.from_pretrained draws no random numbers).
    """
    torch.manual_seed(init_seed)
    encoder = clslib.LSTMEncoder(
        embedding_matrix=torch.tensor(np.array(embedding_matrix, dtype=np.float32, copy=True)),
        padding_idx=pad_id,
        freeze_embeddings=freeze_embeddings,
        **enc_kwargs,
    )
    return clslib.SoftmaxRegressor(encoder=encoder, num_classes=num_classes, **regressor_kwargs)


def train_lstm(model, train_ids, train_labels, es_ids, es_labels, pad_id, opt_kwargs, shuffle_seed, device, log=print):
    """
    Train with AdamW; after every epoch compute the loss on the early-stopping set.
    Stop when it has not improved for `patience` epochs in a row (or after `max_epochs`),
    then load the weights of the best epoch back into `model`.
    Returns (model, history, best_epoch, epochs_run); epochs are 1-based.
    """
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_kwargs["learning_rate"],
        weight_decay=opt_kwargs["weight_decay"],
    )
    train_loader = _make_loader(train_ids, train_labels, opt_kwargs["batch_size"], pad_id, shuffle=True, seed=shuffle_seed)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, opt_kwargs["max_epochs"] + 1):
        t0 = time.time()
        train_metrics = experiment.training_one_epoch(model, train_loader, optimizer, criterion, device)
        es = _scores(predict_logprobs_ids(model, es_ids, opt_kwargs["batch_size"], pad_id, device), es_labels)
        improved = es["loss"] < best_loss - opt_kwargs.get("min_delta", 0.0)
        if improved:
            best_loss, best_epoch = es["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],   # running mean over the epoch, dropout on
            "train_accuracy": train_metrics["accuracy"],
            "es_loss": es["loss"],
            "es_accuracy": es["accuracy"],
            "improved": improved,
            "seconds": time.time() - t0,
        })
        log(f"  epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"early-stop loss {es['loss']:.4f} acc {es['accuracy']:.4f}{' *' if improved else ''} | "
            f"{time.time() - t0:.0f}s")
        if epochs_without_improvement >= opt_kwargs["patience"]:
            break

    if best_state is None:
        raise RuntimeError("The early-stopping loss was never finite (NaN/inf), so there is no checkpoint to restore")
    model.load_state_dict(best_state)
    return model, history, best_epoch, len(history)


# ---------------------------------------------------------------------------
# Cross-validation on the Project 1 folds
# ---------------------------------------------------------------------------

def _fold_metrics(gold, predicted) -> dict:
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "precision": float(precision_score(gold, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(gold, predicted, average="macro", zero_division=0)),
        "f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
    }


def run_cv_system(
    name: str,
    results_folder: Path,
    cv_sets_per_seed: list,
    seeds: list,
    es_pairs: list,
    tokenizer: clslib.Tokenizer,
    emb_model: clslib.PretrainedEmbeddings,
    embedding_init: str,
    freeze_embeddings: bool,
    enc_kwargs: dict,
    regressor_kwargs: dict,
    opt_kwargs: dict,
    num_classes: int = 2,
    device=None,
    folds=None,
    log=print,
) -> list:
    """
    Train and evaluate one LSTM system on every (seed, fold) of the given cross-validation splits.

    cv_sets_per_seed: one dict per seed from part1.experiment.cross_validation_split (same folds as Project 1 and Task 3)
    es_pairs: (text, label) pairs used ONLY for early stopping (we use the fixed 2,000-review validation set),
        so the cross-validation fold that is scored never influences which epoch is kept
    tokenizer: the tokenizer (already truncated) whose vocabulary is emb_model.vocab
    embedding_init: 'pretrained' (emb_model.vectors) or 'random' (random_embedding_matrix, same shape and scale)
    folds: optionally a subset of fold ids (e.g. [0] for a timing check); results.json is only written for full runs

    Saves per fold: model.json + model.pt (reload with part2.experiment.load(path, emb_model)) and fold_results.json.
    Saves per seed: results.json with the same keys as the Task 3 systems plus training curves and probabilities.
    Finished folds and seeds are loaded, not retrained, so an interrupted run resumes where it stopped.
    Returns the per-seed result dicts.
    """
    if embedding_init not in ("pretrained", "random"):
        raise ValueError(f"Unknown embedding_init: {embedding_init}")
    if len(seeds) != len(cv_sets_per_seed):
        raise ValueError("Give one cross-validation split per seed")
    if list(tokenizer.vocab.words) != list(emb_model.vocab.words):
        raise ValueError("The tokenizer's vocabulary is not emb_model's vocabulary: token ids would point at the wrong vectors")
    device = default_device() if device is None else torch.device(device)
    pad_id = emb_model.vocab.index(emb_model.pad_token)
    max_tokens = getattr(tokenizer, "max_tokens", None)
    es_ids = [tokenizer.ids(t) for t, _ in es_pairs]
    es_labels = [int(y) for _, y in es_pairs]

    config = {
        "system": name,
        "enc_type": "lstm",
        "embedding_init": embedding_init,
        "max_tokens": max_tokens,
        "vocab_size": len(emb_model.vocab),
        "emb_dim": int(emb_model.vectors.shape[1]),
        "enc_kwargs": {**enc_kwargs, "freeze_embeddings": freeze_embeddings, "padding_idx": pad_id},
        "regressor_kwargs": {**regressor_kwargs, "num_classes": num_classes},
        "opt_kwargs": dict(opt_kwargs),
        "early_stopping_set": f"{len(es_pairs)} fixed validation reviews (not the scored fold)",
    }

    runs = []
    for seed, cv_sets in zip(seeds, cv_sets_per_seed):
        run_folder = Path(results_folder) / name / f"seed-{seed}"
        results_file = run_folder / "results.json"
        # Never mix runs made with different settings under one system name
        config_file = run_folder / "config.json"
        if config_file.exists():
            saved_config = json.loads(config_file.read_text(encoding="utf-8"))
            if saved_config != json.loads(json.dumps(config)):
                raise ValueError(f"{config_file} was made with different settings; use a new system name or delete the folder.\n"
                                 f"saved:   {saved_config}\ncurrent: {config}")
        else:
            run_folder.mkdir(parents=True, exist_ok=True)
            config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        if results_file.exists() and folds is None:
            log(f"[{name}] seed {seed}: loading {results_file}")
            runs.append(json.loads(results_file.read_text(encoding="utf-8")))
            continue

        fold_ids = sorted(cv_sets) if folds is None else list(folds)
        per_fold = OrderedDict()
        for fold in fold_ids:
            fold_folder = run_folder / f"fold-{fold}"
            fold_file = fold_folder / "fold_results.json"
            if fold_file.exists():
                log(f"[{name}] seed {seed} fold {fold}: loading {fold_file}")
                per_fold[fold] = json.loads(fold_file.read_text(encoding="utf-8"))
                continue
            fold_folder.mkdir(parents=True, exist_ok=True)
            log(f"[{name}] seed {seed} fold {fold}: training on {device}")

            train_pairs = cv_sets[fold]["training"]
            eval_pairs = cv_sets[fold]["validation"]
            train_ids = [tokenizer.ids(t) for t, _ in train_pairs]
            train_labels = [int(y) for _, y in train_pairs]
            eval_ids = [tokenizer.ids(t) for t, _ in eval_pairs]
            eval_labels = [int(y) for _, y in eval_pairs]
            lengths = [len(x) for x in train_ids + eval_ids + es_ids]
            assert min(lengths) >= 1, "Empty token sequence (pack_padded_sequence needs length >= 1)"
            assert max_tokens is None or max(lengths) <= max_tokens, "Truncation was not applied"

            # Same seed for A, B and C on a given (seed, fold): same LSTM/classifier init and batch order
            run_seed = 1000 * seed + fold
            if embedding_init == "pretrained":
                matrix = emb_model.vectors
            else:
                matrix = random_embedding_matrix(emb_model.vectors, seed=run_seed, pad_idx=pad_id)
            model = build_model(matrix, freeze_embeddings, pad_id, num_classes, enc_kwargs, regressor_kwargs, init_seed=run_seed)
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            t0 = time.time()
            model, history, best_epoch, epochs_run = train_lstm(
                model, train_ids, train_labels, es_ids, es_labels, pad_id, opt_kwargs,
                shuffle_seed=run_seed, device=device, log=log,
            )
            train_seconds = time.time() - t0

            # Score the restored best model on the held-out fold
            logprobs = predict_logprobs_ids(model, eval_ids, opt_kwargs["batch_size"], pad_id, device)
            preds = logprobs.argmax(axis=1)
            metrics = _fold_metrics(eval_labels, preds)
            log(f"[{name}] seed {seed} fold {fold}: best epoch {best_epoch}/{epochs_run}, "
                f"fold accuracy {metrics['accuracy']:.4f} ({train_seconds / 60:.1f} min)")

            model_path = fold_folder / "model"
            with open(f"{model_path}.json", "w", encoding="utf-8") as f:
                json.dump({k: config[k] for k in ("enc_kwargs", "regressor_kwargs", "enc_type")}, f, indent=2)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"{model_path}.pt")

            per_fold[fold] = {
                "model": str(model_path),
                **metrics,
                "predictions": preds.tolist(),
                "prob_positive": np.exp(logprobs[:, 1]).tolist(),
                "validation_ids": [int(i) for i in cv_sets[fold]["validation_ids"]],
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "history": history,
                "num_trainable_params": int(num_trainable),
                "train_seconds": train_seconds,
            }
            fold_file.write_text(json.dumps(per_fold[fold]), encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if folds is not None:
            runs.append({"seed": seed, "partial_folds": per_fold})
            continue

        # Same keys as the Task 3 results.json (fold keys become strings in JSON), plus Task 4 extras
        result = {
            **config,
            "seed": seed,
            "num_folds": len(per_fold),
            "models": {f: r["model"] for f, r in per_fold.items()},
            "accuracies": {f: r["accuracy"] for f, r in per_fold.items()},
            "precisions": {f: r["precision"] for f, r in per_fold.items()},
            "recalls": {f: r["recall"] for f, r in per_fold.items()},
            "f1s": {f: r["f1"] for f, r in per_fold.items()},
            "predictions": {f: r["predictions"] for f, r in per_fold.items()},
            "validation_ids": {f: r["validation_ids"] for f, r in per_fold.items()},
            "prob_positive": {f: r["prob_positive"] for f, r in per_fold.items()},
            "best_epochs": {f: r["best_epoch"] for f, r in per_fold.items()},
            "epochs_run": {f: r["epochs_run"] for f, r in per_fold.items()},
            "histories": {f: r["history"] for f, r in per_fold.items()},
            "num_trainable_params": next(iter(per_fold.values()))["num_trainable_params"],
        }
        run_folder.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps(result), encoding="utf-8")
        runs.append(json.loads(results_file.read_text(encoding="utf-8")))
    return runs
