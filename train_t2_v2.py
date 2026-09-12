"""T2 v2: cabeza adversarial de "que generador es" con Gradient Reversal
Layer (GRL, Ganin & Lempitsky 2015), inspirado en "Breaking the Generator
Barrier: Disentangled Representation for Generalizable AI-Text Detection"
(ACL 2026, arXiv 2604.13692) -- version simplificada de su idea central: en
vez de su doble cuello de botella + regularizacion cross-view + adaptacion
guiada por discriminador, aqui una sola cabeza auxiliar predice el
generador (humano / claude / qwen2.5 / qwen3 / ...) a partir de la misma
representacion que usa el clasificador humano-vs-IA, pero con el gradiente
invertido antes de llegar al encoder compartido. El encoder se entrena para
que esa cabeza NO pueda distinguir generadores -- forzandolo a dejar de
usar tics especificos de Claude/Qwen y aprender algo mas parecido a
"IA-nidad" generica, que es justo el fallo que diagnostico el T2 baseline
(loss de train a 0.003, val ~1.0, pero solo 0.50 en el generador held-out).

Se compara honestamente contra `train.py t2` (mismo protocolo de held-out,
mismo _tpr_at_fpr) -- si no mejora, se documenta igual.

Uso:
    python train_t2_v2.py [--model base|large] [--batch-size N] [--epochs N]

Held-out: OpenAI (gpt-4o/-mini) y, si existe, DeepSeek-R1-Distill-Llama-8B
-- ninguno de los dos entra en train ni en la cabeza adversarial de
generador (si entraran, el propio truco de disentanglement no significaria
nada: la prueba es si generaliza a un generador que ni siquiera la cabeza
auxiliar conoce).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from features_text import get_device
from train import DATA_CSV, MODELS_DIR, OWN_CORPUS_DIR, _save_metrics, _tpr_at_fpr

BASE_DIR = Path(__file__).parent

TRAIN_GENERATOR_FILES = {
    "claude_generated.csv": "claude",
    "qwen_generated.csv": "qwen2.5",
    "qwen3_generated.csv": "qwen3",
}
HELD_OUT_FILES = {
    "openai_generated.csv": "openai",
    "deepseek_generated.csv": "deepseek",
}
AUX_LOSS_WEIGHT = 0.3


def _resolve_model_name(size: str) -> str:
    repo = f"microsoft/deberta-v3-{size}"
    local_dir = BASE_DIR / "_model_cache" / f"deberta-v3-{size}"
    marker = "pytorch_model.bin"
    return str(local_dir) if (local_dir / marker).exists() else repo


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DisentangledDetector(torch.nn.Module):
    def __init__(self, model_name: str, num_generators: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(0.15)
        self.classifier = torch.nn.Linear(hidden, 2)
        self.generator_head = torch.nn.Linear(hidden, num_generators)

    @staticmethod
    def _pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, grl_lambda: float):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self._pool(out.last_hidden_state, attention_mask))
        cls_logits = self.classifier(pooled)
        gen_logits = self.generator_head(GradientReversalFunction.apply(pooled, grl_lambda))
        return cls_logits, gen_logits


class ReviewDataset(Dataset):
    def __init__(self, encodings: dict, cls_labels: torch.Tensor, gen_labels: torch.Tensor):
        self.encodings = encodings
        self.cls_labels = cls_labels
        self.gen_labels = gen_labels

    def __len__(self) -> int:
        return len(self.cls_labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["cls_labels"] = self.cls_labels[idx]
        item["gen_labels"] = self.gen_labels[idx]
        return item


def _build_datasets():
    baseline = pd.read_csv(DATA_CSV)
    ott = baseline[baseline["source_dataset"] == "ott_2013"]
    salminen = baseline[baseline["source_dataset"] == "salminen_2022_gpt2"]

    frames = [pd.DataFrame({"text": ott["text"], "label": 0, "generator": "human"})]
    for fname, gname in TRAIN_GENERATOR_FILES.items():
        path = OWN_CORPUS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            frames.append(pd.DataFrame({"text": df["text"], "label": 1, "generator": gname}))
    data = pd.concat(frames, ignore_index=True).dropna(subset=["text"])

    gen_to_id = {g: i for i, g in enumerate(sorted(data["generator"].unique()))}
    data["generator_id"] = data["generator"].map(gen_to_id)

    train_df, val_df = train_test_split(data, test_size=0.15, random_state=0, stratify=data["generator"])

    held_out = {}
    for fname, gname in HELD_OUT_FILES.items():
        path = OWN_CORPUS_DIR / fname
        if path.exists():
            held_out[gname] = pd.read_csv(path)

    held_out_human = salminen[~salminen["is_fake"]]["text"]
    return train_df, val_df, held_out, held_out_human, salminen, gen_to_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "large"], default="base")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--aux-weight", type=float, default=AUX_LOSS_WEIGHT)
    args = parser.parse_args()
    batch_size = args.batch_size or (16 if args.model == "base" else 8)

    device = get_device()
    model_name = _resolve_model_name(args.model)
    train_df, val_df, held_out, held_out_human, salminen, gen_to_id = _build_datasets()

    print(f"Modelo: {model_name}")
    print(f"Train: {len(train_df)} {dict(train_df['generator'].value_counts())}")
    print(f"Val: {len(val_df)}")
    print(f"Generadores en train (id): {gen_to_id}")
    for gname, df in held_out.items():
        print(f"Held-out '{gname}': {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DisentangledDetector(model_name, num_generators=len(gen_to_id)).to(device)

    def _tokenize(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=256, return_tensors="pt")

    train_enc = _tokenize(train_df["text"])
    train_ds = ReviewDataset(
        train_enc, torch.tensor(train_df["label"].values), torch.tensor(train_df["generator_id"].values)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    gen_loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    step_count = 0
    for epoch in range(args.epochs):
        total_cls_loss, total_gen_loss = 0.0, 0.0
        for batch in train_loader:
            # rampa estandar de DANN: grl_lambda va de 0 a 1 a lo largo del
            # entrenamiento -- al principio el encoder aprende la tarea
            # principal sin presion adversarial, luego se va introduciendo.
            progress = step_count / max(total_steps, 1)
            grl_lambda = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            cls_logits, gen_logits = model(batch["input_ids"], batch["attention_mask"], grl_lambda)
            cls_loss = cls_loss_fn(cls_logits, batch["cls_labels"])
            gen_loss = gen_loss_fn(gen_logits, batch["gen_labels"])
            (cls_loss + args.aux_weight * gen_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_cls_loss += cls_loss.item()
            total_gen_loss += gen_loss.item()
            step_count += 1
        print(
            f"  epoch {epoch + 1}/{args.epochs} cls_loss={total_cls_loss / len(train_loader):.4f} "
            f"gen_loss_adversarial={total_gen_loss / len(train_loader):.4f} (sube = el encoder "
            f"esta consiguiendo despistar a la cabeza de generador, que es el objetivo)"
        )

    model_dir = MODELS_DIR / f"t2_adversarial_{args.model}"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    tokenizer.save_pretrained(model_dir)
    (model_dir / "gen_to_id.json").write_text(json.dumps(gen_to_id))

    def _predict(texts) -> np.ndarray:
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size * 2):
                chunk = list(texts[i:i + batch_size * 2])
                enc = tokenizer(chunk, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
                logits, _ = model(enc["input_ids"], enc["attention_mask"], grl_lambda=0.0)
                scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        model.train()
        return np.array(scores)

    val_scores, val_y = _predict(val_df["text"].tolist()), val_df["label"].values
    metric_key = f"t2_adversarial_{args.model}_aux{args.aux_weight}"
    metrics = {
        metric_key: {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "generators_in_train": list(gen_to_id.keys()),
            "tpr_at_1pct_fpr_val": _tpr_at_fpr(val_y, val_scores, 0.01),
            "tpr_at_5pct_fpr_val": _tpr_at_fpr(val_y, val_scores, 0.05),
        }
    }

    for gname, ai_df in held_out.items():
        human_sample = held_out_human.sample(n=min(300, len(held_out_human)), random_state=0)
        texts = list(ai_df["text"]) + list(human_sample)
        y = np.array([1] * len(ai_df) + [0] * len(human_sample))
        scores = _predict(texts)
        metrics[metric_key][f"n_held_out_{gname}"] = len(ai_df)
        metrics[metric_key][f"tpr_at_1pct_fpr_{gname}"] = _tpr_at_fpr(y, scores, 0.01)
        metrics[metric_key][f"tpr_at_5pct_fpr_{gname}"] = _tpr_at_fpr(y, scores, 0.05)

    if len(salminen):
        sal_scores = _predict(salminen["text"].tolist())
        sal_y = salminen["is_fake"].astype(int).values
        metrics[metric_key]["tpr_at_1pct_fpr_salminen_2022_gpt2"] = _tpr_at_fpr(sal_y, sal_scores, 0.01)
        metrics[metric_key]["tpr_at_5pct_fpr_salminen_2022_gpt2"] = _tpr_at_fpr(sal_y, sal_scores, 0.05)

    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
