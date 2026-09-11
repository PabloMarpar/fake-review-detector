"""Señales de texto: T1 (zero-shot cross-perplexity, estilo Binoculars) y T3
(estilometría + LightGBM). T2 (DeBERTa-v3 afinado) vive en train.py, que es
donde se entrena de verdad -- aquí solo las funciones de extracción de señal
que no requieren entrenar nada.

T1 es la señal principal (ver README): en textos cortos como reviews, el
texto por sí solo no es decisivo, así que preferimos un detector zero-shot
bien calibrado sobre datos propios frente a fiarnos de un fine-tune que
puede estar sobreajustado a un único generador (el problema documentado en
RAID: ~96% en el propio generador de entrenamiento, ~7% en uno distinto).
"""

import math
from functools import lru_cache

import spacy
import textstat
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OBSERVER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
PERFORMER_MODEL = "HuggingFaceTB/SmolLM2-360M"


def get_device() -> torch.device:
    """GPU si está disponible (CUDA), CPU si no -- misma llamada en cualquier
    máquina, sin flags ni configuración manual (ver CONTEXTO.md: la máquina de
    trabajo es CPU-only, la de casa tiene GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_binoculars_models():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(PERFORMER_MODEL)
    performer = AutoModelForCausalLM.from_pretrained(PERFORMER_MODEL).to(device)
    observer = AutoModelForCausalLM.from_pretrained(OBSERVER_MODEL).to(device)
    performer.eval()
    observer.eval()
    print(f"[features_text] Binoculars: modelos cargados en '{device}'.")
    return tokenizer, performer, observer, device


@lru_cache(maxsize=1)
def _load_spacy():
    return spacy.load("en_core_web_sm")


def binoculars_score(text: str, max_tokens: int = 512) -> float:
    """Ratio de perplejidad cruzada estilo Binoculars.

    score = perplexity(performer) / cross_perplexity(observer, performer)

    Valores bajos = más probable que sea texto generado por un LLM (el
    "performer" lo predice con mucha más confianza de la que el "observer"
    -- un modelo instruct de la misma familia -- asignaría de forma
    independiente). El umbral no se fija aquí a ciegas: se calibra en
    train.py sobre los propios datos de reviews, no con el umbral publicado
    en el paper original (pensado para texto largo, no para reviews).
    """
    tokenizer, performer, observer, device = _load_binoculars_models()
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encoding["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")

    with torch.no_grad():
        logits_performer = performer(input_ids).logits[0, :-1, :]
        logits_observer = observer(input_ids).logits[0, :-1, :]
        targets = input_ids[0, 1:]

        log_probs_performer = torch.log_softmax(logits_performer, dim=-1)
        token_log_probs = log_probs_performer.gather(1, targets.unsqueeze(1)).squeeze(1)
        perplexity = torch.exp(-token_log_probs.mean())

        probs_observer = torch.softmax(logits_observer, dim=-1)
        cross_entropy = -(probs_observer * log_probs_performer).sum(dim=-1).mean()
        cross_perplexity = torch.exp(cross_entropy)

    return (perplexity / cross_perplexity).item()


def stylometric_features(text: str) -> dict:
    """Features T3: interpretables, alimentan LightGBM y SHAP -- su función
    es explicabilidad y diversidad de ensemble, no ser la señal principal.
    """
    nlp = _load_spacy()
    doc = nlp(text)

    sentences = list(doc.sents)
    sentence_lengths = [len(sent) for sent in sentences] or [0]
    tokens = [t for t in doc if not t.is_space]
    words = [t.text.lower() for t in tokens if t.is_alpha]

    n_tokens = max(len(tokens), 1)
    n_words = max(len(words), 1)

    pos_counts = {}
    for t in tokens:
        pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1
    superlative_count = sum(1 for t in tokens if t.tag_ in ("JJS", "RBS"))

    mean_len = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)

    return {
        "n_words": len(words),
        "sentence_length_mean": mean_len,
        "sentence_length_variance": variance,
        "type_token_ratio": len(set(words)) / n_words,
        "avg_word_length": sum(len(w) for w in words) / n_words,
        "superlative_density": superlative_count / n_tokens,
        "adj_ratio": pos_counts.get("ADJ", 0) / n_tokens,
        "adv_ratio": pos_counts.get("ADV", 0) / n_tokens,
        "noun_ratio": pos_counts.get("NOUN", 0) / n_tokens,
        "verb_ratio": pos_counts.get("VERB", 0) / n_tokens,
        "exclamation_density": text.count("!") / max(len(text), 1),
        "flesch_reading_ease": textstat.flesch_reading_ease(text) if text.strip() else 0.0,
    }
