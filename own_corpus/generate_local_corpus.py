"""Generador local generico para el corpus propio -- reemplaza a
generate_qwen_corpus.py con soporte para cualquier modelo local (Qwen3-8B,
DeepSeek-R1-Distill-Llama-8B, etc.) y mas variedad de tecnicas de prompt.

Anadido sobre el generador anterior (ver CONTEXTO.md, sesion de mejora de T2):
  - "persona":   persona de cliente detallada (nombre, edad, ocupacion, un par
                 de rasgos) antes de pedir la review -- mas variedad de "voz"
                 que solo naive/adversarial.
  - "translate": pide la review en otro idioma y que la traduzca al ingles --
                 imita el patron real de un no-nativo escribiendo con ayuda de
                 traduccion automatica, y es una tecnica de evasion documentada
                 (variar la "huella" del generador via traduccion).
  - Longitud mucho mas variable (10-150 palabras) en vez de rangos fijos por
    estilo, para no dejar un patron de longitud reconocible por generador.

Uso:
    python own_corpus/generate_local_corpus.py <repo_id> <carpeta_local> <out_csv> <source_dataset> <generator_label> <n_total> [--strip-think]

    --strip-think: quita el bloque <think>...</think> de la salida (necesario
    para modelos "reasoning" como DeepSeek-R1-Distill, que razonan antes de
    responder).

Reanudable igual que el generador anterior: guarda cada 10 y no repite
combinaciones (category, rating, prompt_style, seed) ya generadas.
"""

import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from features_text import get_device  # noqa: E402

CATEGORIES = [
    "restaurant", "hotel", "coffee_shop", "electronics_store", "mobile_app",
    "skincare_product", "gym", "plumber_service", "clothing_boutique", "saas_tool",
]
CATEGORY_LABEL = {
    "restaurant": "a local restaurant called Bella Vista",
    "hotel": "a hotel called The Riverside Inn",
    "coffee_shop": "a coffee shop called Daily Grind",
    "electronics_store": "an electronics store called CircuitHub",
    "mobile_app": "a productivity mobile app called TaskFlow",
    "skincare_product": "a skincare serum called GlowRenew",
    "gym": "a gym called Iron Peak Fitness",
    "plumber_service": "a plumbing service called QuickFix Plumbing",
    "clothing_boutique": "a clothing boutique called Thread & Co",
    "saas_tool": "a business software tool called Streamline",
}

PERSONAS = [
    "a 24-year-old grad student who types fast and uses casual abbreviations",
    "a 58-year-old retired teacher who writes in complete, formal sentences",
    "a busy parent of three who writes in a hurry between errands",
    "a tech-savvy reviewer who compares everything to competitors",
    "a first-time customer who is pleasantly surprised",
    "a regular customer who has been coming back for years",
    "a skeptical customer who almost didn't try it",
    "an enthusiastic reviewer who uses a lot of exclamation points",
]
LANGUAGES = ["Spanish", "French", "German", "Portuguese", "Italian"]

NAIVE_PROMPT = (
    "Write a {rating}-star customer review for {business}. {length_hint} "
    "Only output the review text, nothing else."
)
ADVERSARIAL_PROMPT = (
    "Write a {rating}-star customer review for {business}. Write it casually, "
    "like a real person typing quickly -- lowercase is fine, minor imperfections "
    "are fine, avoid sounding like an AI wrote it, avoid generic marketing "
    "phrases. {length_hint} Only output the review text, nothing else."
)
PERSONA_PROMPT = (
    "Imagine you are {persona}. Write a {rating}-star customer review for "
    "{business} in your own voice. {length_hint} Only output the review text, "
    "nothing else."
)
TRANSLATE_PROMPT = (
    "Write a {rating}-star customer review for {business} in {language}, "
    "then translate it to English yourself. {length_hint} Only output the "
    "final English review text, nothing else -- not the original, not your "
    "translation process."
)

LENGTH_HINTS = [
    "Keep it very short, 10-20 words.",
    "Keep it short, 20-40 words.",
    "Write a medium-length review, 40-70 words.",
    "Write a detailed review, 80-150 words.",
]

# 50 combinaciones por categoria: reparto entre los 4 estilos y ratings.
PLAN = (
    [("naive", 5)] * 12 + [("naive", 4)] * 5 + [("naive", 2)] * 1 + [("naive", 1)] * 4
    + [("adversarial", 5)] * 7 + [("adversarial", 3)] * 2 + [("adversarial", 1)] * 1
    + [("persona", 5)] * 7 + [("persona", 4)] * 3 + [("persona", 2)] * 1 + [("persona", 1)] * 2
    + [("translate", 5)] * 3 + [("translate", 4)] * 1 + [("translate", 2)] * 1
)


def _load_model(repo_id: str, local_dir: Path):
    device = get_device()
    model_name = str(local_dir) if (local_dir / "model.safetensors.index.json").exists() else repo_id
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device)
    model.eval()
    print(f"[generate_local_corpus] Modelo cargado en '{device}' desde '{model_name}'.")
    return tokenizer, model, device


def _generate(tokenizer, model, device, prompt: str, seed: int, strip_think: bool) -> str:
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=350 if strip_think else 150,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if strip_think and "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip().strip('"')


def main() -> None:
    repo_id, local_dir, out_csv, source_dataset, generator_label, n_total = (
        sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
    )
    strip_think = "--strip-think" in sys.argv
    random.seed(0)

    done = set()
    rows = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev["category"], prev["rating"], prev["prompt_style"], prev["seed"]))
        print(f"Reanudando: {len(done)} ya generadas de una ejecucion anterior.")

    tokenizer, model, device = _load_model(repo_id, local_dir)

    total = min(n_total, len(CATEGORIES) * len(PLAN))
    n_done = len(done)
    combos = [(c, i) for c in CATEGORIES for i in range(len(PLAN))][:total]

    for category, seed in combos:
        style, rating = PLAN[seed]
        key = (category, rating, style, seed)
        if key in done:
            continue
        business = CATEGORY_LABEL[category]
        length_hint = LENGTH_HINTS[hash((category, seed, "len")) % len(LENGTH_HINTS)]
        if style == "naive":
            prompt = NAIVE_PROMPT.format(rating=rating, business=business, length_hint=length_hint)
        elif style == "adversarial":
            prompt = ADVERSARIAL_PROMPT.format(rating=rating, business=business, length_hint=length_hint)
        elif style == "persona":
            persona = PERSONAS[hash((category, seed, "persona")) % len(PERSONAS)]
            prompt = PERSONA_PROMPT.format(persona=persona, rating=rating, business=business, length_hint=length_hint)
        else:
            language = LANGUAGES[hash((category, seed, "lang")) % len(LANGUAGES)]
            prompt = TRANSLATE_PROMPT.format(rating=rating, business=business, language=language, length_hint=length_hint)

        text = _generate(tokenizer, model, device, prompt, seed=hash((category, seed)) % (2**31), strip_think=strip_think)
        rows.append({"category": category, "rating": rating, "prompt_style": style, "seed": seed, "text": text})
        n_done += 1
        if n_done % 10 == 0 or n_done == total:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  {n_done}/{total} guardado")

    df = pd.DataFrame(rows)
    df["is_fake"] = True
    df["source_dataset"] = source_dataset
    df["generator"] = generator_label
    df.to_csv(out_csv, index=False)
    print(f"Total filas: {len(df)}")
    print(f"Guardado en {out_csv}")


if __name__ == "__main__":
    main()
