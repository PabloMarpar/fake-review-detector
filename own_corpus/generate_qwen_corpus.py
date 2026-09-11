"""Genera el Generador B del corpus propio: reviews falsas escritas por un
LLM open-weight moderno y de familia distinta a Claude (Qwen2.5-1.5B-Instruct,
Alibaba, 2024), corriendo local en GPU via `features_text.get_device()`.

Nota sobre descargas en esta red (diagnosticado en sesion, ver CONTEXTO.md):
`from_pretrained` dispara `snapshot_download` con `max_workers=8` por
defecto -- varios ficheros del modelo en paralelo. En una conexion con poco
ancho de banda eso colapsa (las descargas paralelas se pisan entre si hasta
quedarse a 0 bytes/s de forma indefinida), aunque `curl`/`requests` con una
sola conexion secuencial funcionan bien. Por eso aqui se precarga el modelo
con `snapshot_download(..., max_workers=1)` antes de `from_pretrained` --
mas lento por fichero, pero no se cuelga.

Complementa a `claude_generated.csv` (Generador A, autoria manual de Claude
en sesion) -- mismo esquema de categorias/ratings/prompt_style para que
ambos generadores sean comparables entre si y uno se pueda reservar como
held-out frente al otro (ver CONTEXTO.md, seccion "corpus propio").

Uso:
    python own_corpus/generate_qwen_corpus.py

Reanudable: guarda incrementalmente en qwen_generated.csv y no repite
combinaciones (category, rating, prompt_style, seed) ya generadas.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from features_text import get_device  # noqa: E402

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
# Si ya se descargo a mano con _download_model_direct.py (ver ese fichero:
# huggingface_hub se colgaba de forma reproducible en esta red incluso con
# max_workers=1), se carga desde ahi en vez de dejar que from_pretrained
# dispare otra vez snapshot_download.
_LOCAL_MODEL_DIR = Path(__file__).parent / "_qwen_model"
MODEL_NAME = str(_LOCAL_MODEL_DIR) if (_LOCAL_MODEL_DIR / "model.safetensors").exists() else MODEL_REPO
OUT_CSV = Path(__file__).parent / "qwen_generated.csv"

# Mismas 10 categorias que claude_generated.csv, para que ambos generadores
# cubran el mismo espacio de negocios/tonos y sean comparables.
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

# 50 combinaciones por categoria (500 en total) -- Qwen corre local y gratis
# en la GPU, así que aquí es donde conviene generar reserva de verdad para
# uso futuro, no solo el minimo para comparar generadores (a diferencia de
# Claude, que requiere que alguien lo escriba activamente, o de OpenAI, que
# tiene tope diario de cuenta).
PLAN = (
    [("naive", 5)] * 20 + [("naive", 4)] * 8 + [("naive", 2)] * 2 + [("naive", 1)] * 6
    + [("adversarial", 5)] * 10 + [("adversarial", 3)] * 3 + [("adversarial", 1)] * 1
)

NAIVE_PROMPT = (
    "Write a {rating}-star customer review for {business}. "
    "Make it sound genuine and convincing, 30-60 words. "
    "Only output the review text, nothing else."
)
ADVERSARIAL_PROMPT = (
    "Write a {rating}-star customer review for {business}. "
    "Write it casually, like a real person typing quickly -- lowercase is fine, "
    "minor imperfections are fine, avoid sounding like an AI wrote it, avoid "
    "generic marketing phrases. 20-50 words. Only output the review text, nothing else."
)


def _load_model():
    device = get_device()
    if MODEL_NAME == MODEL_REPO:
        snapshot_download(MODEL_NAME, max_workers=1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device)
    model.eval()
    print(f"[generate_qwen_corpus] Modelo cargado en '{device}'.")
    return tokenizer, model, device


def _generate(tokenizer, model, device, prompt: str, seed: int) -> str:
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True -- en esta version de transformers, apply_chat_template
    # sin return_dict devuelve un objeto que no expone .shape de forma fiable
    # (rompia model.generate). Con return_dict=True obtenemos un BatchEncoding
    # normal (input_ids + attention_mask), que se desempaqueta con **inputs.
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().strip('"')


def main() -> None:
    done = set()
    rows = []
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done = set(zip(prev["category"], prev["rating"], prev["prompt_style"], prev["seed"]))
        print(f"Reanudando: {len(done)} ya generadas de una ejecucion anterior.")

    tokenizer, model, device = _load_model()

    total = len(CATEGORIES) * len(PLAN)
    n_done = len(done)
    for category in CATEGORIES:
        business = CATEGORY_LABEL[category]
        for seed, (style, rating) in enumerate(PLAN):
            key = (category, rating, style, seed)
            if key in done:
                continue
            prompt_template = NAIVE_PROMPT if style == "naive" else ADVERSARIAL_PROMPT
            prompt = prompt_template.format(rating=rating, business=business)
            text = _generate(tokenizer, model, device, prompt, seed=hash((category, seed)) % (2**31))
            rows.append({
                "category": category, "rating": rating, "prompt_style": style,
                "seed": seed, "text": text,
            })
            n_done += 1
            if n_done % 10 == 0 or n_done == total:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  {n_done}/{total} guardado")

    df = pd.DataFrame(rows)
    df["is_fake"] = True
    df["source_dataset"] = "own_corpus_qwen2.5_1.5b"
    df["generator"] = "qwen2.5-1.5b-instruct"
    df.to_csv(OUT_CSV, index=False)
    print(f"Total filas: {len(df)}")
    print(f"Guardado en {OUT_CSV}")


if __name__ == "__main__":
    main()
