"""Genera el Generador C del corpus propio: reviews falsas escritas con la
API de OpenAI (gpt-4o-mini) -- tercera familia de LLM distinta a Claude
(Generador A) y Qwen (Generador B), y la mas representativa de "atacante
usando un LLM moderno de verdad" ya que es la API que mas se menciona en
los foros donde la gente comparte tecnicas para esto (ver CONTEXTO.md).
Se uso gpt-4o-mini en vez de gpt-4o porque la cuenta nueva de OpenAI tiene
un limite de solo 50 peticiones/dia a gpt-4o (se agoto en esta sesion);
gpt-4o-mini tiene un limite diario mucho mas alto en cuentas nuevas y el
coste es igualmente trivial para este volumen.

Tres prompt_style, en orden de sofisticacion creciente -- basados en
tecnicas reales encontradas en hilos de BlackHatWorld sobre como la gente
usa ChatGPT para fake reviews, no en suposiciones nuestras:
  - "naive":       prompt directo, "escribe una review de N estrellas para X".
  - "adversarial": framing "imaginary review" / roleplay + instruccion
                    explicita de sonar casual, con alguna imperfeccion,
                    evitando frases de marketing genericas -- la tecnica de
                    "usa la palabra imaginary para saltar el filtro" y
                    "cambia el tono" que se comparte en esos foros.
  - "fewshot":      se le pega una review REAL (sampleada de
                    reviews_baseline.csv) como referencia de estilo y se le
                    pide reescribirla para otro negocio con otro tono --
                    la tecnica mas avanzada del foro ("scrape real reviews,
                    mention as reference, rewrite in different tone").

Presupuesto: tiene un tope duro en USD (ver BUDGET_CAP_USD) calculado con
el uso REAL de tokens que devuelve la API en cada respuesta, no una
estimacion a priori -- se para antes de acercarse al limite real fijado en
el dashboard de OpenAI (que es el backstop de verdad, esto es una capa
extra de seguridad por si la estimacion de precio de aqui esta desfasada).

Uso:
    python own_corpus/generate_openai_corpus.py
"""

import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

OUT_CSV = Path(__file__).parent / "openai_generated.csv"
MODEL = "gpt-4o-mini"

# Precios conservadores (sobreestimados a proposito) en USD por token, solo
# para el tope de seguridad -- el limite real esta en el dashboard.
PRICE_IN_PER_TOK = 3.0 / 1_000_000
PRICE_OUT_PER_TOK = 12.0 / 1_000_000
BUDGET_CAP_USD = 3.0

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

# 30 combinaciones por categoria: reparto entre los 3 estilos y ratings.
PLAN = (
    [("naive", 5)] * 6 + [("naive", 4)] * 2 + [("naive", 1)] * 2
    + [("adversarial", 5)] * 5 + [("adversarial", 3)] * 2 + [("adversarial", 1)] * 1
    + [("fewshot", 5)] * 7 + [("fewshot", 4)] * 3 + [("fewshot", 2)] * 2
)

NAIVE_PROMPT = (
    "Write a {rating}-star customer review for {business}. 30-60 words. "
    "Only output the review text, nothing else."
)
ADVERSARIAL_PROMPT = (
    "Imagine you are a real customer writing an imaginary {rating}-star review for "
    "{business}. Write it casually like a real person typing quickly on their phone -- "
    "lowercase is fine, minor imperfections are fine, change your tone compared to a "
    "typical review, avoid generic marketing phrases and avoid sounding like an AI wrote "
    "it. 20-50 words. Only output the review text, nothing else."
)
FEWSHOT_PROMPT = (
    "Here is a real customer review, as a style reference:\n\"\"\"\n{reference}\n\"\"\"\n\n"
    "Using a similar natural, human tone and level of detail as that reference, write a "
    "new {rating}-star review for {business}. Vary the persona and specific details so it "
    "reads as a different person. 20-60 words. Only output the review text, nothing else."
)


def _load_reference_reviews() -> list[str]:
    df = pd.read_csv(BASE_DIR / "reviews_baseline.csv")
    real = df[~df["is_fake"]]["text"].dropna()
    real = real[real.str.len().between(80, 400)]
    return real.sample(n=min(200, len(real)), random_state=0).tolist()


def main() -> None:
    client = OpenAI()  # lee OPENAI_API_KEY del entorno (cargado via .env)
    references = _load_reference_reviews()
    random.seed(0)

    done = set()
    rows = []
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done = set(zip(prev["category"], prev["rating"], prev["prompt_style"], prev["seed"]))
        print(f"Reanudando: {len(done)} ya generadas de una ejecucion anterior.")

    total_cost = 0.0
    total = len(CATEGORIES) * len(PLAN)
    n_done = len(done)

    for category in CATEGORIES:
        business = CATEGORY_LABEL[category]
        for seed, (style, rating) in enumerate(PLAN):
            key = (category, rating, style, seed)
            if key in done:
                continue

            if style == "naive":
                prompt = NAIVE_PROMPT.format(rating=rating, business=business)
            elif style == "adversarial":
                prompt = ADVERSARIAL_PROMPT.format(rating=rating, business=business)
            else:
                reference = random.choice(references)
                prompt = FEWSHOT_PROMPT.format(reference=reference, rating=rating, business=business)

            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.9,
            )
            text = resp.choices[0].message.content.strip().strip('"')
            cost = (
                resp.usage.prompt_tokens * PRICE_IN_PER_TOK
                + resp.usage.completion_tokens * PRICE_OUT_PER_TOK
            )
            total_cost += cost

            rows.append({
                "category": category, "rating": rating, "prompt_style": style,
                "seed": seed, "text": text, "is_fake": True,
                "source_dataset": f"own_corpus_{MODEL.replace('-', '_')}",
                "generator": MODEL,
            })
            n_done += 1

            if total_cost >= BUDGET_CAP_USD:
                print(f"TOPE DE PRESUPUESTO alcanzado (${total_cost:.4f} >= ${BUDGET_CAP_USD}). Parando.")
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                return

            if n_done % 10 == 0 or n_done == total:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  {n_done}/{total} guardado -- coste acumulado estimado: ${total_cost:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Total filas: {len(df)} -- coste total estimado: ${total_cost:.4f}")
    print(f"Guardado en {OUT_CSV}")


if __name__ == "__main__":
    main()
