"""Genera el Generador C del corpus propio: reviews falsas escritas con la
API de OpenAI -- tercera familia de LLM distinta a Claude (Generador A) y
Qwen (Generador B), y la mas representativa de "atacante usando un LLM
moderno de verdad" ya que es la API que mas se menciona en los foros donde
la gente comparte tecnicas para esto (ver CONTEXTO.md).

Generalizado (2026-09-12) para elegir el modelo de OpenAI por linea de
comandos en vez de tener "gpt-4o-mini" fijo -- la cuenta subio de tier a
10.000 RPD/500 RPM a mitad de la sesion anterior, y ahora hay presupuesto
(~5 USD de credito real) para ampliar el corpus probando varios modelos
mas de la cuenta (gpt-5.6-luna/terra/sol, gpt-6-astra), no solo el barato.
Cada modelo escribe en su propio CSV, mismo patron de "un fichero por
generador" que claude_generated.csv / qwen_generated.csv / etc., EXCEPTO
gpt-4o-mini que sigue escribiendo en openai_generated.csv (nombre historico,
train.py ya lo referencia como held-out -- no se renombra para no romper el
checkpoint ni la integracion existente).

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

Presupuesto: tope duro en USD por ejecucion (--budget), calculado con el uso
REAL de tokens que devuelve la API en cada respuesta, no una estimacion a
priori de numero de llamadas. El precio por token SI es una estimacion a
priori, y para los modelos gpt-5.6-*/gpt-6-astra es una conjetura conservadora
(sobreestimada a proposito, ver PRICE_TABLE) porque estos modelos son
posteriores al corte de conocimiento del agente que escribio esto y no hay
forma de consultar el precio real via API -- el backstop de verdad sigue
siendo el credito real limitado en el dashboard de OpenAI.

Uso:
    python own_corpus/generate_openai_corpus.py --model gpt-4o-mini --budget 0.5
    python own_corpus/generate_openai_corpus.py --model gpt-5.6-luna --budget 1.0
    python own_corpus/generate_openai_corpus.py --model gpt-5.6-terra --budget 1.5 --max-rows 150
    python own_corpus/generate_openai_corpus.py --model gpt-6-astra --budget 1.5 --max-rows 50
    python own_corpus/generate_openai_corpus.py --model gpt-6-astra --budget 1.7 --style hard_evasion
"""

import argparse
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Precios conservadores (sobreestimados a proposito) en USD por token, solo
# para el tope de seguridad -- el limite real esta en el dashboard.
# gpt-4o-mini: precio real conocido (confirmado con gasto real de la sesion
# anterior, esta bien calibrado). gpt-5.6-*/gpt-6-astra: el agente no tiene
# forma de verificar el precio real de estos modelos (son de despues de su
# corte de conocimiento y la API no expone precio), asi que se usa un
# sobreestimado deliberadamente creciente por tier -- si esta mal, el efecto
# es parar antes de lo necesario (seguro), no gastar de mas.
PRICE_TABLE = {
    "gpt-4o-mini": (3.0 / 1_000_000, 12.0 / 1_000_000),
    "gpt-4o": (5.0 / 1_000_000, 15.0 / 1_000_000),
    "gpt-5.6-luna": (5.0 / 1_000_000, 20.0 / 1_000_000),
    "gpt-5.6-terra": (20.0 / 1_000_000, 80.0 / 1_000_000),
    "gpt-5.6-sol": (40.0 / 1_000_000, 160.0 / 1_000_000),
    "gpt-6-astra": (60.0 / 1_000_000, 240.0 / 1_000_000),
}
DEFAULT_PRICE = (60.0 / 1_000_000, 240.0 / 1_000_000)  # modelo desconocido: el mas conservador

# gpt-4o-mini conserva el nombre de fichero historico (train.py lo referencia
# como held-out ya integrado) -- el resto sigue el patron
# openai_<modelo-sin-guiones>_generated.csv de un fichero por generador.
_LEGACY_FILENAMES = {"gpt-4o-mini": "openai_generated.csv"}


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _out_csv_for(model: str) -> Path:
    fname = _LEGACY_FILENAMES.get(model, f"openai_{_slug(model)}_generated.csv")
    return Path(__file__).parent / fname


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

# "hard_evasion": prompt_style adicional, pensado como el mas dificil de
# distinguir de una review humana real -- combina en una sola llamada (sin
# segundo paso de auto-revision, para no perder volumen) las tres cosas que
# por separado ya usaban naive/adversarial/fewshot: ancla de estilo real
# (fewshot), persona detallada con un detalle idiosincratico irrelevante
# (tipico de gente real, no de un LLM que va al grano), y una lista
# explicita de tics de IA a evitar + longitud impredecible + imperfecciones.
PERSONA_AGES = [
    "a 24-year-old", "a 58-year-old", "a tired new parent in their 30s",
    "a college student", "a retired teacher in her 60s", "a busy contractor",
    "a first-time visitor from out of town", "a regular customer of 5 years",
    "someone on their lunch break", "a night-shift nurse",
]
PERSONA_REASONS = [
    "stopping by after a long day at work", "celebrating a birthday",
    "here on a whim while running errands", "recommended by a coworker",
    "killing time before catching a flight", "trying to find a last-minute gift",
    "here for the third time this month", "just moved to the neighborhood",
    "meeting an old friend", "needing a quick fix on short notice",
]
PERSONA_QUIRKS = [
    "mentions their dog was waiting in the car the whole time",
    "can't stop thinking about a parking ticket they got outside",
    "compares it offhand to a place back in their hometown",
    "keeps mentioning how bad the weather was that day",
    "got distracted by a phone call partway through and loses the thread a bit",
    "brings up an unrelated argument with a sibling from that morning",
    "notes they were wearing the wrong shoes for the weather",
    "mentions their kid was cranky the whole time",
    "was still thinking about a work email the whole visit",
    "got a text mid-visit and admits they lost their train of thought",
]
HARD_EVASION_PROMPT = (
    "Here is a real customer review, as a style reference:\n\"\"\"\n{reference}\n\"\"\"\n\n"
    "You are {persona_age}, {persona_reason}. Write a genuine, personal {rating}-star review "
    "for {business}, in a natural voice as authentic as the reference above (but about "
    "different specifics -- do not copy it). Naturally work in a small, mostly irrelevant "
    "detail: {persona_quirk}. Write like a real person typing quickly, not like a template: "
    "avoid stock phrases like 'highly recommend', 'will definitely return', 'exceeded my "
    "expectations', 'overall', or 'in conclusion'; do not give a neatly balanced "
    "pros-and-cons structure; let the length vary unpredictably, anywhere from about 15 to "
    "120 words; punctuation can be a little irregular, a sentence can trail off or feel "
    "unfinished, and capitalization at the start of a sentence doesn't have to be "
    "consistent. Only output the review text, nothing else."
)


def _load_reference_reviews() -> list[str]:
    df = pd.read_csv(BASE_DIR / "reviews_baseline.csv")
    real = df[~df["is_fake"]]["text"].dropna()
    real = real[real.str.len().between(80, 400)]
    return real.sample(n=min(200, len(real)), random_state=0).tolist()


_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s")

# max_completion_tokens generoso para toda la familia -- confirmado en esta
# sesion que gpt-5.6-luna/terra/sol y gpt-6-astra son modelos "de razonamiento"
# que gastan ~75-150 tokens de razonamiento OCULTO antes de emitir el texto
# visible (igual de espiritu al bug de <think> de Qwen3, ver CONTEXTO.md) --
# con 150 (el valor que bastaba para gpt-4o-mini) la respuesta salia vacia
# porque el razonamiento se comia todo el presupuesto. 600 deja margen de
# sobra en los cuatro modelos probados sin inflar el coste real (el
# razonamiento no "rellena" el tope, se para solo cuando termina).
MAX_COMPLETION_TOKENS = 600

# Que modelos no admiten 'temperature' personalizada (piden que se omita o
# se deje en 1) se descubre en caliente por si acaso un modelo futuro se
# comporta distinto -- confirmado en esta sesion que aplica a los 4 modelos
# gpt-5.6-*/gpt-6-astra, no a gpt-4o/gpt-4o-mini. Se pre-siembra con los ya
# confirmados para no desperdiciar la primera oleada entera de llamadas en
# paralelo reintentando lo mismo (con varios hilos a la vez, ese primer golpe
# de 400 se repetiria en cada worker antes de que el set se propague).
_NO_CUSTOM_TEMPERATURE_MODELS: set[str] = {
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra",
}


class BillingLimitReached(Exception):
    """Senal de que la API ha rechazado la llamada por un motivo real de
    facturacion/cuota (no un rate-limit transitorio) -- ver
    _create_with_rate_limit_wait. El llamador debe parar limpio, no reintentar.
    """


def _create_with_rate_limit_wait(client: OpenAI, model: str, prompt: str):
    """Reintenta ante un RateLimitError SOLO si trae el patron "try again in
    Xs" (rate-limit transitorio de verdad, con ventana conocida) -- espera
    exactamente ese tiempo (mas 5s de margen) y reintenta. Si el
    RateLimitError NO trae ese patron, ya no se asume la ventana movil de 24h
    del tier viejo de 50 RPD (esta cuenta ya no tiene ese limite) -- se trata
    como limite real de facturacion/cuota (p.ej. 'insufficient_quota') y se
    propaga como BillingLimitReached para que el llamador pare limpio en vez
    de reintentar a ciegas.
    Tambien absorbe el 400 de 'temperature no soportada' que dan los modelos
    de razonamiento reintentando sin ese parametro.
    """
    while True:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        if model not in _NO_CUSTOM_TEMPERATURE_MODELS:
            kwargs["temperature"] = 0.9
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            match = _WAIT_RE.search(str(e))
            if match:
                minutes, seconds = match.groups()
                wait_s = (int(minutes) * 60 if minutes else 0) + float(seconds) + 5
                print(f"  rate limit transitorio -- esperando {wait_s:.0f}s antes de reintentar")
                time.sleep(wait_s)
            else:
                raise BillingLimitReached(str(e)) from e
        except BadRequestError as e:
            if "temperature" in str(e):
                # OJO concurrencia: con varios hilos a la vez, mas de uno
                # puede fallar por 'temperature' antes de que el primero
                # actualice este set -- NO condicionar el reintento a "si ya
                # estaba marcado", o los hilos que pierden la carrera acaban
                # relanzando el error en vez de reintentar sin el parametro.
                # Siempre reintentar sea cual sea el estado del set.
                if model not in _NO_CUSTOM_TEMPERATURE_MODELS:
                    _NO_CUSTOM_TEMPERATURE_MODELS.add(model)
                    print(f"  {model} no admite 'temperature' personalizada -- se omite en adelante.")
                continue
            raise


def _build_prompt(category: str, style: str, rating: int) -> str:
    business = CATEGORY_LABEL[category]
    if style == "naive":
        return NAIVE_PROMPT.format(rating=rating, business=business)
    if style == "adversarial":
        return ADVERSARIAL_PROMPT.format(rating=rating, business=business)
    if style == "hard_evasion":
        reference = random.choice(references_global)
        return HARD_EVASION_PROMPT.format(
            reference=reference, rating=rating, business=business,
            persona_age=random.choice(PERSONA_AGES),
            persona_reason=random.choice(PERSONA_REASONS),
            persona_quirk=random.choice(PERSONA_QUIRKS),
        )
    reference = random.choice(references_global)
    return FEWSHOT_PROMPT.format(reference=reference, rating=rating, business=business)


# Poblado por main() antes de lanzar el ThreadPoolExecutor -- las referencias
# reales se cargan una vez (no por hilo) y solo se leen (nunca se escriben)
# desde los workers, asi que compartirlas asi entre hilos es seguro.
references_global: list[str] = []


def _generate_one(client: OpenAI, model: str, price_in: float, price_out: float, task):
    """Se ejecuta en un hilo del pool. Devuelve un dict con status
    'ok'/'empty'/'billing' -- nunca escribe directamente a disco ni a la
    lista compartida 'rows' (eso lo hace el hilo principal al consumir el
    resultado, para no necesitar lock sobre el CSV/la lista).
    """
    category, seed, style, rating = task
    prompt = _build_prompt(category, style, rating)
    try:
        resp = _create_with_rate_limit_wait(client, model, prompt)
    except BillingLimitReached as e:
        return {"status": "billing", "error": str(e), "task": task}

    text = (resp.choices[0].message.content or "").strip().strip('"')
    cost = resp.usage.prompt_tokens * price_in + resp.usage.completion_tokens * price_out
    if not text:
        reasoning = getattr(resp.usage.completion_tokens_details, "reasoning_tokens", "?")
        return {"status": "empty", "task": task, "cost": cost, "reasoning_tokens": reasoning}

    row = {
        "category": category, "rating": rating, "prompt_style": style,
        "seed": seed, "text": text, "is_fake": True,
        "source_dataset": f"own_corpus_{_slug(model)}",
        "generator": model,
    }
    return {"status": "ok", "row": row, "cost": cost, "task": task}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="Model id de OpenAI a usar")
    parser.add_argument(
        "--budget", type=float, default=None,
        help=(
            "SOLO INFORMATIVO: se imprime en el log junto al coste acumulado estimado, "
            "pero ya NO decide si la ejecucion para (el precio por token de los modelos "
            "gpt-5.6-*/gpt-6-astra es una conjetura no verificada -- se comprobo en sesion "
            "que sobreestima el coste real ~5-6x frente al dashboard real de OpenAI). El "
            "unico motivo real de parada es un error de facturacion/cuota de la propia API."
        ),
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Tope de filas nuevas en esta ejecucion (por defecto: el PLAN completo, 300)",
    )
    parser.add_argument(
        "--style", default=None, choices=["hard_evasion"],
        help=(
            "Si se indica, genera SOLO con este prompt_style (en vez del PLAN mixto "
            "naive/adversarial/fewshot) -- pensado para anadir un lote dedicado de un "
            "estilo nuevo sobre un CSV que ya tiene filas de los estilos originales."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=12,
        help=(
            "Llamadas concurrentes a la API (ThreadPoolExecutor) -- la cuenta tiene 500 "
            "RPM de margen en el tier actual, asi que 10-15 es un buen equilibrio entre "
            "velocidad y no acercarse al limite real. Bajalo si empiezas a ver "
            "'rate limit transitorio' con frecuencia."
        ),
    )
    args = parser.parse_args()

    model = args.model
    budget_cap_usd = args.budget
    max_new_rows = args.max_rows
    n_workers = args.workers
    price_in, price_out = PRICE_TABLE.get(model, DEFAULT_PRICE)
    if model not in PRICE_TABLE:
        print(f"AVISO: precio desconocido para '{model}', usando el mas conservador de la tabla.")
    out_csv = _out_csv_for(model)

    client = OpenAI()  # lee OPENAI_API_KEY del entorno (cargado via .env)
    global references_global
    references_global = _load_reference_reviews()

    done = set()
    rows = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev["category"], prev["rating"], prev["prompt_style"], prev["seed"]))
        print(f"Reanudando '{out_csv.name}': {len(done)} ya generadas de una ejecucion anterior.")

    if args.style == "hard_evasion":
        # Sin PLAN fijo de 30 combos: no hay numero objetivo, se genera lo
        # que el presupuesto permita. Pool grande (200 seeds/categoria) con
        # ratings sesgados a positivo (igual de sesgo que el PLAN original)
        # para que nunca se quede corto de combinaciones nuevas.
        ratings_pool = [5, 5, 5, 5, 5, 5, 4, 4, 4, 3, 2, 1]
        all_tasks = [
            (category, seed, "hard_evasion", ratings_pool[seed % len(ratings_pool)])
            for category in CATEGORIES
            for seed in range(200)
        ]
    else:
        # Orden aleatorio de combinaciones (no categoria-por-categoria en
        # bloque): asi, si --max-rows corta antes de cubrir el PLAN completo,
        # la muestra resultante sigue teniendo variedad de categorias/
        # estilos/ratings en vez de agotar solo las primeras categorias.
        all_tasks = [
            (category, seed, style, rating)
            for category in CATEGORIES
            for seed, (style, rating) in enumerate(PLAN)
        ]
    random.Random(42).shuffle(all_tasks)

    pending = [t for t in all_tasks if (t[0], t[3], t[2], t[1]) not in done]
    if max_new_rows is not None:
        pending = pending[:max_new_rows]

    total_cost = 0.0
    total = len(all_tasks)
    n_new = 0
    n_done_total = len(done)
    save_lock = threading.Lock()
    billing_limit_hit = False

    print(f"Lanzando {len(pending)} tareas pendientes con {n_workers} workers en paralelo...")

    # Se somete en oleadas (no todas las 'pending' de golpe) para poder
    # comprobar billing_limit_hit y checkpointear entre oleadas sin tener que
    # cancelar peticiones HTTP ya en vuelo a media oleada.
    CHUNK = max(n_workers * 3, 10)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for start in range(0, len(pending), CHUNK):
            if billing_limit_hit:
                break
            chunk = pending[start:start + CHUNK]
            futures = [
                executor.submit(_generate_one, client, model, price_in, price_out, t)
                for t in chunk
            ]
            for fut in as_completed(futures):
                result = fut.result()

                if result["status"] == "billing":
                    # Unico motivo real para parar: la API ha rechazado la
                    # llamada por facturacion/cuota real (p.ej.
                    # insufficient_quota), no una estimacion propia en USD --
                    # ver criterio acordado con el usuario tras comprobar el
                    # dashboard real de OpenAI.
                    print(f"LIMITE DE FACTURACION REAL alcanzado: {result['error']}")
                    billing_limit_hit = True
                    continue

                total_cost += result["cost"]  # solo informativo, no decide si parar

                if result["status"] == "empty":
                    print(f"  AVISO: respuesta vacia para {result['task']} "
                          f"(reasoning_tokens={result['reasoning_tokens']}) -- fila descartada.")
                    continue

                with save_lock:
                    rows.append(result["row"])
                    n_new += 1
                    n_done_total += 1
                    if n_new % 20 == 0:
                        pd.DataFrame(rows).to_csv(out_csv, index=False)
                        print(f"  {n_done_total}/{total} guardado -- coste acumulado (informativo) esta ejecucion: ${total_cost:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Total filas en '{out_csv.name}': {len(df)} -- coste de esta ejecucion (informativo): ${total_cost:.4f}")
    print(f"Guardado en {out_csv}")


if __name__ == "__main__":
    main()
