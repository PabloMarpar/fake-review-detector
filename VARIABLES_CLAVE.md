# Variables clave del perfilado — qué importa y por qué (uso interno)

Chuleta rápida de qué campo alimenta qué señal, con las cifras reales medidas hasta el
2026-09-14. Detalle completo y metodología en `README.md` (Niveles A-D) y `CONTEXTO.md`
(histórico de sesiones) — este fichero es solo el resumen para no tener que releer todo eso
cada vez.

## Nivel A — estructural/temporal (siempre disponible, ya validado)

Campos: `reviewer_id`, `business_id`, `rating`, `date`, `text` — la tabla de reviews que
cualquier cliente ya tiene, sin pedir nada nuevo.

- **`net_rur` (mismo reviewer) es la señal más fuerte medida hasta ahora**: LOO-AUC 0,90 en
  Yelp-NYC, 0,997 en Amazon (bretthollenbeck). **Ojo**: la dirección no es universal — en
  Yelp-NYC un reviewer prolífico es más probable que sea genuino (tipo "Yelp Elite"); en Amazon
  (productos con campaña de fraude conocida) es al revés, más reviews del mismo reviewer = más
  fraude. Nunca fundir esta señal en un número único sin el contexto del dataset/plataforma.
- `net_rtr`/`net_rsr` (mismo negocio+rating[+ventana]): más débil (LOO-AUC 0,55-0,73), sirve de
  complemento, no como señal principal.
- Ratio de cuentas de una sola review, densidad vs. modelo de configuración: implementadas en
  `profile_cluster.py`.

## Nivel B — cohorte de cuenta (bloqueado, la pieza que más falta)

Campo que falta: `users.created_at` (fecha de alta de cuenta). Es casi universal en cualquier
plataforma, pero **ningún dataset público de reviews lo trae** (comprobado con Yelp-Chi, Amazon,
Yelp-NYC y bretthollenbeck) — es la mejora de mayor impacto pendiente, y solo llega con datos
reales de un cliente. En cuanto exista, activa: concentración de altas de cuenta frente a la
tasa base de la plataforma, edad de la cuenta en su primera review.

## Nivel C — texto

Cubierto por T1/T2/T3 (ver README para el detalle). **T2 (DeBERTa-v3) es la única señal de texto
que generaliza a LLMs modernos** — TPR@5%FPR 0,825 contra un generador nunca visto en
entrenamiento.

## Nivel D — metadatos (parcialmente implementado)

- `email_domain` (solo el dominio, nunca el email completo): implementado
  (`profile_cluster.py::nivel_d_evidence`) contra la API gratuita `disposable.debounce.io`.
- `username`, `avatar_url`: diseñados en el README, sin implementar todavía (ningún dataset
  cargado los trae).
- `campaign_start_date` (si el cliente sospecha una campaña concreta): probado en
  bretthollenbeck — la proximidad temporal a una fecha de evento conocida da AUC 0,65, mejor que
  el burst genérico por volumen (AUC 0,55). Solo aplica si el cliente puede aportar esa fecha, no
  es un campo universal.

## Lo que NO aporta señal (probado, no supuesto)

- **T1 (Binoculars) y T3 (estilometría) fallan casi por completo contra LLMs modernos**
  (TPR@5%FPR < 0,15 en general, 0,0 contra el estilo `hard_evasion` de `gpt-6-astra`) — solo
  sirven contra generadores viejos tipo GPT-2.
- **Fusión T1+T2+T3**: generaliza peor que T2 solo — no usar.
- **`net_homo`** (unión sin ponderar de las tres redes): señal inútil en los tres datasets
  probados (AUC ~0).
