# Qué datos ayudan a CheckGraph a detectar mejor las reseñas falsas

Para analizar tus reseñas nos basta con lo mínimo que ya tienes en tu base de datos: quién
escribió cada reseña, sobre qué proveedor, qué nota puso, el texto y la fecha. Con eso ya
funcionamos.

Si además nos puedes compartir esto (todo opcional), la detección mejora bastante:

## Ayuda mucho y es fácil de compartir

- **Fecha de alta de cada cuenta de usuario.** Ya la tienes en tu sistema, no hay que capturar
  nada nuevo. Nos permite ver si un grupo de cuentas se creó todo junto en pocos días — uno de
  los patrones más claros de una red de cuentas falsas.

## Ayuda, si lo tienes

- **Nombre de usuario y foto de perfil.** Permiten detectar cuentas que reutilizan la misma foto
  o siguen un patrón de nombre muy parecido entre sí.
- **Dominio de email** (por ejemplo `gmail.com`) — **nunca la dirección completa**. Con solo el
  dominio podemos ver si varias reseñas sospechosas vienen de servicios de email "de usar y
  tirar".

## Solo si te ha pasado algo así

- Si sospechas que una tanda de reseñas falsas empezó en una fecha concreta (por ejemplo, viste
  una oferta pública de "reseñas a cambio de dinero" dirigida a tu negocio), esa fecha
  aproximada nos ayuda mucho a encontrar el patrón exacto alrededor de ella.

## Lo que nunca te vamos a pedir

- Acceso a tu base de datos.
- La dirección de email completa de tus usuarios.
- Ningún dato pensado para identificar quién es una persona en la vida real.

Cuanta más de esta información nos compartas, más preciso es el análisis — pero funcionamos
igual, solo más limitados, con únicamente tus reseñas.
