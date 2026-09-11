"""Descarga manual de un modelo de HuggingFace via `requests` puro, sin pasar
por `huggingface_hub.snapshot_download`.

Diagnosticado en sesion (ver CONTEXTO.md): en esta red, `snapshot_download`
se cuelga de forma consistente incluso con `max_workers=1` -- un fichero
concreto se corta siempre en el mismo punto (parece un problema especifico
de como esa libreria gestiona la conexion/reintentos en esta red, no del
ancho de banda: `curl` y `requests` sueltos descargan los mismos ficheros
sin problema, incluido ese mismo fichero completo). Este script hace la
descarga con `requests.get(..., stream=True)` fichero a fichero a una
carpeta local plana, que luego se carga con
`AutoModelForCausalLM.from_pretrained(carpeta_local)` en vez de un repo id
-- transformers soporta cargar directamente desde una carpeta local con
config.json + tokenizer + safetensors, sin necesitar la estructura de cache
de huggingface_hub.

Uso:
    python own_corpus/_download_model_direct.py <repo_id> <carpeta_destino>
"""

import sys
from pathlib import Path

import requests

FILES = [
    "config.json", "generation_config.json", "merges.txt",
    "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
]


def download_file(repo_id: str, filename: str, dest_dir: Path) -> None:
    dest = dest_dir / filename
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    if dest.exists():
        print(f"  {filename}: ya existe, salto")
        return

    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        done = resume_from
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                done += len(chunk)
        print(f"  {filename}: {done}/{total} bytes")
        if total and done < total:
            raise RuntimeError(f"{filename} incompleto: {done}/{total}")
    tmp.rename(dest)


def main() -> None:
    repo_id, dest = sys.argv[1], Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        for attempt in range(5):
            try:
                download_file(repo_id, filename, dest)
                break
            except Exception as e:
                print(f"  {filename}: intento {attempt + 1} fallo ({e}), reintentando")
        else:
            raise RuntimeError(f"No se pudo descargar {filename} tras 5 intentos")
    print(f"OK: modelo completo en {dest}")


if __name__ == "__main__":
    main()
