"""Descarga la carpeta datos/ desde Hugging Face.

El repositorio es privado: la primera vez hay que iniciar sesión con `hf auth login`
(o definir la variable de entorno HF_TOKEN).

Uso:
    python descargar_datos.py                 # todo (~540 MB)
    python descargar_datos.py --solo dataset  # solo dataset_limpio (para entrenar)
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "YoelRuso/MatriculasVC"
DATOS = Path(__file__).resolve().parent / "datos"
PARTES = {
    "dataset": ["dataset_limpio/**"],
    "ocr": ["matriculas_recortadas/**"],
    "video": ["videos/**"],
    "originales": ["originales_roboflow/**"],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=PARTES, nargs="+", help="descargar solo estas partes")
    args = ap.parse_args()
    patrones = None if not args.solo else ["README.md"] + [p for s in args.solo for p in PARTES[s]]
    snapshot_download(REPO, repo_type="dataset", local_dir=DATOS, allow_patterns=patrones)
    print("Datos en", DATOS)
