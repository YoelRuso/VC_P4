"""Recorta las matrículas de dataset_limpio para evaluar el OCR y marca en el CSV las difíciles.

Uso:
    python P4b/recortar_matriculas.py [--min-ancho 100] [--min-nitidez 60] [--margen 0.05]

La etiqueta de texto sale del nombre del archivo (p. ej. 0041LZN_jpg.rf.<hash>.jpg -> 0041LZN).
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).resolve().parent.parent / "datos"
DATASET = DATOS / "dataset_limpio"
SALIDA = DATOS / "matriculas_recortadas"

# Matrícula española actual: 4 dígitos + 3 consonantes (sin vocales, Ñ ni Q)
PATRON = re.compile(r"^(\d{4})([BCDFGHJKLMNPRSTVWXYZ]{3})")


# Nombres de archivo mal escritos, corregidos leyendo la imagen
CORRECCIONES = {"220LNW": "2202LNW"}


def etiqueta_desde_nombre(nombre):
    prefijo = re.split(r"_jpe?g", nombre, maxsplit=1)[0]
    prefijo = CORRECCIONES.get(prefijo, prefijo)
    m = PATRON.match(prefijo)
    return (m.group(1) + m.group(2)) if m else None


def leer(ruta):
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar(ruta, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(ruta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ancho", type=int, default=100, help="ancho mínimo del recorte en px")
    ap.add_argument("--min-nitidez", type=float, default=60.0, help="varianza del Laplaciano mínima")
    ap.add_argument("--max-area", type=float, default=0.20, help="fracción máxima de la imagen que puede ocupar la caja")
    ap.add_argument("--aspecto", type=float, nargs=2, default=[1.6, 6.0], metavar=("MIN", "MAX"), help="rango ancho/alto del recorte")
    ap.add_argument("--margen", type=float, default=0.05, help="margen extra alrededor de la caja")
    args = ap.parse_args()

    # Solo se regeneran los recortes: ocr_dataset.csv y escribir_matriculas.csv tienen etiquetas manuales
    if (SALIDA / "todas").exists():
        shutil.rmtree(SALIDA / "todas")
    (SALIDA / "todas").mkdir(parents=True)

    filas = []
    for split in ("train", "valid", "test"):
        for img_path in sorted((DATASET / split / "images").iterdir()):
            lbl = DATASET / split / "labels" / (img_path.stem + ".txt")
            if not lbl.exists():
                filas.append([split, img_path.name, "", "", "", "", "dificil", "sin etiqueta YOLO", ""])
                continue
            img = leer(img_path)
            if img is None:
                filas.append([split, img_path.name, "", "", "", "", "dificil", "imagen ilegible", ""])
                continue
            h, w = img.shape[:2]
            texto = etiqueta_desde_nombre(img_path.name)
            lineas = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
            for i, partes in enumerate(lineas):
                cx, cy, bw, bh = (float(v) for v in partes[1:5])
                mx, my = bw * args.margen, bh * args.margen
                x1 = max(0, int((cx - bw / 2 - mx) * w))
                x2 = min(w, int((cx + bw / 2 + mx) * w))
                y1 = max(0, int((cy - bh / 2 - my) * h))
                y2 = min(h, int((cy + bh / 2 + my) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ancho = crop.shape[1]
                nitidez = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

                motivos = []
                if ancho < args.min_ancho:
                    motivos.append(f"ancho {ancho}px < {args.min_ancho}")
                if nitidez < args.min_nitidez:
                    motivos.append(f"nitidez {nitidez:.0f} < {args.min_nitidez:.0f}")
                area = (bw * bh)
                if area > args.max_area:
                    motivos.append(f"caja ocupa {area:.0%} de la imagen")
                aspecto = ancho / crop.shape[0]
                if not args.aspecto[0] <= aspecto <= args.aspecto[1]:
                    motivos.append(f"proporción {aspecto:.2f} fuera de rango (placa girada o caja errónea)")
                if len(lineas) > 1:
                    motivos.append("varias cajas en la imagen")

                destino = "dificil" if motivos else "facil"
                base = texto or re.split(r"_jpe?g", img_path.name, maxsplit=1)[0]
                sufijo = f"_{i}" if len(lineas) > 1 else ""
                nombre_png = f"{base}{sufijo}.png"
                n = 2
                while (SALIDA / "todas" / nombre_png).exists():  # misma matrícula en varias fotos
                    nombre_png = f"{base}{sufijo}_foto{n}.png"
                    n += 1
                guardar(SALIDA / "todas" / nombre_png, crop)
                filas.append([split, img_path.name, texto or "", ancho, f"{nitidez:.0f}", i, destino, "; ".join(motivos), nombre_png])

    with open(SALIDA / "informe.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "imagen", "matricula", "ancho_px", "nitidez", "caja", "dificultad", "motivos", "recorte"])
        wr.writerows(filas)

    faciles = sum(1 for f in filas if f[6] == "facil")
    print(f"Total recortes: {len(filas)} | faciles: {faciles} | dificiles: {len(filas) - faciles}")
    print(f"Salida: {SALIDA}")


if __name__ == "__main__":
    main()
