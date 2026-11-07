"""Genera datos/dataset_lejanas: data augmentation con matrículas reducidas a partir de dataset_limpio.

Cada imagen es un lienzo de 960x960 con la escala de píxel del vídeo (1920x1080):
- fondo: otra imagen del mismo split, con sus matrículas etiquetadas;
- encima, de 3 a 6 recortes de coches reducidos para que la matrícula mida entre 12 y 120 px,
  con el borde difuminado, desenfoque y compresión JPEG para imitar un coche lejano en vídeo;
- alrededor de un 10 % de los lienzos no llevan matrículas etiquetadas.

Cada split usa solo sus propias imágenes, así test no comparte fotos con train.
El vídeo de test no se usa para generar nada. Con la misma semilla sale siempre el mismo dataset.

Uso: python generar_dataset_reducido.py
"""
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent
ORIGEN = RAIZ / "datos" / "dataset_limpio"
DESTINO = RAIZ / "datos" / "dataset_lejanas"
LIENZOS = {"train": 1200, "valid": 200, "test": 200}
LADO = 960
PROP_NEGATIVOS = 0.10  # lienzos de fondo sin matrículas
SEMILLA = 42


def leer_img(ruta):
    return cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR)


def cargar_split(split):
    muestras = []
    for ruta in sorted((ORIGEN / split / "images").glob("*.jpg")):
        etiqueta = ORIGEN / split / "labels" / (ruta.stem + ".txt")
        cajas = []
        for linea in etiqueta.read_text().splitlines():
            p = linea.split()
            if len(p) == 5:
                cajas.append(tuple(float(v) for v in p[1:]))  # cx cy w h normalizados
        muestras.append({"ruta": ruta, "cajas": cajas})
    return muestras


def a_pixeles(cajas, W, H):
    return [((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H) for cx, cy, w, h in cajas]


def solapa(a, b, margen=4):
    return not (a[2] + margen <= b[0] or b[2] + margen <= a[0] or a[3] + margen <= b[1] or b[3] + margen <= a[1])


def ancho_objetivo(rng):
    """80 %: lejanas (12-45 px, log-uniforme); 20 %: tamaños intermedios (45-120 px)."""
    bajo, alto = (12, 45) if rng.random() < 0.8 else (45, 120)
    return float(np.exp(rng.uniform(np.log(bajo), np.log(alto))))


def degradar(img, rng, escala_extra=True):
    """Imita la pérdida de nitidez de un objeto lejano en vídeo."""
    h, w = img.shape[:2]
    if escala_extra and min(h, w) > 8:
        f = rng.uniform(1.0, 1.6)
        peq = cv2.resize(img, (max(int(w / f), 2), max(int(h / f), 2)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(peq, (w, h), interpolation=cv2.INTER_LINEAR)
    sigma = rng.uniform(0.3, 1.0)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    if rng.random() < 0.3:  # movimiento horizontal
        k = rng.choice([3, 5])
        img = cv2.filter2D(img, -1, np.ones((1, k), np.float32) / k)
    return img


def cubrir_lienzo(img, cajas_px, rng):
    """Escala y recorta la imagen de fondo a LADOxLADO; devuelve las cajas que quedan enteras."""
    H, W = img.shape[:2]
    esc = max(LADO / W, LADO / H) * rng.uniform(1.0, 1.3)
    nw, nh = int(round(W * esc)), int(round(H * esc))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
    ox, oy = rng.randint(0, nw - LADO), rng.randint(0, nh - LADO)
    lienzo = img[oy:oy + LADO, ox:ox + LADO].copy()
    fuera = []
    for x1, y1, x2, y2 in cajas_px:
        c = (x1 * esc - ox, y1 * esc - oy, x2 * esc - ox, y2 * esc - oy)
        ix = max(0, min(c[2], LADO) - max(c[0], 0))
        iy = max(0, min(c[3], LADO) - max(c[1], 0))
        area = (c[2] - c[0]) * (c[3] - c[1])
        if ix * iy >= 0.9 * area:
            fuera.append(tuple(np.clip(c, 0, LADO)))
        elif ix * iy > 0:
            # Matrícula cortada por el borde: se tapa para no dejar una placa sin etiqueta
            x1i, y1i = int(max(c[0], 0)), int(max(c[1], 0))
            x2i, y2i = int(min(c[2], LADO)), int(min(c[3], LADO))
            lienzo[y1i:y2i, x1i:x2i] = cv2.GaussianBlur(lienzo, (0, 0), 25)[y1i:y2i, x1i:x2i]
    return lienzo, fuera


def recorte_coche(muestra, img, rng):
    """Recorte alrededor de una matrícula, con tamaño aproximado de coche (más contexto)."""
    H, W = img.shape[:2]
    cajas = a_pixeles(muestra["cajas"], W, H)
    x1, y1, x2, y2 = rng.choice(cajas)
    pw = x2 - x1
    ancho = pw * rng.uniform(3.5, 8.0)
    alto = pw * rng.uniform(2.0, 5.0)
    cx, cy = (x1 + x2) / 2 + pw * rng.uniform(-0.8, 0.8), (y1 + y2) / 2 - pw * rng.uniform(0.2, 1.2)
    rx1, ry1 = int(max(cx - ancho / 2, 0)), int(max(cy - alto / 2, 0))
    rx2, ry2 = int(min(cx + ancho / 2, W)), int(min(cy + alto / 2, H))
    rec = img[ry1:ry2, rx1:rx2]
    dentro = []
    for c in cajas:
        if c[0] >= rx1 and c[1] >= ry1 and c[2] <= rx2 and c[3] <= ry2:
            dentro.append((c[0] - rx1, c[1] - ry1, c[2] - rx1, c[3] - ry1))
    return rec, dentro, pw


def pegar(lienzo, rec, x, y):
    """Pega con el borde difuminado para que el rectángulo no sea una pista para el modelo."""
    h, w = rec.shape[:2]
    borde = max(1, min(h, w) // 8)
    alfa = np.zeros((h, w), np.float32)
    alfa[borde:h - borde, borde:w - borde] = 1
    alfa = cv2.GaussianBlur(alfa, (0, 0), max(borde / 2, 0.6))[..., None]
    zona = lienzo[y:y + h, x:x + w].astype(np.float32)
    lienzo[y:y + h, x:x + w] = (rec * alfa + zona * (1 - alfa)).astype(np.uint8)


def negativo(muestras, cache, rng):
    """Zona ampliada de una imagen que no contiene ninguna matrícula etiquetada."""
    for _ in range(40):
        m = rng.choice(muestras)
        img = cache(m)
        H, W = img.shape[:2]
        cajas = a_pixeles(m["cajas"], W, H)
        lado = int(min(H, W) * rng.uniform(0.25, 0.5))
        x, y = rng.randint(0, W - lado), rng.randint(0, H - lado)
        if not any(solapa((x, y, x + lado, y + lado), c, 0) for c in cajas):
            return degradar(cv2.resize(img[y:y + lado, x:x + lado], (LADO, LADO), interpolation=cv2.INTER_LINEAR), rng, False)
    return None


def generar_lienzo(muestras, cache, rng):
    con_placa = [m for m in muestras if m["cajas"]]
    fondo = rng.choice(con_placa)
    img_f = cache(fondo)
    H, W = img_f.shape[:2]
    lienzo, cajas = cubrir_lienzo(img_f, a_pixeles(fondo["cajas"], W, H), rng)
    lienzo = cv2.GaussianBlur(lienzo, (0, 0), rng.uniform(0.3, 0.8))
    ocupado = list(cajas)
    anchos = []
    for _ in range(rng.randint(3, 6)):
        m = rng.choice(con_placa)
        if m is fondo:
            continue
        rec, dentro, pw = recorte_coche(m, cache(m), rng)
        if not dentro:
            continue
        objetivo = min(ancho_objetivo(rng), pw)  # nunca se amplía
        esc = objetivo / pw
        nw, nh = max(int(rec.shape[1] * esc), 3), max(int(rec.shape[0] * esc), 3)
        if nw >= LADO - 2 or nh >= LADO - 2:
            continue
        peq = degradar(cv2.resize(rec, (nw, nh), interpolation=cv2.INTER_AREA), rng)
        for _ in range(20):
            x, y = rng.randint(0, LADO - nw), rng.randint(0, LADO - nh)
            zona = (x, y, x + nw, y + nh)
            if any(solapa(zona, c) for c in ocupado):
                continue
            pegar(lienzo, peq, x, y)
            ocupado.append(zona)
            for c in dentro:
                caja = (c[0] * esc + x, c[1] * esc + y, c[2] * esc + x, c[3] * esc + y)
                cajas.append(caja)
                anchos.append(caja[2] - caja[0])
            break
    return lienzo, cajas, anchos


def generar_dataset(destino):
    rng = random.Random(SEMILLA)
    if destino.exists():
        shutil.rmtree(destino)
    filas = []
    for split, total in LIENZOS.items():
        (destino / split / "images").mkdir(parents=True)
        (destino / split / "labels").mkdir(parents=True)
        muestras = cargar_split(split)
        memo = {}

        def cache(m):
            if m["ruta"] not in memo:
                if len(memo) > 40:  # limita la RAM
                    memo.pop(next(iter(memo)))
                memo[m["ruta"]] = leer_img(m["ruta"])
            return memo[m["ruta"]]

        negativos = 0
        for i in range(total):
            neg = negativo(muestras, cache, rng) if rng.random() < PROP_NEGATIVOS else None
            if neg is not None:
                lienzo, cajas, anchos = neg, [], []
                negativos += 1
            else:
                lienzo, cajas, anchos = generar_lienzo(muestras, cache, rng)
            nombre = f"lejana_{split}_{i:04d}"
            ok, buf = cv2.imencode(".jpg", lienzo, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 85)])
            assert ok
            buf.tofile(str(destino / split / "images" / f"{nombre}.jpg"))
            lineas = [f"0 {(x1 + x2) / 2 / LADO:.6f} {(y1 + y2) / 2 / LADO:.6f} {(x2 - x1) / LADO:.6f} {(y2 - y1) / LADO:.6f}"
                      for x1, y1, x2, y2 in cajas]
            (destino / split / "labels" / f"{nombre}.txt").write_text("\n".join(lineas))
            filas += [(split, nombre, round(a, 1)) for a in anchos]
        print(f"{split}: {total} lienzos ({negativos} negativos), {sum(1 for f in filas if f[0] == split)} placas pegadas")

    with open(destino / "placas_pegadas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "imagen", "ancho_px"])
        w.writerows(filas)

    (destino / "data.yaml").write_text(
        "# Lienzos con matrículas reducidas + imágenes originales de dataset_limpio\n"
        "train: [train/images, ../dataset_limpio/train/images]\n"
        "val: [valid/images, ../dataset_limpio/valid/images]\n"
        "test: test/images\n\nnc: 1\nnames: ['License Plate']\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generar_dataset(DESTINO)
