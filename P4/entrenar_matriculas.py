"""Entrena y evalúa el detector de matrículas YOLO11n sobre datos/dataset_limpio.

Uso (desde cualquier carpeta, con el entorno VC_P4 activado):
    python P4/entrenar_matriculas.py            # entrena y evalúa en test
    python P4/entrenar_matriculas.py --solo-test

Los datos se descargan antes con `python descargar_datos.py`.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent  # carpeta P4
DATASET = AQUI.parent / "datos" / "dataset_limpio"
RUNS = AQUI / "runs"
MODELO = RUNS / "matriculas_yolo11n" / "weights" / "best.pt"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-test", action="store_true", help="no entrenar, solo evaluar best.pt en test")
    args = ap.parse_args()

    if not args.solo_test:
        # Con AMP la pérdida se iba a nan en la época 5, por eso amp=False
        YOLO(str(AQUI / "yolo11n.pt")).train(data=str(DATASET / "data.yaml"), epochs=60, imgsz=640, batch=8,
                                             device=0, workers=2, amp=False,
                                             project=str(RUNS), name="matriculas_yolo11n", exist_ok=True)

    m = YOLO(str(MODELO)).val(data=str(DATASET / "data.yaml"), split="test", project=str(RUNS),
                              name="matriculas_yolo11n_test", exist_ok=True, verbose=False)
    print(f"Test | Precisión {m.box.mp:.3f} | Recall {m.box.mr:.3f} | mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")
