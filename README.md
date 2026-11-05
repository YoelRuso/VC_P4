# Práctica 4 · Detección de vehículos, matrículas y OCR

Trabajo en curso. Por ahora están hechos el dataset de matrículas y el entrenamiento y la evaluación del detector de matrículas (YOLO11n). Lo que falta está en [Pendiente](#pendiente).

## Estructura

```
VC_P4/
├── README.md
├── requirements.txt                dependencias del entorno
├── descargar_datos.py / .ipynb     descarga datos/ desde Hugging Face
├── datos/                          (no se sube a GitHub: está en Hugging Face)
│   └── dataset_limpio/             dataset YOLO de matrículas usado para entrenar
├── videos/                         (no se sube a GitHub) vídeos de prueba y resultados
├── P4/                             detección
│   ├── entrenar_matriculas.py      entrena y evalúa el detector (local, 640)
│   ├── entrenar_matriculas.ipynb   lo mismo en cuaderno
│   ├── entrenar_matriculas_1280.ipynb  entrenamiento a 1280 para matrículas lejanas (pensado para Colab)
│   ├── probar_pesos_video.ipynb    aplica unos pesos a un vídeo y guarda el resultado en videos/
│   ├── runs/matriculas_yolo11n/    entrenamiento local a 640: curvas, matriz de confusión, weights/best.pt
│   ├── runs/matriculas_yolo11n_colab/  entrenamiento a 640 repetido en Colab (`batch=16`)
│   ├── runs/matriculas_yolo11n_1280/   entrenamiento a 1280 (Colab, `batch=4`, `scale=0.9`)
│   ├── runs/matriculas_yolo11n_test/   evaluación en test
│   ├── load_rtdetr.py / .ipynb     carga RT-DETR con pesos COCO (aún no se ha entrenado con matrículas)
│   ├── RT-DETR/                    (no se sube) clon de https://github.com/lyuwenyu/RT-DETR
│   └── VC_P4.ipynb, README.md…     material de la asignatura
└── P4b/                            OCR (material de la asignatura)
```

## Entorno

```bash
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
pip install -r requirements.txt
python -m ipykernel install --user --name vc_p4 --display-name "Python (VC_P4)"
```

### Trabajar desde otro equipo

```bash
git clone https://github.com/YoelRuso/VC_P4.git
cd VC_P4
# crear el entorno (ver arriba) y descargar el dataset
python descargar_datos.py --solo dataset
# solo si se usa RT-DETR
git clone https://github.com/lyuwenyu/RT-DETR P4/RT-DETR
```

Los pesos (`best.pt`) de cada entrenamiento sí se suben a GitHub (pesan unos 5 MB). Los vídeos no: copiar `C0142.MP4` y similares a `videos/`. Los resultados de un entrenamiento hecho en Colab se descomprimen dentro de `P4/`.

Los datos están en el repositorio privado de Hugging Face [YoelRuso/MatriculasVC](https://huggingface.co/datasets/YoelRuso/MatriculasVC). Tras iniciar sesión una vez con `hf auth login`, el dataset se descarga en `datos/` con:

```bash
python descargar_datos.py --solo dataset
```

**Equipo de pruebas:** Intel Core i5-10400F, 16 GB RAM, NVIDIA GeForce RTX 3050 (8 GB), Windows 11, PyTorch 2.8 + CUDA 12.6, Ultralytics 8.4.

## 1. Detector de matrículas

### Dataset

Imágenes de vehículos con matrícula española, anotadas en Roboflow con una única clase (`License Plate`) y exportadas en formato YOLOv8. Se fusionaron dos exports de Roboflow, se descartaron 9 imágenes en una limpieza manual y se movieron 24 de train a valid/test para equilibrar el reparto (`datos/dataset_limpio/movimientos_reparto.csv`):

| Split | Imágenes |
|---|---|
| train | 151 |
| valid | 28 |
| test | 31 |

Acceso al dataset: [YoelRuso/MatriculasVC](https://huggingface.co/datasets/YoelRuso/MatriculasVC) en Hugging Face (licencia CC BY 4.0, la de los proyectos de Roboflow de origen).

### Entrenamiento

`python P4/entrenar_matriculas.py`: YOLO11n preentrenado en COCO, 60 épocas, `imgsz=640`, `batch=8`, GPU. Con AMP activado la pérdida se iba a `nan` en la época 5, por eso se entrena con `amp=False`. Tarda unos 5 minutos.

![Curvas de entrenamiento](P4/runs/matriculas_yolo11n/results.png)

- Las pérdidas de entrenamiento (`box`, `cls`, `dfl`) bajan de forma continua; las de validación bajan con ellas sin repuntar, así que no hay sobreajuste claro.
- `cls_loss` cae muy rápido: con una sola clase, el problema es sobre todo de localización.
- El mAP50 sube a ≈0,95 hacia la época 20 y se queda en ≈0,98 desde la 40. El mAP50-95 sigue mejorando hasta el final (≈0,78): lo que más cuesta es ajustar la caja con precisión.

| Conjunto | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validación (28 img.) | 0,963 | 0,935 | 0,975 | 0,785 |
| **Test (31 img.)** | 0,935 | 0,969 | **0,961** | **0,778** |

Los conjuntos de validación y test son pequeños, así que estas cifras tienen bastante margen de error: una matrícula más o menos cambia varios puntos. Ninguna imagen de validación o test se usó para entrenar.

### Comparación de entrenamientos

Mejor época por mAP50-95 de cada `results.csv`, siempre sobre validación:

| Entrenamiento | `imgsz` | `batch` | Época | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| Local (`matriculas_yolo11n`) | 640 | 8 | 58 | 0,963 | 0,935 | 0,975 | 0,785 |
| Colab (`matriculas_yolo11n_colab`) | 640 | 16 | 46 | 0,937 | 0,962 | 0,983 | 0,764 |
| Colab (`matriculas_yolo11n_1280`) | 1280 | 4 | 60 | 0,908 | 1,000 | 0,981 | 0,792 |

Solo el entrenamiento local está evaluado en test, así que estas diferencias todavía no sirven para elegir modelo. Además, son pequeñas para un conjunto de 28 imágenes.

## Pendiente

- Prototipo de vídeo: seguimiento de personas y vehículos, detección de matrículas, conteo por clase, CSV y vídeo de resultados (enlazado, porque los vídeos no se suben a GitHub).
- OCR de las matrículas (P4b): texto en el vídeo y en el CSV, y comparativa de tiempos y acierto entre un OCR clásico y un VLM.
- Evaluar en test los entrenamientos de Colab para elegir el modelo.
- Extras opcionales: flujo por direcciones (+0,25), YOLO frente a RT-DETR para matrículas (+1), detección por contornos (+0,25) y anonimización (+0,25).
