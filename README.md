# Práctica 4 · Detección de vehículos, matrículas y OCR

Trabajo en curso. Están hechos el dataset de matrículas, el detector (YOLO11n), su versión para matrículas lejanas y la detección sobre el vídeo de test. Lo que falta está en [Pendiente](#pendiente).

## Estructura

```
VC_P4/
├── README.md
├── requirements.txt                dependencias del entorno
├── descargar_datos.py / .ipynb     descarga datos/ desde Hugging Face
├── generar_dataset_reducido.py     genera datos/dataset_lejanas a partir de dataset_limpio
├── datos/                          (no se sube a GitHub: está en Hugging Face)
│   ├── dataset_limpio/             dataset YOLO de matrículas usado para entrenar
│   └── dataset_lejanas/            matrículas reducidas (data augmentation), se genera con generar_dataset_reducido.py
├── videos/                         (no se sube a GitHub) vídeos de prueba y resultados
├── P4/                             detección
│   ├── VC_P4_Yolo.ipynb            ★ cuaderno único del modelo final: entrenamiento, comparación y vídeo
│   ├── entrenar_matriculas.py      entrena y evalúa el detector (local, 640)
│   ├── entrenar_matriculas.ipynb   lo mismo en cuaderno
│   ├── runs/matriculas_yolo11n/    entrenamiento local a 640: curvas, matriz de confusión, weights/best.pt
│   ├── runs/matriculas_yolo11n_lejanas/  ★ modelo final: entrenamiento con matrículas reducidas
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

Los pesos (`best.pt`) de cada entrenamiento sí se suben a GitHub (pesan unos 5 MB). Los vídeos no: `C0142.MP4` se descarga en `datos/videos/` con `python descargar_datos.py --solo video`. Los resultados de un entrenamiento hecho en Colab se descomprimen dentro de `P4/`.

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
| Colab (`matriculas_yolo11n_1280`) | 1280 | 4 | 60 | 0,908 | 1,000 | 0,981 | 0,792 |

Las diferencias son pequeñas para un conjunto de 28 imágenes. El problema real aparece en el vídeo: ninguno de los dos detecta las matrículas lejanas.

## 2. Detector de matrículas lejanas (modelo final)

En el dataset, la matrícula más pequeña mide 31 px de ancho a `imgsz=640`. En el vídeo de test, un coche lejano tiene una matrícula de unos 17×4 px, y además `imgsz=1280` reduce el fotograma de 1920 px y la deja en unos 11 px.

El entrenamiento, la comparación y el vídeo final están en el cuaderno [`P4/VC_P4_Yolo.ipynb`](P4/VC_P4_Yolo.ipynb). El dataset se genera aparte, una sola vez, con `python generar_dataset_reducido.py`. Con la configuración por defecto usa los resultados guardados y solo genera el vídeo final (unos 5 minutos); cada paso largo se activa con una opción en la celda de configuración.

**Dataset (`datos/dataset_lejanas`):** 1600 lienzos de 960×960 (1200 train, 200 valid, 200 test), creados solo a partir de `dataset_limpio`, sin usar el vídeo de test y sin mezclar imágenes entre splits. Sobre una imagen de fondo se pegan recortes de coches reducidos, con el borde difuminado, para que la matrícula mida entre 12 y 120 px (mediana de 27 px en train). Se añade desenfoque y compresión JPEG para imitar un coche lejano en vídeo, y alrededor de un 10 % de los lienzos son negativos, sin matrículas etiquetadas.

**Entrenamiento:** parte de los pesos a 1280, 60 épocas, `imgsz=960`, `batch=8`, `amp=False`, `mosaic=0` y `scale=0.25`, para no encoger más las matrículas pequeñas. Se entrena con los lienzos y las imágenes originales. Tarda alrededor de 1 hora. Después se infiere a `imgsz=1920`, que conserva los píxeles del fotograma.

![Curvas de entrenamiento](P4/runs/matriculas_yolo11n_lejanas/results.png)

### Resultados en test

| Modelo | Test | `imgsz` | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| 640 | Real (32 matrículas) | 640 | 0,935 | 0,969 | 0,960 | 0,776 |
| 1280 | Real (32 matrículas) | 1280 | 0,958 | 0,969 | 0,982 | 0,782 |
| **Lejanas** | Real (32 matrículas) | 960 | 0,944 | 0,938 | 0,960 | 0,771 |
| 640 | Reducidas (883 matrículas) | 960 | 0,770 | 0,314 | 0,367 | 0,208 |
| 1280 | Reducidas (883 matrículas) | 960 | 0,874 | 0,353 | 0,413 | 0,266 |
| **Lejanas** | Reducidas (883 matrículas) | 960 | 0,937 | 0,849 | 0,904 | 0,580 |

Con matrículas normales rinde prácticamente igual: la diferencia de recall es una sola matrícula. Con matrículas pequeñas mejora mucho, aunque ese test es sintético y puede dar una imagen optimista.

### Resultados en el vídeo de test

`C0142.MP4`, 1 de cada 10 fotogramas (284), con `conf=0.25`. El vídeo no tiene etiquetas: "dentro de un vehículo" significa que la caja cae en un vehículo y tiene forma de matrícula, lo que **no equivale a precisión**.

| Modelo e `imgsz` | Detecciones | Dentro de un vehículo | Fuera o forma rara | Menos de 20 px | 20-30 px |
|---|---|---|---|---|---|
| 640 a 1280 | 56 | 53 | 3 | 0 | 0 |
| 1280 a 1280 | 86 | 84 | 2 | 0 | 1 |
| 1280 a 1920 | 116 | 112 | 4 | 0 | 8 |
| Lejanas a 1280 | 234 | 233 | 1 | 25 | 74 |
| **Lejanas a 1920** | **292** | **268** | 24 | **69** | **55** |

- De las 24 detecciones "fuera o forma rara", revisadas a ojo, 18 son un mismo hueco blanco en un cartel fijo, 3 son errores sueltos (casco, bolsa y patinete) y 3 son matrículas reales que este criterio descarta.
- Por eso el vídeo final solo dibuja las matrículas situadas en la parte baja de un vehículo, con una forma mínima más permisiva para aceptar las de moto.
- Cuesta unos 25 ms por fotograma en el detector de matrículas a 1920. Con el detector de vehículos, el proceso completo va a unos 9 fotogramas por segundo.

**Limitaciones:** sin etiquetas en el vídeo no se sabe cuántas matrículas se pierden. A estos tamaños se detectan, pero no se pueden leer con OCR.

## Pendiente

- Completar el prototipo de vídeo: seguimiento de personas y vehículos, conteo por clase y CSV. La detección de matrículas ya está en `VC_P4_Yolo.ipynb`.
- Enlazar el vídeo de resultados (`videos/salida_C0142_lejanas.mp4`), porque los vídeos no se suben a GitHub.
- OCR de las matrículas (P4b): texto en el vídeo y en el CSV, y comparativa de tiempos y acierto entre un OCR clásico y un VLM.
- Extras opcionales: flujo por direcciones (+0,25), YOLO frente a RT-DETR para matrículas (+1), detección por contornos (+0,25) y anonimización (+0,25).
