# Plan: HuggingFace Spaces - MyIAModelChat

## Información de Costes (HuggingFace Spaces)

### Hardware disponible

| Hardware | CPU | RAM | VRAM | Disco | Precio/hora |
|----------|-----|-----|------|-------|-------------|
| CPU Basic | 2 vCPU | 16 GB | - | 50 GB | GRATIS |
| CPU Upgrade | 8 vCPU | 32 GB | - | 50 GB | $0.03 |
| 1x Nvidia T4 - small | 4 vCPU | 15 GB | 16 GB | 50 GB | $0.40 |
| 1x Nvidia T4 - medium | 8 vCPU | 30 GB | 16 GB | 100 GB | $0.60 |
| **1x Nvidia L4** | **8 vCPU** | **30 GB** | **24 GB** | **400 GB** | **$0.80** |
| 4x Nvidia L4 | 48 vCPU | 186 GB | 96 GB | 3200 GB | $3.80 |
| 1x Nvidia A10G - small | 4 vCPU | 15 GB | 24 GB | 110 GB | $1.00 |
| 1x Nvidia A100 - large | 12 vCPU | 142 GB | 80 GB | 1000 GB | $2.50 |

**Nota:** El L4 de HF tiene 24GB VRAM. El modelo GPT-2 de este proyecto (~6.8M params, embed_size=256, 4 capas) es extremadamente pequeno para esta GPU.

### Facturacion

- Se cobra **por minuto** mientras el Space este `Starting` o `Running`
- **No se cobra** durante el `Build` (instalacion de dependencias)
- Si el Space esta en pausa o suspendido, no se cobra
- Se puede configurar "sleep time" para que se pause automaticamente
- Para pausar: Settings -> Hardware -> Pause
- Budget alert: Settings -> Billing -> Usage limits

### Estimacion de costes por escenario

| Fase | Duracion estimada | Coste L4 ($0.80/h) |
|------|-------------------|---------------------|
| Build (Docker/dependencies) | 5-15 min | $0.00 (no se cobra) |
| Preparar datos (1K samples) | 1-2 min | ~$0.03 |
| Preparar datos (10K samples) | 5-10 min | ~$0.13 |
| Preparar datos (50K samples) | 15-30 min | ~$0.40 |
| Entrenar 30 epochs (1K samples) | 2-5 min | ~$0.07 |
| Entrenar 30 epochs (10K samples) | 10-20 min | ~$0.27 |
| Entrenar 30 epochs (50K samples) | 30-60 min | ~$0.80 |
| Chat (por sesion de 30 min) | 30 min | $0.40 |

### Coste mensual estimado

**Uso moderado (recomendado para empezar):**
- 10 sesiones de entrenamiento (10K samples, 30 epochs c/u) = 200 min = **$2.67**
- 20 sesiones de chat (30 min c/u) = 600 min = **$8.00**
- **Total mensual: ~$10.67**

**Uso intensivo:**
- 30 sesiones de entrenamiento = 600 min = **$8.00**
- 60 sesiones de chat = 1800 min = **$24.00**
- **Total mensual: ~$32.00**

**Opcion mas economica (entrenar local, solo chat en el Space):**
- 30 sesiones de chat = **$8.00/mes**
- El entrenamiento se hace en local y se sube el checkpoint al Hub

---

## Opcion 1: Gradio (Recomendada)

### Por que Gradio

- SDK **nativo** de HF Spaces (sin Docker custom)
- Chatbot interface **incluida** (`gr.ChatInterface`)
- Soporte GPU directo sin configuracion extra
- Menos archivos que mantener
- Integracion nativa con HF Hub
- Build rapido (3-5 min)
- GPU automatica sin configurar CUDA

### Limitaciones

- Customizacion visual mas limitada que Streamlit/Docker
- No se puede usar el Cache Viewer PyQt5 existente (es web, no desktop)

### Archivos a crear/modificar

| Archivo | Accion | Descripcion |
|---------|--------|-------------|
| `app.py` | CREAR | Entry point Gradio con 3 tabs |
| `requirements.txt` (raiz) | CREAR | Dependencias para el Space |
| `README.md` (raiz) | MODIFICAR | Metadata YAML de HF Spaces |
| `training/trainer.py` | MODIFICAR | Logger callback para Gradio |
| `inference/chat_engine.py` | MODIFICAR | Suprimir prints, usar logger |

### Estructura del Space

```
MyIAModelChat/
  app.py                          # Entry point Gradio (NUEVO)
  README.md                       # Metadata HF Spaces (modificar)
  requirements.txt                # Dependencias Space (NUEVO en raiz)
  datasets_source/                # Datos (subir al repo)
    aiml/                         # Archivos AIML
    pdf/                          # PDFs subidos por el usuario
    epub/                         # EPUBs subidos por el usuario
  training/
  inference/
  commons/
  dataset_preparer/
```

### Interfaz Gradio (app.py)

```python
import gradio as gr

# Tab 1: Data Preparation
# - Upload PDFs/EPUBs (gr.File)
# - Select sources (gr.CheckboxGroup: AIML, HF, PDF, EPUB)
# - BPE vocab size slider (gr.Slider)
# - "Prepare Data" button (gr.Button)
# - Log output (gr.Textbox)

# Tab 2: Training
# - Epochs slider (gr.Slider, 1-100)
# - Checkpoint name input (gr.Textbox)
# - "Train" button (gr.Button)
# - Live training log (gr.Textbox)
# - Training status indicator (gr.Markdown)

# Tab 3: Chat
# - Model selector (gr.Dropdown con checkpoints disponibles)
# - Chat interface (gr.ChatInterface)
# - Thinking toggle (gr.Checkbox)
```

### Dependencias (requirements.txt)

```
torch
transformers
sentencepiece
datasets
numpy
psutil
gradio>=4.0.0
huggingface_hub
pypdf
ebooklib
trafilatura
beautifulsoup4
requests
python-aiml
```

### Pasos de implementacion

**Fase 1: Preparacion local**

1. Crear `requirements.txt` en la raiz del proyecto con las dependencias del Space
2. Crear `app.py` con la interfaz Gradio (3 tabs: Data, Train, Chat)
3. Modificar `training/trainer.py` para aceptar un logger callback que envie mensajes a Gradio en vez de solo print
4. Modificar `inference/chat_engine.py` para usar `logging` en vez de `print()` en DialogueManager
5. Verificar que todo funciona localmente: `python app.py`

**Fase 2: Crear el Space en HF**

1. Ir a https://huggingface.co/new-space
2. Nombre: `MyIAModelChat` (o el que prefieras)
3. SDK: **Gradio**
4. Visibilidad: **Private**
5. Hardware: **1x Nvidia L4** ($0.80/h)
6. Click "Create Space"

**Fase 3: Subir codigo**

```bash
# 1. Clonar el Space recien creado
git clone https://huggingface.co/spaces/TU_USUARIO/MyIAModelChat
cd MyIAModelChat

# 2. Copiar archivos del proyecto (NO copiar .git, ServerFastAPI/, tests/, APP_CACHE_VIEWER/)
# Solo copiar lo necesario:
cp /ruta/proyecto/app.py .
cp /ruta/proyecto/requirements.txt .
cp /ruta/proyecto/README.md .
cp -r /ruta/proyecto/commons/ .
cp -r /ruta/proyecto/training/ .
cp -r /ruta/proyecto/inference/ .
cp -r /ruta/proyecto/dataset_preparer/ .
cp /ruta/proyecto/main.py .
cp /ruta/proyecto/config.py .

# 3. Crear carpeta de datos y copiar
mkdir -p datasets_source
cp -r /ruta/proyecto/datasets_source/aiml/ datasets_source/aiml/

# 4. Subir PDFs y EPUBs adicionales
cp /ruta/mis_documentos/*.pdf datasets_source/pdf/
cp /ruta/mis_documentos/*.epub datasets_source/epub/

# 5. Push al Space
git add .
git commit -m "Initial setup: Gradio app with training + chat"
git push
```

**Fase 4: Configurar el Space**

1. En HuggingFace -> Tu Space -> Settings -> Hardware
2. Seleccionar "1x Nvidia L4"
3. Esperar a que el Space build + start (5-15 min)
4. Verificar que la app carga correctamente

**Fase 5: Usar**

1. **Preparar datos**: Tab "Data" -> upload PDFs -> seleccionar fuentes -> click "Prepare"
2. **Entrenar**: Tab "Train" -> configurar epochs -> click "Train"
3. **Chatear**: Tab "Chat" -> seleccionar modelo -> escribir mensaje

**Fase 6: Gestion de costes**

- Pausar el Space cuando no se use: Settings -> Hardware -> Pause
- O configurar "Sleep time": Settings -> Space sleep time -> 30 min
- Monitorear uso: Settings -> Billing -> Usage

---

## Opcion 2: Streamlit

### Estado actual

Streamlit **ya no es un SDK nativo** de HF Spaces. Se ejecuta via Docker usando el template `streamlit/streamlit-template-space`. HF recomienda usar Docker SDK con Streamlit dentro.

### Por que Streamlit

- Mejor customizacion visual que Gradio
- `st.chat_message()` para interfaces de chat
- `st.tabs()` para organizar la UI
- Mas familiar si ya se conoce Streamlit

### Limitaciones

- No es SDK nativo -> requiere Docker -> Build mas lento (8-15 min)
- Menor integracion con HF Hub que Gradio
- Template de Streamlit en HF Spaces puede tener versiones desactualizadas
- Mas archivos de mantenimiento

### Archivos a crear

| Archivo | Accion | Descripcion |
|---------|--------|-------------|
| `app_streamlit.py` | CREAR | Entry point Streamlit |
| `Dockerfile` | CREAR | Container con Streamlit + dependencias |
| `requirements.txt` (raiz) | CREAR | Dependencias |
| `README.md` (raiz) | MODIFICAR | `sdk: docker` + `app_port: 8501` |

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar proyecto
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### Interfaz Streamlit (app_streamlit.py)

```python
import streamlit as st

st.title("MyIAModelChat")

tab1, tab2, tab3 = st.tabs(["Data", "Train", "Chat"])

with tab1:
    st.subheader("Data Preparation")
    uploaded_files = st.file_uploader("Upload PDFs/EPUBs",
                                       type=["pdf", "epub"],
                                       accept_multiple_files=True)
    sources = st.multiselect("Sources", ["AIML", "HF", "PDF", "EPUB"])
    vocab_size = st.slider("BPE Vocab Size", 1000, 20000, 8000)
    if st.button("Prepare Data"):
        # Run data preparation
        pass

with tab2:
    st.subheader("Training")
    epochs = st.slider("Epochs", 1, 100, 30)
    checkpoint_name = st.text_input("Checkpoint Name", "chat_model")
    if st.button("Train"):
        # Run training
        pass

with tab3:
    st.subheader("Chat")
    # st.chat_message() for chat interface
```

### README.md metadata

```yaml
---
title: MyIAModelChat
emoji: robot
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
hardware: gpu-l4
---
```

### Pasos de implementacion

1. Crear `Dockerfile` con base `python:3.11-slim`
2. Crear `app_streamlit.py` con interfaz de 3 tabs
3. Crear `requirements.txt` con dependencias
4. Modificar `README.md` con `sdk: docker` y `app_port: 8501`
5. Crear Space en HF -> SDK: **Docker**, Hardware: **1x L4**, Private
6. Clonar Space, copiar archivos, push
7. Esperar build (8-15 min)
8. Verificar app

### Desventajas vs Gradio

- Build mas lento (8-15 min vs 3-5 min)
- Docker bajo el hood -> mas complejo
- Menor integracion con HF Hub
- Template de HF Spaces puede tener versiones desactualizadas
- Mas archivos de mantenimiento (Dockerfile + app)

---

## Opcion 3: Docker (Maximo control)

### Por que Docker

- Control **total** del entorno
- Puedes incluir cualquier herramienta (no solo Gradio/Streamlit)
- Puedes combinar FastAPI + interfaz web
- Acceso a sistema de archivos completo
- Puedes personalizar la imagen CUDA base

### Limitaciones

- Mas archivos que mantener (Dockerfile, docker-compose)
- Build mas lento (descargar imagen base CUDA, 10-20 min)
- Sin integracion nativa con HF Hub
- Mas complejo de depurar
- Requiere conocimientos de Docker

### Archivos a crear

| Archivo | Accion | Descripcion |
|---------|--------|-------------|
| `Dockerfile` | CREAR | Container completo con CUDA + app |
| `docker-compose.yml` | CREAR | Para desarrollo local (opcional) |
| `app_web.py` | CREAR | FastAPI + Gradio montado |
| `requirements.txt` (raiz) | CREAR | Dependencias |
| `README.md` (raiz) | MODIFICAR | `sdk: docker` + `app_port: 7860` |

### Dockerfile (con GPU)

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Python + dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Proyecto completo
COPY . .

EXPOSE 7860

CMD ["python3", "app_web.py"]
```

### docker-compose.yml (desarrollo local)

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./datasets_source:/app/datasets_source
      - ./checkpoints:/app/checkpoints
      - ./dataset_cache:/app/dataset_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Interfaz Web (app_web.py)

```python
from fastapi import FastAPI
import gradio as gr

app = FastAPI()

# Crear interfaz Gradio
with gr.Blocks() as demo:
    with gr.Tab("Data"):
        # Upload, sources, prepare button
        pass
    with gr.Tab("Train"):
        # Epochs, train button, logs
        pass
    with gr.Tab("Chat"):
        # Chat interface
        pass

# Montar Gradio en FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

### README.md metadata

```yaml
---
title: MyIAModelChat
emoji: robot
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
hardware: gpu-l4
---
```

### Pasos de implementacion

1. Crear `Dockerfile` con base `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
2. Crear `app_web.py` con FastAPI + Gradio montado
3. Crear `requirements.txt` con todas las dependencias
4. Crear `docker-compose.yml` para testing local
5. Modificar `README.md` con `sdk: docker` y `app_port: 7860`
6. Crear Space en HF -> SDK: **Docker**, Hardware: **1x L4**, Private
7. Clonar Space, copiar archivos, push
8. Esperar build (10-20 min)
9. Verificar app

### Testing local con Docker

```bash
# Build
docker build -t myiamodelchat .

# Run con GPU
docker run --gpus all -p 7860:7860 myiamodelchat

# O con docker-compose
docker-compose up --build
```

### Desventajas vs Gradio

- Dockerfile mas complejo (imagen CUDA base)
- Build mas lento (10-20 min vs 3-5 min)
- Sin integracion nativa con HF Hub
- Mas archivos de mantenimiento
- Requiere conocimientos de Docker

---

## Comparativa Final

| Criterio | Gradio | Streamlit | Docker |
|----------|--------|-----------|--------|
| **Complejidad** | Baja | Media | Alta |
| **Archivos nuevos** | 3 | 4 | 4 |
| **Build time** | 3-5 min | 8-15 min | 10-20 min |
| **Customizacion UI** | Media | Alta | Total |
| **Chat integration** | Nativa (ChatInterface) | st.chat_message() | Manual |
| **GPU setup** | Automatico | via Dockerfile | Manual (nvidia/cuda) |
| **HF Hub integration** | Nativa | Manual | Manual |
| **Mantenimiento** | Bajo | Medio | Alto |
| **Coste L4** | $0.80/h | $0.80/h | $0.80/h |
| **Ideal para** | Demo rapida, chat | UI customizada | Control total |

### Recomendacion

**Gradio** es la mejor opcion para este proyecto porque:

1. SDK nativo de HF Spaces (sin Docker custom)
2. `gr.ChatInterface` resuelve el chat con 5 lineas de codigo
3. Menos archivos que crear y mantener
4. Integracion nativa con GPU (sin configurar CUDA)
5. Build mas rapido
6. El modelo es pequeno (6.8M params) -> no necesitas optimizaciones especiales

**Si necesitas UI mas customizada**, usa Docker con Gradio dentro (no Streamlit).

**Si ya conoces Streamlit**, usalo pero via Docker (el template de HF Spaces esta desactualizado).

---

## Checklist de implementacion (Opcion Gradio)

- [ ] Crear `requirements.txt` en la raiz del proyecto
- [ ] Crear `app.py` con interfaz Gradio (3 tabs)
- [ ] Modificar `training/trainer.py` para logging callback
- [ ] Modificar `inference/chat_engine.py` para usar logger
- [ ] Modificar `README.md` con metadata HF Spaces
- [ ] Crear Space en huggingface.co (Gradio, L4, Private)
- [ ] Clonar Space y copiar archivos
- [ ] Copiar datasets_source/aiml/ al Space
- [ ] Subir PDFs/EPUBs a datasets_source/pdf/ y epub/
- [ ] Push al Space
- [ ] Configurar hardware L4 en Settings
- [ ] Verificar que la app carga
- [ ] Probar: preparar datos -> entrenar -> chatear
- [ ] Configurar sleep time para ahorrar costes
