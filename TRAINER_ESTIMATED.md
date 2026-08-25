# Estimaciones de Entrenamiento - MyIAModelChat

## Arquitectura del Modelo (GPT-2)

| Parámetro | Valor |
|-----------|-------|
| **Arquitectura** | GPT-2 (`GPT2LMHeadModel` de HuggingFace) |
| **Embed size** | 256 |
| **Hidden size** | 512 |
| **Cabezas de atención** | 4 |
| **Capas Transformer** | 2 (por defecto, configurable) |
| **Posiciones máximas** | 512 |
| **Tokenizer** | SentencePiece BPE (vocab size: 8000 por defecto) |

## Configuración de Entrenamiento (TRAINING_CONFIG)

| Parámetro | Valor |
|-----------|-------|
| **Batch size** | 4 |
| **Gradient accumulation steps** | 8 (batch efectivo = 32) |
| **Learning rate** | 1e-3 (0.001) |
| **Gradient clip norm** | 1.0 |
| **Epochs** | 1 (por defecto, configurable con `--epochs`) |
| **Mixed precision** | Sí (FP16 en GPU) |
| **Gradient checkpointing** | Solo en GPUs Tesla K80 |

### Warm-up

| Parámetro | Valor |
|-----------|-------|
| **Habilitado** | Sí |
| **Ratio** | 10% del dataset |
| **Máx. pasos warm-up** | 100 batches |

### Thinking (Razonamiento)

| Parámetro | Valor |
|-----------|-------|
| **Habilitado** | Sí |
| **Pérdida ponderada** | 1.0 (default) — configurable con `--thinking-loss-weight` |
| **Máx. tokens thinking** | 64 |

---

## Estimaciones por GPU (50M muestras)

### Tesla K80

**Especificaciones:**
- VRAM: 24 GB
- Compute Capability: 3.7 (Kepler - arquitectura antigua)
- FP32: ~8.7 TFLOPS
- FP16: Soporte limitado

| Tiempo por micro-step | Tiempo total 1 epoch |
|----------------------|---------------------|
| 5ms (optimista) | ~17 horas |
| 10ms (realista) | **~35 horas** |
| 20ms (conservador) | ~70 horas |

**Estimación realista: ~35 horas (1.5 días)**

**Factores que afectan:**
- K80 es antigua: Compute Capability 3.7 (Kepler)
- Mixed precision no ayuda: K80 no soporta bien FP16
- Gradient checkpointing: Ahorra VRAM pero añade ~20-30% overhead
- Data loading: `num_workers=0` puede ser cuello de botella

---

### RTX 5050

**Especificaciones:**
- VRAM: 8 GB
- Compute Capability: 10.0 (Blackwell - arquitectura moderna)
- Tensor Cores: Sí (aceleración FP16/BF16 nativa)
- FP32: ~15-20 TFLOPS
- FP16 con Tensor Cores: ~30-40 TFLOPS

| Tiempo por micro-step | Tiempo total 1 epoch |
|----------------------|---------------------|
| 1ms (optimista) | ~3.5 horas |
| 2-3ms (realista) | **~7-10 horas** |
| 5ms (conservador) | ~17 horas |

**Estimación realista: ~8 horas (un turno de trabajo)**

**Ventajas sobre K80:**
- Mixed precision real: Tensor Cores aceleran FP16
- Arquitectura moderna: Blackwell vs Kepler (10+ años de diferencia)
- Mejor throughput: ~3-5x más TFLOPS
- Memoria más rápida: Mayor bandwidth
- Optimizaciones CUDA: Soporte completo para kernels modernos

---

### NVIDIA L4 (Hugging Face Space)

**Especificaciones:**
- FP32: 30.3 TFLOPS
- FP16 Tensor Cores: 121 TFLOPS (sin sparsity)
- BFLOAT16 Tensor Cores: 121 TFLOPS
- VRAM: 24 GB
- Memory Bandwidth: 300 GB/s
- Arquitectura: Ada Lovelace
- **Precio HF Space: $0.80/hora**

| Tiempo por micro-step | Tiempo total 1 epoch |
|----------------------|---------------------|
| 1ms (optimista) | ~3.5 horas |
| 2-3ms (realista) | **~7-10 horas** |
| 5ms (conservador) | ~17 horas |

**Estimación realista: ~10 horas**

**Costo en HF Space:**

| Escenario | Tiempo | Costo |
|-----------|--------|-------|
| Optimista | 7 horas | $5.60 |
| Realista | 10 horas | $8.00 |
| Conservador | 17 horas | $13.60 |

---

### NVIDIA T4 Small (Hugging Face Space)

**Especificaciones:**
- FP32: 8.1 TFLOPS
- FP16 (mixed precision): 65 TFLOPS
- VRAM: 16 GB GDDR6
- Memory Bandwidth: 300 GB/s
- Arquitectura: Turing
- **Precio HF Space: $0.40/hora** (T4 small) / $0.60/hora (T4 medium)

**Comparación T4 vs L4:**

| Métrica | T4 | L4 | Ratio |
|---------|-----|-----|-------|
| FP32 | 8.1 TFLOPS | 30.3 TFLOPS | 3.7x |
| FP16 Tensor Cores | 65 TFLOPS | 121 TFLOPS | 1.9x |
| VRAM | 16 GB | 24 GB | 1.5x |
| Precio HF Space | $0.40/h | $0.80/h | 2x |

**Estimaciones para 50M tokens (GPT-2 puro):**

| Modelo | Tiempo T4 | Costo T4 | Tiempo L4 | Costo L4 |
|--------|-----------|----------|-----------|----------|
| GPT-2 Small | ~75 min | **$0.50** | ~45 min | $0.60 |
| GPT-2 Medium | ~2.5 h | **$1.00** | ~70 min | $0.93 |

**T4 small es la opción más económica** si el tiempo no es crítico.

---

## Comparación de GPUs (50M muestras)

| GPU | Tiempo estimado | Costo | Speedup vs K80 |
|-----|-----------------|-------|----------------|
| Tesla K80 | ~35 horas | - | 1x |
| RTX 5050 | ~8 horas | local | ~4.4x |
| L4 (HF Space) | ~10 horas | $8 | ~3.5x |
| A10G (HF Space) | ~8 horas | $12 | ~4.4x |
| A100 (HF Space) | ~3 horas | $7.50 | ~12x |

---

## Escalamiento Chinchilla (GPT-2 puro desde cero)

### Regla de Chinchilla
- **Óptimo**: ~20 tokens por parámetro
- Fórmula: `Tokens óptimos = 20 × Número de parámetros`

### Tokens necesarios por modelo

| Modelo | Params | Tokens óptimos (Chinchilla) |
|--------|--------|-----------------------------|
| GPT-2 Small | 124M | 2.48B |
| GPT-2 Medium | 345M | 6.9B |
| GPT-2 Large | 774M | 15.5B |
| GPT-2 XL | 1.5B | 30B |

---

## Estimaciones para 50M tokens (GPT-2 puro)

### GPT-2 Small (124M params) - Recomendado para 50M tokens

| Fase | Tiempo estimado |
|------|-----------------|
| Preprocesamiento (tokenización) | ~5-10 min |
| Entrenamiento | **~15-25 min** |
| Guardado modelo | ~1-2 min |
| Upload a HuggingFace | ~2-5 min |
| **Total** | **~25-45 min** |

### GPT-2 Medium (345M params)

| Fase | Tiempo estimado |
|------|-----------------|
| Preprocesamiento | ~5-10 min |
| Entrenamiento | **~30-50 min** |
| Guardado + Upload | ~3-7 min |
| **Total** | **~40-70 min** |

### GPT-2 Large (774M params) - No recomendado para 50M tokens

| Fase | Tiempo estimado |
|------|-----------------|
| Preprocesamiento | ~5-10 min |
| Entrenamiento | **~1-2 horas** |
| Guardado + Upload | ~5-10 min |
| **Total** | **~1.5-2.5 horas** |

### Costo en HF Spaces (L4 a $0.80/hora)

| Modelo | Tiempo | Costo |
|--------|--------|-------|
| GPT-2 Small | ~45 min | $0.60 |
| GPT-2 Medium | ~70 min | $0.93 |
| GPT-2 Large | ~2.5 horas | $2.00 |

---

## Throughput de GPUs (Benchmarks)

| GPU | Throughput | Fuente |
|-----|------------|--------|
| L4 (single) | ~6,200-675 tokens/sec | CloudBench/hybrid LLM |
| A10G (single) | ~208M tokens/hora | Karpathy llm.c |
| A100 (single) | ~667M tokens/hora | Karpathy llm.c |
| 8x L4 | 5,400 tokens/sec | CloudBench (GPT-2 medium) |
| 8x H100 | 48,000 tokens/sec | CloudBench (GPT-2 medium) |

---

## Datos de Hardware HF Spaces

| Hardware | CPU | RAM | GPU VRAM | Disk | Precio/hora |
|----------|-----|-----|----------|------|-------------|
| Nvidia T4 - small | 4 vCPU | 15 GB | 16 GB | 50 GB | $0.40 |
| Nvidia T4 - medium | 8 vCPU | 30 GB | 16 GB | 100 GB | $0.60 |
| 1x Nvidia L4 | 8 vCPU | 30 GB | 24 GB | 400 GB | $0.80 |
| 4x Nvidia L4 | 48 vCPU | 186 GB | 96 GB | 3200 GB | $3.80 |
| 1x Nvidia L40S | 8 vCPU | 62 GB | 48 GB | 380 GB | $1.80 |
| Nvidia A10G - small | 4 vCPU | 15 GB | 24 GB | 110 GB | $1.00 |
| Nvidia A10G - large | 12 vCPU | 46 GB | 24 GB | 200 GB | $1.50 |
| Nvidia A100 - large | 12 vCPU | 142 GB | 80 GB | 1000 GB | $2.50 |

---

## ZeroGPU - NO recomendado para entrenamiento

**ZeroGPU está diseñado para inferencia/demos, NO para entrenamiento.**

### Limitaciones de ZeroGPU

| Cuenta | Cuota diaria GPU | Prioridad cola |
|--------|------------------|----------------|
| Sin autenticar | 2 minutos | Baja |
| Gratis | 5 minutos | Media |
| PRO | 40 minutos | Alta |
| Team/Enterprise | 40-60 minutos | Alta |

- **Duración por llamada**: 60 segundos por defecto
- **GPU**: RTX Pro 6000 Blackwell (48GB VRAM)
- **Costo excedente**: $1 por 10 minutos (solo PRO)

### Por qué NO sirve para entrenamiento

1. **Tiempo muy limitado**: 5-40 minutos diarios vs horas/días necesarios
2. **Diseñado para inferencia**: Asigna y libera GPU dinámicamente
3. **No puede mantener estado**: El modelo se descarga entre llamadas
4. **Cuota diaria**: Se resetea cada 24 horas, no acumula

---

## Entrenamiento en la nube - Desconexión y reconexión

**Sí puedes apagar el equipo local y reconectar después.** El entrenamiento corre en la nube de Hugging Face, no en tu máquina local.

### ¿Qué afecta el entrenamiento?

| Acción | ¿Afecta el entrenamiento? |
|--------|---------------------------|
| Cerrar navegador | **No** - sigue corriendo |
| Apagar PC local | **No** - sigue corriendo |
| Desconectar internet | **No** - sigue corriendo |
| Cerrar sesión HF | **No** - sigue corriendo |

### Para reconectar y verificar

1. Ve a `https://huggingface.co/spaces/TU_USUARIO/TU_SPACE`
2. Los logs se muestran en la interfaz del Space
3. Si usas `print()` en tu código, se ven en los logs

### Ejemplo de código con logs

```python
import time

@spaces.GPU(duration=600)  # 10 minutos por llamada
def train():
    for epoch in range(epochs):
        loss = train_epoch()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        print(f"Progress: {(epoch+1)/epochs*100:.1f}%")
        time.sleep(1)
```

---

## HF PRO vs Hardware Directo

### Beneficios de HF PRO ($9/mes)

| Beneficio | Detalle |
|-----------|---------|
| **8x ZeroGPU quota** | 40 min/día GPU (vs 5 min gratis) |
| **Prioridad máxima cola** | Tu Space se ejecuta primero |
| **1 TB almacenamiento privado** | 10x más que gratis (100 GB) |
| **20x Inference credits** | 2M créditos mensuales |
| **10 ZeroGPU Spaces** | Hosting de apps GPU |
| **Dev Mode** | SSH/VS Code a tus Spaces |
| **Badge PRO** | Perfil verificado |

### Comparación de costos (50M tokens, GPT-2 Small)

| Opción | Costo | Tiempo | Notas |
|--------|-------|--------|-------|
| **HF PRO + ZeroGPU** | $9/mes | No viable | 40 min/día insuficiente |
| **HF PRO + L4 dedicado** | $9 + $0.80/h | ~45 min | **$9.60 total** |
| **L4 directo (sin PRO)** | $0.80/h | ~45 min | **$0.60 total** |
| **T4 small directo** | $0.40/h | ~75 min | **$0.50 total** |

### ¿Cuándo vale la pena PRO?

| Escenario | ¿PRO vale? | Razón |
|-----------|------------|-------|
| Uso frecuente de ZeroGPU | **Sí** | 8x más cuota diaria |
| Múltiples Spaces GPU | **Sí** | Hasta 10 Spaces |
| Inference API intensivo | **Sí** | 2M créditos/mes |
| Almacenamiento grande | **Sí** | 1TB privado |
| Entrenamiento puntual | **No** | Paga solo el hardware |
| Un solo Space | **No** | El hardware dedicado es suficiente |

### Alternativas a PRO para entrenamiento

| Opción | Costo | Ventaja |
|--------|-------|---------|
| **L4 directo** | $0.80/h | Sin suscripción, pago por uso |
| **T4 small directo** | $0.40/h | Más barato, sin compromiso |
| **A100** | $2.50/h | Más rápido, sin suscripción |

### Cómo cancelar HF PRO

1. Ve a `https://huggingface.co/settings/billing/subscription`
2. Haz clic en **"Cancel subscription"**
3. Confirma la cancelación
4. La suscripción se cancela al final del mes facturado
5. **No hay reembolso** por el tiempo restante del mes

**Nota**: La cancelación es inmediata pero la suscripción permanece activa hasta la fecha de facturación.

---

## Métodos de pago

### ¿Se puede pagar con PayPal?

**NO.** Hugging Face solo acepta **tarjetas de crédito/débito** a través de Stripe.

| Método | Soportado |
|--------|-----------|
| Tarjeta de crédito | **Sí** |
| Tarjeta de débito | **Sí** |
| PayPal | **No** |
| AWS Marketplace | **Sí** (solo organizaciones) |
| Transferencia bancaria | **Sí** (solo Enterprise) |

### Configurar método de pago

1. Ve a `https://huggingface.co/settings/billing/payment`
2. Haz clic en **"Add payment method"**
3. Ingresa los datos de tu tarjeta
4. Se realizará un cargo temporal de $10 (se reembolsa en 7 días)
5. Confirma el pago

---

## Tutorial completo: Entrenar en HF Spaces

### Paso 1: Crear cuenta en Hugging Face

1. Ve a `https://huggingface.co/join`
2. Crea una cuenta gratuita
3. Verifica tu email

### Paso 2: Configurar método de pago

1. Ve a `https://huggingface.co/settings/billing/payment`
2. Agrega tu tarjeta de crédito/débito
3. Confirma el cargo temporal de $10

### Paso 3: Crear el Space

1. Ve a `https://huggingface.co/new-space`
2. Configura:
   - **Space name**: `mi-entrenamiento`
   - **SDK**: `Gradio` o `Docker`
   - **Hardware**: `CPU Basic` (gratis por ahora)
   - **Visibility**: `Private` (recomendado para entrenamiento)
3. Haz clic en **"Create Space"**

### Paso 4: Subir el proyecto

**Opción A: Git (recomendado)**

```bash
# Clonar el Space
git clone https://huggingface.co/spaces/TU_USUARIO/mi-entrenamiento
cd mi-entrenamiento

# Copiar tu proyecto
cp -r /ruta/a/MyIAModelChat/* .

# Subir archivos
git add .
git commit -m "Subir proyecto de entrenamiento"
git push
```

**Opción B: Interfaz web**

1. Ve a tu Space en el navegador
2. Haz clic en **"Files and versions"**
3. Haz clic en **"Upload file"**
4. Sube tus archivos uno por uno o en ZIP

### Paso 5: Configurar hardware GPU

1. Ve a tu Space
2. Haz clic en **"Settings"** (pestaña superior)
3. En **"Hardware"**, selecciona la GPU deseada:
   - **T4 small**: $0.40/hora (más barato)
   - **L4**: $0.80/hora (mejor relación calidad-precio)
   - **A10G**: $1.00-1.50/hora (más rápido)
4. Haz clic en **"Save"**
5. El Space se reiniciará automáticamente

### Paso 6: Configurar el código de entrenamiento

Crea un archivo `app.py` en la raíz de tu Space:

```python
import os
import sys
import time
from huggingface_hub import HfApi

# Tu código de entrenamiento aquí
def train_model():
    print("Iniciando entrenamiento...")
    
    # Ejemplo: importar y ejecutar tu trainer
    # from training.trainer import Trainer, TrainingConfig
    # config = TrainingConfig(epochs=1)
    # trainer = Trainer(config)
    # trainer.performMainTrain()
    
    # Simular entrenamiento
    for epoch in range(10):
        loss = 1.0 / (epoch + 1)
        print(f"Epoch {epoch+1}/10 - Loss: {loss:.4f}")
        time.sleep(60)  # Simular trabajo
    
    print("Entrenamiento completado!")
    
    # Guardar modelo
    # torch.save(model.state_dict(), "model.pth")
    
    # Subir modelo a HuggingFace Hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj="model.pth",
        path_in_repo="model.pth",
        repo_id="TU_USUARIO/mi-modelo",
        repo_type="model",
    )
    print("Modelo subido a HuggingFace Hub!")

if __name__ == "__main__":
    train_model()
```

### Paso 7: Configurar requirements.txt

```
torch>=2.0.0
transformers
sentencepiece
datasets
huggingface_hub
```

### Paso 8: Iniciar entrenamiento

1. Haz push de tus cambios:
```bash
git add .
git commit -m "Iniciar entrenamiento"
git push
```

2. El Space se reiniciará y comenzará el entrenamiento
3. Ve a la pestaña **"Logs"** para monitorear el progreso

### Paso 9: Reconectar para verificar

**Puedes apagar tu PC y reconectar después.** El entrenamiento corre en la nube.

1. Ve a `https://huggingface.co/spaces/TU_USUARIO/mi-entrenamiento`
2. Haz clic en **"Logs"** para ver el progreso
3. Los mensajes de `print()` se muestran en los logs

### Paso 10: Descargar el modelo entrenado

**Opción A: Desde la interfaz web**

1. Ve a tu Space
2. Haz clic en **"Files and versions"**
3. Busca el archivo del modelo (ej: `model.pth`)
4. Haz clic en el icono de descarga

**Opción B: Usando Git**

```bash
git clone https://huggingface.co/spaces/TU_USUARIO/mi-entrenamiento
# El modelo estará en la carpeta clonada
```

**Opción C: Usando la API de HuggingFace**

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="TU_USUARIO/mi-entrenamiento",
    filename="model.pth",
    local_dir="./modelo"
)
print(f"Modelo descargado en: {path}")
```

### Paso 11: Cerrar el Space definitivamente

**IMPORTANTE:** Para evitar cargos adicionales, debes cerrar el Space.

**Opción A: Pausar el Space (recomendado)**

1. Ve a tu Space
2. Haz clic en **"Settings"**
3. En **"Hardware"**, cambia a **"CPU Basic"** (gratis)
4. Haz clic en **"Save"**
5. O haz clic en **"Pause"** para detener completamente

**Opción B: Eliminar el Space**

1. Ve a tu Space
2. Haz clic en **"Settings"**
3. Scroll hasta **"Delete this Space"**
4. Escribe el nombre del Space para confirmar
5. Haz clic en **"I understand, delete this Space"**

### Resumen de costos

| Fase | Costo |
|------|-------|
| Crear Space | Gratis |
| Subir código | Gratis |
| Hardware CPU | Gratis |
| Hardware GPU (L4, 45 min) | ~$0.60 |
| Almacenamiento | Gratis (hasta 50GB) |
| **Total** | **~$0.60** |

### Consejos importantes

1. **Siempre pausa o elimina el Space** cuando termines
2. **Usa hardware CPU** si solo estás editando código
3. **Monitorea los logs** para saber cuándo termina
4. **Guarda el modelo** en HuggingFace Hub antes de cerrar
5. **Configura "Sleep time"** para que el Space se duerma automáticamente

---

## Recomendaciones

### Para 50M muestras (entrenamiento existente)
- **GPU recomendada**: L4 o A10G en HF Space
- **Tiempo estimado**: ~8-10 horas
- **Costo**: ~$8-12

### Para 50M tokens (GPT-2 puro desde cero)
- **Modelo recomendado**: GPT-2 Small (124M params)
- **Tiempo estimado**: ~25-45 minutos
- **Costo**: ~$0.60
- **Nota**: El estará **muy subentrenado** (2% de óptimo Chinchilla)

### Para entrenamiento óptimo (Chinchilla)
- **GPT-2 Small**: 2.48B tokens (~20-25 horas, ~$16-20)
- **GPT-2 Medium**: 6.9B tokens (~50-60 horas, ~$40-48)

---

*Documento generado el 2026-08-12*
