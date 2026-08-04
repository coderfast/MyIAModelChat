# ROADMAP: Sistema de Device Management — CPU, GPU, Vulkan, DDP, CPU+GPU

## Resumen de Cambios

Reemplazar `--use-cpuonly` y `--cuda-device` por un sistema completo de gestión de dispositivos con soporte para CPU, GPU (una o múltiples con DDP), Vulkan, CPU+GPU combinado, y enumeración de GPUs disponibles.

### Archivos a modificar

| Archivo | Cambios |
|---------|---------|
| `main.py` | Argumentos CLI, validación, resolución de device, `--gpu-enum` |
| `training/trainer.py` | TrainingConfig, DDP, CPU+GPU layer distribution |
| `inference/chat_engine.py` | ChatConfig, get_device(), CPU+GPU inference |
| `commons/utils/device_utils.py` | **Nuevo** — utilidad compartida de detección/resolución de device |

---

## Argumentos CLI (main.py)

### Eliminar

- `--use-cpuonly` (línea 286)
- `--cuda-device` (líneas 289-290)

### Agregar

| Argumento | Tipo | Descripción |
|-----------|------|-------------|
| `--cpu` | flag | Fuerza ejecución en CPU puro |
| `--gpu` | opcional | Sin valor = GPU por defecto. Con valor `0,1,2` = GPUs específicas |
| `--vulkan` | flag | Fuerza backend Vulkan si está disponible |
| `--gpu-enum` | flag | Enumera GPUs disponibles con info detallada y sale |

### Restricciones

- `--cpu` y `--gpu` son mutuamente excluyentes → error si ambos
- `--vulkan` y `--gpu` son mutuamente excluyentes → error si ambos
- `--vulkan` y `--cpu` son mutuamente excluyentes → error si ambos
- `--gpu` sin valor → usa GPU 0 por defecto

### `--gpu-enum` — Salida esperada

```
=== GPU Enumeration ===
[0] NVIDIA GeForce RTX 5050
    VRAM: 8.00 GB (7.20 GB free)
    Compute Capability: 10.0
    Status: Available
    AI Recommendation: GOOD (8GB — models up to embed_size=512, num_layers=8)

[1] NVIDIA GeForce RTX 3080
    VRAM: 10.00 GB (8.50 GB free)
    Compute Capability: 8.6
    Status: Available
    AI Recommendation: EXCELLENT (10GB — models up to embed_size=512, num_layers=12)

[2] NVIDIA Tesla K80
    VRAM: 24.00 GB (22.10 GB free)
    Compute Capability: 3.7
    Status: Available
    AI Recommendation: EXCELLENT (24GB — large models, watch: slower compute)
=======================
```

### Lógica de CUDA_VISIBLE_DEVICES (main.py líneas 400-406)

- `--cpu`: `CUDA_VISIBLE_DEVICES=''`
- `--gpu` con índices: `CUDA_VISIBLE_DEVICES='0,1,2'`
- `--gpu` sin índices: no tocar (auto-detect)
- `--vulkan`: no tocar CUDA, Vulkan se maneja por separado

---

## Configuraciones (dataclasses)

### TrainingConfig (training/trainer.py)

```python
@dataclass
class TrainingConfig:
    epochs: int = 1
    checkpoint_name: str = 'chat_model'
    dataset_source: str = 'dataset_cache'
    # --- Nuevos campos (reemplazan use_cpuonly y cuda_device) ---
    device_mode: str = 'auto'           # 'cpu' | 'gpu' | 'cpu+gpu' | 'auto'
    gpu_indices: Optional[List[int]] = None  # [0, 1, 2] o None=auto
    use_vulkan: bool = False
    # --- Campos existentes sin cambios ---
    num_cores: int = 0
    num_threads: int = 0
    max_ram_fraction: float = 0.75
    max_ram_bytes: Optional[int] = None
    thinking_loss_weight: float = 0.5
    thinking_enabled: bool = True
    thinking_max_tokens: int = 64
    statistics: bool = False
```

### ChatConfig (inference/chat_engine.py)

```python
@dataclass
class ChatConfig:
    model_name: Optional[str] = None
    # --- Nuevos campos (reemplazan use_cpuonly y cuda_device) ---
    device_mode: str = 'auto'
    gpu_indices: Optional[List[int]] = None
    use_vulkan: bool = False
    # --- Campos existentes sin cambios ---
    show_thinking: bool = False
    thinking_enabled: bool = True
    thinking_max_tokens: int = 64
```

---

## device_utils.py (commons/utils/device_utils.py)

### enumerate_gpus()

 Retorna lista de diccionarios con info de cada GPU:
 ```python
 [
     {
         "index": 0,
         "name": "NVIDIA GeForce RTX 5050",
         "vram_total_gb": 8.0,
         "vram_free_gb": 7.2,
         "compute_capability": (10, 0),
         "status": "Available",
         "ai_recommended": True,
         "ai_level": "GOOD",           # NOT_RECOMMENDED | BASIC | GOOD | EXCELLENT | OPTIMAL
         "max_model_config": "embed_size=512, num_layers=8",
         "notes": "Blackwell arch, good for small-medium models"
     },
     ...
 ]
 ```

 **ai_level** se determina por VRAM total:
 - < 4 GB → `NOT_RECOMMENDED`
 - 4-8 GB → `BASIC`
 - 8-16 GB → `GOOD`
 - 16-24 GB → `EXCELLENT`
 - > 24 GB → `OPTIMAL`

### resolve_device(config) -> torch.device

 Resuelve el device final según `device_mode`:
 - `'cpu'` → `torch.device('cpu')`
 - `'gpu'` → `torch.device('cuda:0')` o `torch.device('cuda:N')`
 - `'cpu+gpu'` → `torch.device('cuda:0')` (capas extras se manejan en model strategy)
 - `'auto'` → auto-detect mejor disponible (CUDA > Vulkan > MPS > CPU)
 - Si `use_vulkan=True` → `torch.device('vulkan')` si disponible, si no error

### check_vulkan_available() -> bool

```python
try:
    return torch.device('vulkan').type == 'vulkan'
except:
    return False
```

### get_optimal_layer_split(model, gpu_device, vram_budget_gb) -> int

 Calcula cuántas capas transformer caben en GPU:
 1. Contar parámetros totales del modelo
 2. Calcular memoria por capa (parámetros × 4 bytes FP32)
 3. Restar memoria de embeddings + lm_head
 4. Retornar número de capas que caben en el budget

### GPU_REFERENCE_DB (diccionario de GPUs conocidas)

Referencia para `--gpu-enum` y recomendaciones:

```python
GPU_REFERENCE_DB = {
    "NVIDIA GeForce RTX 5050": {
        "vram_gb": 8,
        "compute_capability": (10, 0),
        "ai_recommended": True,
        "notes": "Blackwell arch, good for small-medium models"
    },
    "NVIDIA GeForce RTX 3080": {
        "vram_gb": 10,
        "compute_capability": (8, 6),
        "ai_recommended": True,
        "notes": "Ampere arch, solid performance"
    },
    "NVIDIA GeForce RTX 4090": {
        "vram_gb": 24,
        "compute_capability": (8, 9),
        "ai_recommended": True,
        "notes": "Ada Lovelace, excellent for large models"
    },
    "NVIDIA Tesla K80": {
        "vram_gb": 24,
        "compute_capability": (3, 7),
        "ai_recommended": True,
        "notes": "Kepler arch, slower compute, needs memory fraction limit"
    },
    "NVIDIA A100": {
        "vram_gb": 80,
        "compute_capability": (8, 0),
        "ai_recommended": True,
        "notes": "Datacenter GPU, ideal for large-scale training"
    },
}
```

**VRAM Recommendations para `--gpu-enum`:**

| VRAM | Recomendación |
|------|---------------|
| < 4 GB | NOT RECOMMENDED — solo modelos muy pequeños |
| 4-8 GB | BASIC — modelos pequeños (embed_size<=256, num_layers<=4) |
| 8-16 GB | GOOD — modelos medianos (embed_size<=512, num_layers<=8) |
| 16-24 GB | EXCELLENT — modelos grandes |
| > 24 GB | OPTIMAL — modelos muy grandes o multi-GPU training |

---

## Trainer — Device Setup (trainer.py:368-431)

### _setup_device_and_config()

Reescribir con la siguiente lógica:

```python
def _setup_device_and_config(self):
    device_mode = self.config.device_mode
    gpu_indices = self.config.gpu_indices or []
    use_vulkan = self.config.use_vulkan

    # --- CPU puro ---
    if device_mode == 'cpu':
        self.use_gpu = False
        self.use_mixed_precision = False
        self.use_gradient_checkpointing = False
        return torch.device('cpu')

    # --- Vulkan ---
    if use_vulkan:
        if not check_vulkan_available():
            raise RuntimeError("Vulkan not available on this system")
        self.use_gpu = False  # Vulkan no soporta mixed precision aún
        self.use_mixed_precision = False
        return torch.device('vulkan')

    # --- GPU (una o múltiples) ---
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        self.use_gpu = False
        return torch.device('cpu')

    # Configurar CUDA_VISIBLE_DEVICES si hay índices explícitos
    if gpu_indices:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_indices)
        torch.cuda.set_device(gpu_indices[0])

    # Log de GPUs detectadas
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"  GPU {i}: {name} ({mem:.2f} GB)")

    # DDP si múltiples GPUs
    if num_gpus > 1:
        logger.info(f"Multi-GPU mode: {num_gpus} devices → DistributedDataParallel")

    # Configuración de entrenamiento
    device_name = torch.cuda.get_device_name(0)
    if "K80" in device_name or "Tesla" in device_name:
        self.use_gradient_checkpointing = True
        torch.cuda.set_per_process_memory_fraction(0.9)
    else:
        self.use_gradient_checkpointing = False

    self.use_gpu = True
    self.use_mixed_precision = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return torch.device(f'cuda:{gpu_indices[0] if gpu_indices else 0}')
```

### _setup_model_with_device_strategy()

Reescribir para soportar DDP y CPU+GPU:

```python
def _setup_model_with_device_strategy(self, model, device):
    device_mode = self.config.device_mode
    gpu_indices = self.config.gpu_indices or []

    # --- CPU puro ---
    if device_mode == 'cpu':
        model = model.to(device)
        logger.info("Model deployed on CPU")
        return model

    # --- CPU+GPU: distribución por capas ---
    if device_mode == 'cpu+gpu':
        return self._setup_model_cpu_gpu_split(model, device)

    # --- GPU estándar (1 o DDP) ---
    try:
        model = model.to(device)

        # DDP si múltiples GPUs
        if len(gpu_indices) > 1:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            if not dist.is_initialized():
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                dist.init_process_group(backend='nccl', rank=0, world_size=len(gpu_indices))

            model = DDP(model, device_ids=gpu_indices)
            logger.info(f"Model wrapped with DDP on devices: {gpu_indices}")

        if self.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        logger.info(f"Model deployed on GPU: {device}")
        return model

    except RuntimeError as e:
        logger.warning(f"Could not move model to GPU: {e}, falling back to CPU")
        model = model.to(torch.device("cpu"))
        self.use_gpu = False
        self.use_mixed_precision = False
        return model


def _setup_model_cpu_gpu_split(self, model, gpu_device):
    """Distribute model layers across CPU and GPU based on VRAM budget."""
    import gc

    model = model.to('cpu')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        transformer_layers = model.model.transformer.h
    else:
        logger.warning("Cannot split model: no transformer layers found. Using GPU only.")
        return model.to(gpu_device)

    num_layers = len(transformer_layers)
    gpu_memory_gb = torch.cuda.get_device_properties(gpu_device).total_memory / 1e9
    vram_budget = gpu_memory_gb * 0.80  # 80% usable

    # Estimate memory per layer (rough: params * 4 bytes FP32)
    total_params = sum(p.numel() for p in transformer_layers.parameters())
    params_per_layer = total_params / num_layers
    mem_per_layer_gb = (params_per_layer * 4) / 1e9

    # Embeddings + lm_head memory
    embed_params = sum(p.numel() for p in model.model.transformer.wte.parameters())
    embed_params += sum(p.numel() for p in model.model.transformer.wpe.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    overhead_gb = ((embed_params + head_params) * 4) / 1e9

    # Calculate layers that fit
    available_for_layers = vram_budget - overhead_gb
    layers_on_gpu = min(num_layers, max(1, int(available_for_layers / mem_per_layer_gb)))

    logger.info("=" * 80)
    logger.info("CPU+GPU MODEL SPLIT")
    logger.info(f"  Total layers: {num_layers}")
    logger.info(f"  GPU VRAM: {gpu_memory_gb:.2f} GB (budget: {vram_budget:.2f} GB)")
    logger.info(f"  Overhead (embeddings + head): {overhead_gb:.4f} GB")
    logger.info(f"  Layers on GPU: {layers_on_gpu}")
    logger.info(f"  Layers on CPU: {num_layers - layers_on_gpu}")
    logger.info("=" * 80)

    # Move embeddings + head to GPU
    model.model.transformer.wte = model.model.transformer.wte.to(gpu_device)
    model.model.transformer.wpe = model.model.transformer.wpe.to(gpu_device)
    model.lm_head = model.lm_head.to(gpu_device)

    # Move first N layers to GPU
    for i in range(layers_on_gpu):
        transformer_layers[i] = transformer_layers[i].to(gpu_device)

    # Store mapping for forward pass
    self._gpu_layers_count = layers_on_gpu
    self._gpu_device = gpu_device

    return model
```

---

## ChatEngine — Device Setup (inference/chat_engine.py)

### get_device()

Reescribir:

```python
def get_device(self):
    device_mode = self.config.device_mode
    use_vulkan = self.config.use_vulkan

    if device_mode == 'cpu':
        logger.info("CPU-only mode enabled")
        return torch.device('cpu')

    if use_vulkan:
        if check_vulkan_available():
            logger.info("Using Vulkan backend")
            return torch.device('vulkan')
        else:
            raise RuntimeError("Vulkan not available")

    if torch.cuda.is_available():
        gpu_indices = self.config.gpu_indices
        if gpu_indices:
            device = torch.device(f'cuda:{gpu_indices[0]}')
            logger.info(f"Using CUDA device {gpu_indices[0]}: {torch.cuda.get_device_name(gpu_indices[0])}")
        else:
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
        return torch.device('mps')

    logger.info("No accelerator available, using CPU")
    return torch.device('cpu')
```

### CPU+GPU Inference

En `__init__`, después de cargar el modelo, si `device_mode == 'cpu+gpu'`:

```python
if self.config.device_mode == 'cpu+gpu':
    self._setup_model_cpu_gpu_split(model, device)
```

Usar la misma lógica de `_setup_model_cpu_gpu_split` del Trainer (extraer a `device_utils.py` como función compartida).

---

## main.py — Integración

### Validación (validate_arguments)

Agregar al final de la función:

```python
# Device mutual exclusion
if getattr(args, 'cpu', False) and getattr(args, 'gpu', None) is not None:
    return False, "--cpu and --gpu are mutually exclusive"

if getattr(args, 'vulkan', False) and getattr(args, 'gpu', None) is not None:
    return False, "--vulkan and --gpu are mutually exclusive"

if getattr(args, 'vulkan', False) and getattr(args, 'cpu', False):
    return False, "--vulkan and --cpu are mutually exclusive"
```

### Resolución de device_mode (después de parse_args)

```python
# Resolve device mode from CLI args
if args.cpu:
    args.device_mode = 'cpu'
    args.gpu_indices = None
elif args.gpu is not None:
    if args.gpu == '':
        args.device_mode = 'gpu'
        args.gpu_indices = None  # auto-detect
    else:
        args.device_mode = 'gpu'
        args.gpu_indices = [int(x) for x in args.gpu.split(',')]
elif args.vulkan:
    args.device_mode = 'gpu'  # vulkan handled by use_vulkan flag
    args.gpu_indices = None
    args.use_vulkan = True
else:
    args.device_mode = 'auto'
    args.gpu_indices = None
    args.use_vulkan = False
```

### --gpu-enum (antes de validate_arguments)

```python
if getattr(args, 'gpu_enum', False):
    from commons.utils.device_utils import enumerate_gpus
    gpus = enumerate_gpus()
    if not gpus:
        print("No GPUs detected.")
    else:
        print("=" * 60)
        print("GPU ENUMERATION")
        print("=" * 60)
        for g in gpus:
            rec = "RECOMMENDED" if g['ai_recommended'] else "NOT RECOMMENDED"
            print(f"[{g['index']}] {g['name']}")
            print(f"    VRAM: {g['vram_total_gb']:.2f} GB ({g['vram_free_gb']:.2f} GB free)")
            print(f"    Compute Capability: {g['compute_capability'][0]}.{g['compute_capability'][1]}")
            print(f"    Status: {g['status']}")
            print(f"    AI Recommendation: {rec}")
            print()
        print("=" * 60)
    sys.exit(0)
```

### TrainingConfig / ChatConfig (líneas 434-448, 484-491)

```python
config = TrainingConfig(
    epochs=args.epochs,
    checkpoint_name=args.checkpoint_name,
    dataset_source=args.dataset,
    device_mode=args.device_mode,
    gpu_indices=args.gpu_indices,
    use_vulkan=getattr(args, 'use_vulkan', False),
    num_cores=args.num_cores,
    num_threads=args.num_threads,
    max_ram_fraction=args.max_ram_fraction,
    max_ram_bytes=getattr(args, 'max_ram_bytes', None),
    thinking_loss_weight=args.thinking_loss_weight,
    thinking_enabled=args.thinking_enabled,
    thinking_max_tokens=args.thinking_max_tokens,
    statistics=args.statistics,
)
```

---

## Orden de Ejecución

1. `commons/utils/device_utils.py` — utilidad compartida (base)
2. `main.py` — argumentos CLI, validación, resolución, `--gpu-enum`
3. `training/trainer.py` — TrainingConfig + device setup + DDP + CPU+GPU
4. `inference/chat_engine.py` — ChatConfig + get_device() + CPU+GPU
5. Verificar linting/typecheck

---

## Distribución DDP (Multi-GPU)

Cuando `len(gpu_indices) > 1`:

1. `os.environ['MASTER_ADDR'] = 'localhost'`
2. `os.environ['MASTER_PORT'] = '12355'`
3. `dist.init_process_group(backend='nccl', rank=0, world_size=len(gpu_indices))`
4. `model = DDP(model, device_ids=gpu_indices)`
5. Forward: `model(inputs)` → DDP replica y sincroniza gradientes en backward
6. Cleanup: `dist.destroy_process_group()` al finalizar entrenamiento

**Nota**: El training loop existente (líneas 648-649) ya mueve inputs/targets a device. Con DDP, `model(inputs)` funciona porque DDP maneja la distribución internamente.

---

## CPU+GPU — Estrategia de Distribución

Por VRAM disponible:

1. Mover modelo completo a CPU
2. Calcular VRAM disponible × 0.80 (budget)
3. Restar memoria de embeddings + lm_head
4. Dividir restante por memoria por capa transformer
5. Mover primeras N capas + embeddings + head a GPU
6. Las demás capas se quedan en CPU

**Forward pass**: PyTorch maneja automáticamente los transfers CPU↔GPU entre capas (con overhead de transferencia).

---

*Fecha: 2026-08-05*
