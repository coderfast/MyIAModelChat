# ROADMAP: Distributed Data Parallel (DDP) Multi-Nodo

## Estado Actual

**Fase 1 COMPLETADA** (2026-08-05): DDP single-machine funciona correctamente.

Ver seccion "Fase 1: Corregir DDP Single-Machine" abajo para detalles de la implementacion.

---

## Fase 1: Corregir DDP Single-Machine (COMPLETADA)

Objetivo: Hacer que DDP funcione correctamente en 1 maquina con multiples GPUs.

### Resumen de Implementacion (2026-08-05)

| Problema | Solucion |
|----------|----------|
| `rank=0` hardcodeado | TrainingConfig con campos `rank`, `local_rank`, `world_size` |
| Sin DistributedSampler | `TokenPairIterableDataset` con sharding por rank (`i % world_size == rank`) |
| Sin `no_sync()` | Context manager en gradient accumulation |
| Checkpoints con `module.` prefix | `model.module.state_dict()` en rank-0 |
| Sin `destroy_process_group()` | Cleanup en `finally` block |
| Logs duplicados | Todos los `logger.info()` protegidos con `if self.rank == 0` |
| `MASTER_ADDR=localhost` hardcodeado | Leer de environment o config |

### 1.1 TrainingConfig -- agregar campos DDP

Archivo: `training/trainer.py`

```python
@dataclass
class TrainingConfig:
    # ... campos existentes ...
    master_addr: str = 'localhost'
    master_port: int = 29500
    nnodes: int = 1
    node_rank: int = 0
    local_rank: int = 0        # rank dentro de esta maquina
    rank: int = 0              # rank global
    world_size: int = 1        # total de procesos
```

### 1.2 Refactorizar `_setup_device_and_config()`

- Leer `LOCAL_RANK`, `RANK`, `WORLD_SIZE` de variables de entorno (torchrun las setea)
- Llamar `torch.cuda.set_device(local_rank)` por proceso
- No hardcodear `MASTER_ADDR`/`MASTER_PORT` si ya estan en el environment

### 1.3 Refactorizar `_setup_model_with_device_strategy()`

- Mover modelo a `cuda:{local_rank}` en vez de `cuda:{gpu_indices[0]}`
- wrapping DDP: `DDP(model, device_ids=[local_rank])`
- Quitar hardcoded `rank=0` y `world_size=len(gpu_indices)`

### 1.4 DistributedSampler o sharding del IterableDataset

El dataset actual es `IterableDataset` (no soporta DistributedSampler estandar).

Opciones:
- **Opcion A**: Shard el generator por rank usando `rank` y `world_size` (modular slicing)
- **Opcion B**: Convertir a map-style dataset y usar `DistributedSampler`
- **Recomendada**: Opcion A (menos cambios, mantiene IterableDataset)

```python
# En el dataloader, filtrar por rank
def distributed_generator(generator, rank, world_size):
    for i, item in enumerate(generator):
        if i % world_size == rank:
            yield item
```

### 1.5 Gradient Accumulation con `no_sync()`

```python
for micro_step in range(accumulation_steps):
    is_last = (micro_step == accumulation_steps - 1)
    context = model.no_sync() if not is_last and world_size > 1 else nullcontext()
    with context:
        loss = compute_loss(...)
        backward(loss)
```

### 1.6 Checkpoints — solo rank-0

```python
if self.config.rank == 0:
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({...}, path)
if world_size > 1:
    dist.barrier()  # todos esperan a que rank-0 termine de guardar
```

### 1.7 Cleanup — `destroy_process_group()`

```python
try:
    # ... training loop ...
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
```

### 1.8 Logs — solo rank-0

Todos los `logger.info()` y `print()` deben ser:
```python
if self.config.rank == 0:
    logger.info("...")
```

---

## Fase 2: Multi-Nodo

Objetivo: Soportar entrenamiento distribuido en 2+ maquinas.

### 2.1 Nuevos argumentos CLI

Archivo: `main.py`

| Argumento | Propiedad | Default | Ejemplo |
|-----------|-----------|---------|---------|
| `--master-addr` | IP/hostname del nodo rank-0 | `localhost` | `--master-addr 192.168.1.100` |
| `--master-port` | Puerto NCCL rendezvous | `29500` | `--master-port 29500` |
| `--nnodes` | Numero de maquinas | `1` | `--nnodes 2` |
| `--node-rank` | Rank de esta maquina | `0` | `--node-rank 1` |
| `--nproc-per-node` | GPUs por maquina | auto | `--nproc-per-node 4` |

### 2.2 Soporte para torchrun

`torchrun` es el launcher estandar de PyTorch. Setea automaticamente:
- `RANK` — rank global
- `LOCAL_RANK` — rank dentro de la maquina
- `WORLD_SIZE` — total de procesos
- `MASTER_ADDR` — direccion del nodo maestro
- `MASTER_PORT` — puerto del nodo maestro

main.py debe:
1. Detectar si se ejecuto con `torchrun` (verificar `RANK` en env)
2. Si si: leer config de environment
3. Si no: usar argumentos CLI o modo single-process

```python
# En main.py, despues de parse_args
if 'RANK' in os.environ:
    # Ejecutado con torchrun
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    args.master_port = int(os.environ.get('MASTER_PORT', 29500))
    args.ddp = True
else:
    # Modo single-process
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.ddp = len(args.gpu_indices or []) > 1
```

### 2.3 Configuracion NCCL para cross-node

Variables de entorno necesarias para multi-nodo:

```bash
# Interfaz de red para comunicacion entre nodos
export NCCL_SOCKET_IFNAME=eth0  # o wlan0, eno1, etc.

# Deshabilitar InfiniBand si no esta disponible
export NCCL_IB_DISABLE=1

# Debug (opcional)
export NCCL_DEBUG=INFO
```

### 2.4 Flujo de ejecucion multi-nodo

**Nodo 0 (maestro):**
```bash
python main.py --train --epochs 30 --master-addr 192.168.1.100 --nnodes 2 --node-rank 0 --nproc-per-node 2
```

**Nodo 1 (trabajador):**
```bash
python main.py --train --epochs 30 --master-addr 192.168.1.100 --nnodes 2 --node-rank 1 --nproc-per-node 2
```

**O con torchrun (recomendado):**

Nodo 0:
```bash
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 --nproc_per_node=2 main.py --train --epochs 30
```

Nodo 1:
```bash
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=29500 --nproc_per_node=2 main.py --train --epochs 30
```

### 2.5 Firewall

Abrir puerto TCP entre nodos:
```bash
# Windows
netsh advfirewall firewall add rule name="NCCL" dir=in action=allow protocol=tcp localport=29500

# Linux
sudo ufw allow 29500/tcp
```

---

## Archivos a modificar

| Archivo | Cambios Fase 1 | Cambios Fase 2 |
|---------|----------------|----------------|
| `training/trainer.py` | TrainingConfig, device setup, DDP wrapping, checkpoints, cleanup, logs | Soportar multi-nodo via env vars |
| `main.py` | Pasar rank/local_rank/world_size a TrainingConfig | Nuevos args CLI, deteccion torchrun |
| `commons/utils/device_utils.py` | - | Funcion para resolver DDP config |

---

## Orden de ejecucion

### Fase 1
1. `training/trainer.py` — TrainingConfig con campos DDP
2. `training/trainer.py` — Refactorizar `_setup_device_and_config()` para leer rank de env
3. `training/trainer.py` — Refactorizar `_setup_model_with_device_strategy()` con local_rank
4. `training/trainer.py` — Shard del IterableDataset por rank
5. `training/trainer.py` — `no_sync()` para gradient accumulation
6. `training/trainer.py` — Checkpoints solo rank-0 con `model.module.state_dict()`
7. `training/trainer.py` — `destroy_process_group()` en cleanup
8. `training/trainer.py` — Logs solo rank-0
9. `main.py` — Soportar torchrun (leer RANK, LOCAL_RANK, WORLD_SIZE)

### Fase 2
1. `main.py` — Nuevos args: `--master-addr`, `--master-port`, `--nnodes`, `--node-rank`, `--nproc-per-node`
2. `commons/utils/device_utils.py` — Funcion `resolve_ddp_config()`
3. `training/trainer.py` — Configuracion NCCL cross-node
4. Documentacion de uso multi-nodo

---

## Pruebas

### Fase 1 — 1 maquina, 2+ GPUs
```bash
# Verificar que funciona con 2 GPUs
python main.py --train --gpu 0,1 --epochs 5

# Verificar checkpoints sin prefijo module.
python main.py --model-info chat_model
```

### Fase 2 — 2 maquinas
```bash
# Nodo 0 (192.168.1.100)
python main.py --train --gpu 0 --master-addr 192.168.1.100 --nnodes 2 --node-rank 0 --epochs 5

# Nodo 1 (192.168.1.101)
python main.py --train --gpu 0 --master-addr 192.168.1.100 --nnodes 2 --node-rank 1 --epochs 5
```

---

*Fecha: 2026-08-05*
