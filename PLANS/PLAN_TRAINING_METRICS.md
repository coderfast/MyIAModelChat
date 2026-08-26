# PLAN_TRAINING_METRICS — Métricas de Entrenamiento Mejoradas

> Basado en conversación con ChatGPT 5 mini sobre training/validation/inference.
> Objetivo:能看到 más allá del training loss para saber si el modelo está mejorando realmente.
> Status: **IMPLEMENTADO** (2026-08-26)

### Implemented Features
- Validation split with `--val-split`
- Early stopping with `--early-stopping-patience`
- CSV metrics logging with `--no-metrics-csv` to disable
- HTML report with Chart.js graphs (Loss, Perplexity, Gap, LR, Speed, Thinking, Agent, MoE)
- Green/red/yellow status indicators for training health
- Correlated epoch numbering (continues from last training session)
- Per-epoch `chat_model.pth` overwrite (not just at end)
- Python `None` → JavaScript `null` serialization in HTML report

---

## Problema actual

El trainer solo reporta **training loss** por epoch. No hay forma de saber:
- Si el modelo está **generalizando** o **sobreajustando** (overfitting)
- Qué tan "seguro" está el modelo (**perplexity**)
- Si el modelo está **estable** durante entrenamiento (**grad norm**)
- Historial de métricas para análisis posterior

---

## Métricas nuevas a implementar

| Métrica | Qué mide | Cuándo se calcula |
|---------|----------|-------------------|
| **Val loss** | Error en datos no vistos | Al final de cada epoch |
| **Perplexity** (train + val) | "Confianza" del modelo. Menor = mejor. `e^loss` | Al final de cada epoch |
| **Train/Val gap** | Detección de sobreajuste. Si `val_loss - train_loss` crece, hay overfitting | Al final de cada epoch |
| **Grad norm** | Estabilidad del entrenamiento (ya se calcula, solo falta loggear) | Cada optimizer step |
| **Tokens/seg** | Velocidad efectiva de entrenamiento | Cada epoch |
| **Mejor val loss** | Tracking del mejor modelo visto | Al final de cada epoch |

---

## Cambios en `TrainingConfig` (dataclass, línea ~109)

Agregar campos:

```python
val_split: float = 0.1              # 10% del dataset para validación
val_batches: int = 0                # 0 = usar todo el split; >0 = limitar batches de val
early_stopping_patience: int = 0    # 0 = deshabilitado; N = parar si val_loss no mejora en N epochs
log_metrics_csv: bool = True        # Guardar métricas en metrics.csv
```

---

## Cambios en `performMainTrain()` (línea ~1286)

### 1. Split train/validación

**Ubicación:** Después de cargar `_pre_tokenized_dataset` (línea ~1291), antes de crear el DataLoader.

**Lógica:**
- Calcular `val_size = int(len(dataset) * config.val_split)`
- Usar `dataset.train_test_split(test_size=val_size, seed=42)` de HuggingFace datasets
- Crear `train_dataset` y `val_dataset` separados
- Crear `val_dataloader` con el mismo `collate_fn` pero sin shuffle

**Nota:** El dataset actual usa `TokenPairIterableDataset` con streaming. Para el split, necesitamos materializar al menos una parte. Alternativa: usar un `torch.utils.data.Subset` o simplemente reservar los últimos N samples para validación.

**Enfoque propuesto:** Dado que el dataset es iterable y potencialmente grande, usar los últimos `val_size` samples como validación (no aleatorio, pero determinista y eficiente en memoria).

### 2. Nuevo método `validate()`

**Ubicación:** Después del método `train()` (línea ~1232).

**Firma:**
```python
def validate(self, model, dataloader, criterion, device, max_batches=0):
    """Run validation loop. Returns (avg_loss, perplexity)."""
```

**Lógica:**
```python
model.eval()
total_loss = 0
total_batches = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss, _ = self._compute_loss(model, inputs, targets, criterion)
        total_loss += loss.item()
        total_batches += 1

avg_loss = total_loss / max(1, total_batches)
perplexity = torch.exp(torch.tensor(avg_loss)).item()
model.train()
return avg_loss, perplexity
```

### 3. Epoch loop modificado (línea ~1539)

**Cambio:** Después de `loss = self.train(...)`, agregar validación y cálculo de métricas.

```python
for epoch in range(num_epochs):
    # ... stop check ...

    # --- TRAINING ---
    train_loss = self.train(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps)
    train_perplexity = torch.exp(torch.tensor(train_loss)).item()

    # --- VALIDATION ---
    val_loss, val_perplexity = 0.0, 0.0
    if val_dataloader is not None:
        val_loss, val_perplexity = self.validate(
            model, val_dataloader, criterion, device,
            max_batches=self.config.val_batches
        )

    # --- GRAD NORM (ya se calcula en _backward_pass, falta loggear) ---
    # Se obtiene del último optimizer step

    # --- METRICS ---
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    gap = val_loss - train_loss if val_dataloader else 0.0

    # Epoch time & tokens/sec
    epoch_time = time.time() - epoch_start
    tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0

    # --- LOG ---
    logger.info(
        f"Epoch {epoch+1:2d}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Perplexity: {train_perplexity:.2f}/{val_perplexity:.2f} | "
        f"Gap: {gap:+.4f} | "
        f"LR: {current_lr:.2e} | "
        f"Tokens/s: {tokens_per_sec:.0f}"
    )

    # --- CHECKPOINT (basado en val_loss si hay validación) ---
    best_metric = val_loss if val_dataloader else train_loss
    if best_metric < self.best_loss:
        self.best_loss = best_metric
        # ... guardar checkpoint ...

    # --- EARLY STOPPING ---
    if self.config.early_stopping_patience > 0 and val_dataloader:
        if best_metric < best_val_loss:
            best_val_loss = best_metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement in {patience_counter} epochs)")
                break
```

### 4. CSV logging

**Ubicación:** Al final de cada epoch, escribir a `metrics.csv`.

**Columnas:**
```
epoch,train_loss,val_loss,train_perplexity,val_perplexity,gap,lr,tokens_per_sec,grad_norm,best_loss
```

**Implementación:**
```python
import csv

metrics_file = os.path.join(MODEL_CHECKPOINT_DIR, f'{self.checkpoint_name}_metrics.csv')
# Escribir header si el archivo no existe
# Append row con todas las métricas del epoch
```

### 5. Grad norm logging

**Cambio en `_backward_pass()`** (línea ~999): Retornar el grad norm calculado por `clip_grad_norm_`.

```python
def _backward_pass(self, loss, optimizer, scaler, accumulation_step=1):
    loss = loss / accumulation_step
    if self.use_mixed_precision and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return loss * accumulation_step
```

El `grad_norm` ya se calcula en la línea 1148:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG['grad_clip_norm'])
```

Solo falta capturar el return value y loggearlo.

---

## Cambios en `train()` (método epoch, línea ~1012)

### Retorno enriquecido

**Actual:** `return avg_loss`

**Propuesto:** `return avg_loss, total_tokens, last_grad_norm`

Para que `performMainTrain` pueda usar这些 valores para logging.

---

## Salida en consola (ejemplo)

```
Epoch  1/30 | Train Loss: 4.2315 | Val Loss: 4.1892 | Perplexity: 68.9/65.9 | Gap: -0.0423 | LR: 1.0e-03 | Tokens/s: 12450 | Grad: 0.85
Epoch  2/30 | Train Loss: 3.8764 | Val Loss: 3.9201 | Perplexity: 48.3/50.4 | Gap: +0.0437 | LR: 9.2e-04 | Tokens/s: 12680 | Grad: 0.72
...
Epoch 15/30 | Train Loss: 2.1045 | Val Loss: 2.8934 | Perplexity: 8.2/18.1 | Gap: +0.7889 | LR: 2.1e-04 | Tokens/s: 12590 | Grad: 0.45
Early stopping at epoch 15 (no improvement in 5 epochs)
```

**Cómo leerlo:**
- **Train Loss bajando, Val Loss bajando** → Modelo está aprendiendo y generalizando
- **Train Loss bajando, Val Loss subiendo** → Overfitting (el modelo memoriza)
- **Gap positivo y creciente** → Overfitting
- **Gap negativo o cerca de 0** → Buena generalización
- **Perplexity** → "Cuántas opciones considera el modelo". 8 = bastante seguro, 68 = muy incierto
- **Grad norm** → Estabilidad. Si se dispara >10, hay explosión de gradientes

---

## Archivos a modificar

| Archivo | Cambios |
|---------|---------|
| `training/trainer.py` | TrainingConfig, performMainTrain, train(), validate(), _backward_pass, CSV logging |
| `main.py` | Pasar nuevos config params desde CLI args (val_split, early_stopping_patience, etc.) |

---

## Orden de implementación

1. Agregar campos a `TrainingConfig`
2. Implementar método `validate()`
3. Modificar `performMainTrain()` para split + validation loop
4. Modificar `train()` para retornar métricas adicionales
5. Agregar logging CSV
6. Agregar early stopping
7. Modificar `main.py` para nuevos args
8. Probar con `python main.py --train --epochs 5`

---

## Flags CLI nuevos en `main.py`

```
--val-split FLOAT          Validation split ratio (default: 0.1, 0 = no validation)
--val-batches INT          Max validation batches per epoch (default: 0 = all)
--early-stopping-patience INT  Stop if no improvement in N epochs (default: 0 = disabled)
--no-metrics-csv           Disable CSV metrics logging
```
