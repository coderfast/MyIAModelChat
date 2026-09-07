# Project Use Common Machine Learning Algorithms

MyIAModelChat - Algoritmos de ML utilizados vs. disponibles

---

## Algoritmos del MD que **SÍ se usan** en el proyecto

| Algoritmo del MD | Uso en el proyecto | Archivo |
|---|---|---|
| **Neural Networks** | GPT-2 Transformer (modelo principal), BERT (sentiment/intent), MoE, MTP | `commons/model/chatmodel.py`, `chatmodel_moe.py`, `chatmodel_mtp.py` |

---

## Algoritmos del MD que **NO se usan** pero podrían ser útiles

| Algoritmo | Podría usarse para | Viabilidad |
|---|---|---|
| **Naive Bayes** | Clasificación de intención/sentimiento ligera (alternativa a BERT) | Alta - ya existe `intent_classifier` y `sentiment_analyzer` en `dialogmanager.py` |
| **Logistic Regression** | Clasificación binaria de sentimiento o intención | Alta - simple, rápido, interpretable |
| **Random Forest** | Clasificación multi-clase de intención | Media - ensemble robusto |
| **SVM** | Clasificación de texto en alta dimensión | Media - efectivo para datasets pequeños |
| **K-Nearest Neighbors** | Búsqueda de similaridad semántica (retrieval) | Media - computacionalmente costoso |
| **K-Means Clustering** | Descubrimiento de tópicos, agrupación de datos | Media - útil para análisis exploratorio |
| **Decision Trees** | Clasificación interpretable de intención | Baja - propenso a overfitting |
| **Linear Regression** | Predicción de valores continuos | Baja - no aplica a tareas de generación |
| **Gradient Boosting (GBM)** | Clasificación de texto de alta precisión | Media - más pesado que Naive Bayes |

---

## Algoritmos **adicionales en el proyecto** que NO están en el MD

| Componente | Algoritmo real | Archivo |
|---|---|---|
| **BPE Tokenizer** | Byte Pair Encoding (SentencePiece) | `commons/tokenizer/bpe_tokenizer.py` |
| **MinHash LSH** | Deduplicación near-duplicate | `dataset_preparer/contamination/dedup.py` |
| **Top-K / Top-P Sampling** | Decodificación de texto | `commons/dialogue/dialogmanager.py:67` |
| **Mixture of Experts (MoE)** | Enrutamiento gating con múltiples expertos FFN | `commons/model/chatmodel_moe.py` |
| **Multi-Token Prediction (MTP)** | Predicción de múltiples tokens futuros | `commons/model/chatmodel_mtp.py` |
| **Speculative Decoding** | Draft model + llama.cpp para inferencia rápida | `inference/chat_engine.py` |
| **Adam / AdamW** | Optimizadores de gradientes | `training/trainer.py:3323` |
| **CosineAnnealing / OneCycleLR / etc.** | Learning rate scheduling | `training/trainer.py:3329-3337` |
| **Mixed Precision (AMP)** | Entrenamiento con FP16 | `training/trainer.py` |
| **Knowledge Distillation** | Entrenamiento del draft model | `training/trainer.py:3853` |
| **Source Balancing** | Balanceo de datos entre fuentes | `dataset_preparer/contamination/balance.py` |

---

## Algoritmos que **deberían agregarse** al MD original

El documento `Common_Machine_Learning_Algorithms.md` actual solo cubre 10 algoritmos clásicos. Faltan los siguientes que son relevantes para este proyecto:

### Deep Learning & Transformers
1. **Transformers / Attention Mechanisms** - El corazón de este proyecto (GPT-2, BERT)
2. **RNN / LSTM / GRU** - Redes recurrentes (predecesoras de Transformers)
3. **Autoencoders / VAE** - Codificación de representaciones latentes
4. **GANs** - Redes generativas adversarias
5. **Diffusion Models** - Modelos de difusión (generación)

### Entrenamiento & Optimización
6. **Gradient Accumulation** - Técnica de entrenamiento (ya usada)
7. **Mixed Precision Training** - AMP/FP16 (ya usado)
8. **Knowledge Distillation** - Transferencia de conocimiento (ya usado)
9. **RLHF (Reinforcement Learning from Human Feedback)** - Alineamiento de LLMs
10. **Bayesian Optimization** - Optimización de hiperparámetros

### Arquitecturas Avanzadas
11. **Mixture of Experts (MoE)** - Enrutamiento de expertos (ya usado)
12. **Multi-Token Prediction (MTP)** - Predicción múltiple (ya usado)
13. **Speculative Decoding** - Inferencia acelerada (ya usado)

### Procesamiento de Datos
14. **MinHash LSH** - Deduplicación probabilística (ya usado)
15. **HuggingFace Pipelines** - Framework de ML (ya usado)
