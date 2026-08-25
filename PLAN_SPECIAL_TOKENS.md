## ESQUEMA DE TOKENIZER + SPECIAL TOKENS (GPT-2 Estandar OpenAI)

### Problemas Actuales Detectados en el Proyecto

1. Tokens duplicados semanticamente: <|thinking> vs <|thinking|> generan confusion.
   Un token abre bloque de razonamiento y otro indica modo.
   GPT-2 original NO usa esta duplicacion.
2. Tokens legacy innecesarios: <|context|>, <|answer|>, <observation>, </observation>.
   Deberian ser eliminados o mapeados completamente a los tokens estandar.
3. Tool calls sin delimitador de resultado consistente.
4. Falta token de fin de secuencia estandar en todos los formatos.
5. No hay separacion clara entre reasoning vs respuesta final en modo thinking.

### Estandarizacion Propuesta

#### Categoria 1: Tokens Base GPT-2 (vocabulario original)

GPT-2 original (OpenAI, 2019) tiene un solo special token relevante:
- ENDOFTEXT (ID 50256): fin de documento. Se uso para concatenar documentos en WebText.
- Serve tanto como BOS como EOS (es simetrico).

Para nuestro proyecto, ENDOFTEXT sera el terminador universal de secuencia en los 3 formatos.

#### Categoria 2: Tokens de Mensaje (ChatML estilo OpenAI)

OpenAI ChatML (GPT-3.5/4) usa <|im_start|> + role + content + <|im_end|>.
Llama 3 usa <|start_header_id|> + role + <|end_header_id|>.
Nosotros usamos el formato pipe-delimited <|role|> que es comun en GPT-2 custom:

Tokens de Mensaje a MANTENER (consolidados):
- <|user|>         - Inicio turno del usuario
- <|assistant|>    - Inicio turno del asistente
- <|system|>       - Inicio instrucciones del sistema (NUEVO)
- <|end|>          - Fin de mensaje/turno (NUEVO)

Tokens de Mensaje a ELIMINAR:
- <|context|>      -> Reemplazado por <|problem|> o <|system|>
- <|answer|>       -> Reemplazado por <|final|>

#### Categoria 3: Tokens de Reasoning (Thinking)

Para chain-of-thought, OpenAI usa thinking invisible. Nosotros lo hacemos visible:

Tokens a MANTENER:
- <|problem|>      - Prefijo de pregunta/problema
- <|thinking|>     - Inicia bloque de razonamiento (consolidar: eliminar <thinking>)
- <|final|>        - Inicia respuesta final (despues del reasoning)

Tokens a ELIMINAR:
- <thinking>       -> Duplicado de <|thinking|>, eliminar completamente
- </thinking>      -> Reemplazar con <|final|> como delimitador de fin de reasoning

Razon: En GPT-2 estandar, los special tokens usan formato <|name|> (pipe-delimited).
Los tags XML-style (<thinking>) no son nativos de GPT-2 y se rompen en subtokens BPE.
Mantener un solo token <|thinking|> reduce confusion y ahorra vocabulario.

#### Categoria 4: Tokens de Herramientas (Agentic/Tools)

El formato actual es funcional pero necesita un delimitador de resultado:

Tokens a MANTENER:
- <|user|>                  - Turno del usuario
- <|assistant|>             - Turno del asistente
- <tool_call>                  - Inicio llamada a herramienta
- </tool_call>               - Fin llamada a herramienta (cierra el nombre de tool)
- <|tool_result|>           - Inicio del resultado de la herramienta
- <|end|>                   - Fin del mensaje (NUEVO)

Problema actual: el tool_result actual cierra con </tool_result> que es XML-style.
Solucion: usar <|tool_result|> como prefijo del resultado, y </tool_call> como cierre.

#### Categoria 5: Tokens de Control Adicionales (NUEVOS)

- <|end|>          - Fin de mensaje/respuesta (nuevo, alineado con ChatML <|im_end|>)
- <|system|>       - Instrucciones del sistema (nuevo, para prompts de sistema)
- <|sep|>          - Separador intra-mensaje (nuevo, para estructura interna)

### Tabla Completa de Special Tokens (Version Final Consolidada)

ID  | Token             | Categoria   | Estado   | Descripcion
----|-------------------|-------------|----------|------------------------------------------
50256 | ENDOFTEXT       | Base GPT-2  | MANTENER | Fin de secuencia universal
S1  | <|system|>      | Mensaje     | NUEVO    | Inicio instrucciones del sistema
S2  | <|user|>        | Mensaje     | MANTENER | Inicio turno del usuario
S3  | <|assistant|>   | Mensaje     | MANTENER | Inicio turno del asistente
S4  | <|end|>         | Mensaje     | NUEVO    | Fin de mensaje/turno
S5  | <|sep|>         | Mensaje     | NUEVO    | Separador intra-mensaje
S6  | <|problem|>     | Reasoning   | MANTENER | Prefijo de pregunta/problema
S7  | <|thinking|>    | Reasoning   | MANTENER | Inicia bloque de reasoning
S8  | <|final|>       | Reasoning   | MANTENER | Inicia respuesta final
S9  | <tool_call>      | Tool call   | MANTENER | Inicio llamada a herramienta
S10 | </tool_call>   | Tool call   | MANTENER | Fin llamada a herramienta
S11 | <|tool_result|> | Tool call   | MANTENER | Prefijo resultado de herramienta

Tokens a ELIMINAR definitivamente:
- <|context|>      -> Unificado en <|problem|>
- <|answer|>       -> Unificado en <|final|>
- <thinking>       -> Unificado en <|thinking|>
- </thinking>      -> Reemplazado por <|final|>
- <observation>    -> Unificado en <|tool_result|>
- </observation>   -> Unificado en <|tool_result|>

### Plantillas de Formato Consolidadas

#### Formato 1: Texto Normal
```
<|system|>Eres un asistente util.<|end|>
<|user|>{pregunta}<|end|>
<|assistant|>{respuesta}<|end|>
```

Ejemplo JSONL:
{"text":"<|system|>Eres un asistente util.<|end|><|user|>Que es Python?<|end|><|assistant|>Python es un lenguaje de programacion.<|end|>"}

#### Formato 2: Agentic (Tools)
```
<|user|>{pregunta}<|end|>
<|assistant|><tool_call>{tool_name}({args})</tool_call><|tool_result|>{resultado}<|end|>
<|assistant|>{respuesta_final}<|end|>
```

Ejemplo JSONL:
{"text":"<|user|>Busca restaurantes cerca de mi<|end|><|assistant|><tool_call>search_places(ubicacion_actual)</tool_call><|tool_result|>3 resultados encontrados<|end|><|assistant|>Encontre 3 opciones cerca de ti.<|end|>"}

#### Formato 3: Thinking (Chain-of-Thought)
```
<|problem|>{problema}
<|thinking|>{razonamiento paso a paso}
<|final|>{respuesta}
```

Ejemplo JSONL:
{"text":"<|problem|>Si Ana tiene 5 libros y compra 2 mas, cuantos tiene?<|thinking|>Empiezo con 5 y sumo 2 = 7.<|final|>7 libros."}

NOTA: El formato thinking NO usa <|user|>/<|assistant|> porque es un formato de
completacion de texto puro, no de conversacion. Esto permite entrenamiento mas eficiente.

### Plan de Implementacion

FASE 1: Consolidar tokens en el tokenizer (Prioridad ALTA)
1.1 En bpe_tokenizer.py: mapear <thinking> y </thinking> a IDs de <|thinking|> y <|final|>
1.2 En gpt2_tokenizer.py: misma consolidacion
1.3 Eliminar mapeos legacy: <|context|> -> <|problem|>, <|answer|> -> <|final|>
1.4 Eliminar mapeos legacy: <observation> -> <|tool_result|>
1.5 Anyadir tokens nuevos: <|system|>, <|end|>, <|sep|>

FASE 2: Actualizar generadores de dataset (Prioridad ALTA)
2.1 Actualizar dataset_preparer/data_preparer.py para generar los 3 formatos correctos
2.2 Actualizar dataset_preparer/thinking_engine.py para usar <|thinking|>/<|final|> (no <thinking>)
2.3 Actualizar dataset_preparer/aiml/ para generar formato con <|end|>
2.4 Anyadir campo 'format_type' al metadata del dataset: normal, agentic, thinking

FASE 3: Actualizar el modelo y trainer (Prioridad MEDIA)
3.1 Actualizar loss weights en training/trainer.py para el nuevo esquema
3.2 Actualizar chat_engine.py para parsear los nuevos tokens
3.3 Actualizar dialogmanager.py para generar prompts con <|system|>, <|end|>

FASE 4: Migracion y backward compatibility (Prioridad BAJA)
4.1 Anyadir migrador de datasets viejos a formato nuevo
4.2 Mantener tokens legacy como deprecated durante 1-2 versiones
4.3 Actualizar tests

### Por que este esquema sigue el estandar GPT-2 de OpenAI

1. Pipe-delimited tokens (<|name|>) son el formato nativo de GPT-2 para special tokens.
   OpenAI uso <|startoftext|> y  como unico token original.
   El formato <|name|> es la extension estandar de la comunidad.

2. Un solo token por concepto (sin duplicados).
   GPT-2 original tiene solo 1 token por funcion. No hay <|start|> Y <start>.
   Nosotros tenemos <|thinking|> Y <thinking> - esto es incorrecto.

3. Tokens XML-style rompen en BPE.
   <thinking> se tokeniza como < + thin + king + > en BPE.
   <|thinking|> se mantiene como token atomico gracias a los pipes.

4. Alineacion con ChatML (estandar moderno).
   OpenAI ChatML: <|im_start|>role + content + <|im_end|>
   Nuestro formato: <|user|>content<|end|> es la misma filosofia.

5. Separacion clara de concerns.
   Tokens de mensaje (user/assistant/end) vs tokens de formato (thinking/final)
   vs tokens de tooling (tool_call/tool_result) = 3 categorias distintas.

### Presupuesto de Vocabulario

GPT-2 tiene vocabulario de 50,257 tokens. Los special tokens se anyaden arriba de este rango.
Con 11 tokens propuestos (incluyendo ENDOFTEXT que ya existe), necesitamos:
- 10 posiciones nuevas para special tokens custom
- Esto es 0.02% del vocabulario total. Despreciable.
- SentencePiece: los tokens se anyaden con --add_dummy_prefix=false para que sean atomicos.

### Convenciones de Tokenizacion para cada formato

SentencePiece BPE:
- Anyadir tokens con --user_defined_symbols=<|system|>,<|user|>,<|assistant|>,<|end|>,...
- Usar --add_dummy_prefix=false para tokens speciales (evita espacio prefix)
- Vocab size recomendado: 32000 (multilingue) o 8000 (monolingue)


--- FIN DEL PLAN ---
