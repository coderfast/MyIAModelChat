# Guia de Entrenamiento Agentic

## Que es el entrenamiento agentic?

El entrenamiento agentic le ensena al modelo a **usar herramientas** para responder preguntas. En vez de solo dar una respuesta, el modelo aprende a:

1. Detectar que necesita una herramienta
2. Llamar a la herramienta con los parametros correctos
3. Recibir el resultado
4. Dar la respuesta final

## Formato de texto para entrenar

### Formato basico (sin herramienta)

Cuando el modelo NO necesita herramienta:

```
<|user|>¿Que es Python?<|end|><|assistant|><|thinking|>El usuario pregunta sobre Python. Es un lenguaje de programacion. No necesito herramientas.<|final|>Python es un lenguaje de programacion de alto nivel, facil de aprender y muy versatil.<|end|>
```

### Formato con herramienta

Cuando el modelo SÍ necesita herramienta:

```
<|user|>¿Cuanto es 15 + 27?<|end|><|assistant|><tool_call>calculator(15 + 27)</tool_call><|tool_result|>42<|end|><|assistant|><|thinking|>El usuario quiere sumar 15 + 27. Use la calculadora y el resultado es 42.<|final|>15 + 27 = 42<|end|>
```

## Herramientas disponibles

### 1. Calculator (Calculadora)

**Para que sirve:** Evaluar expresiones matematicas.

**Ejemplos de texto para entrenar:**

```
<|user|>¿Cuanto es 5 * 3?<|end|><|assistant|><tool_call>calculator(5 * 3)</tool_call><|tool_result|>15<|end|><|assistant|><|thinking|>Multiplico 5 por 3. El resultado es 15.<|final|>5 * 3 = 15<|end|>
```

```
<|user|>Calcula 100 / 4<|end|><|assistant|><tool_call>calculator(100 / 4)</tool_call><|tool_result|>25<|end|><|assistant|><|thinking|>Divido 100 entre 4. El resultado es 25.<|final|>100 / 4 = 25<|end|>
```

```
<|user|>Suma 23 + 45 + 12<|end|><|assistant|><tool_call>calculator(23 + 45 + 12)</tool_call><|tool_result|>80<|end|><|assistant|><|thinking|>Sumo 23 + 45 + 12. El resultado es 80.<|final|>23 + 45 + 12 = 80<|end|>
```

**Keywords que activan la herramienta:**
- "cuanto es", "calcula", "suma", "resta", "multiplica"
- "how much is", "calculate", "sum", "subtract", "multiply"

---

### 2. Current Date (Fecha Actual)

**Para que sirve:** Obtener la fecha y hora actual.

**Ejemplos de texto para entrenar:**

```
<|user|>¿Que dia es hoy?<|end|><|assistant|><tool_call>current_date()</tool_call><|tool_result|>2026-09-02 12:30:45<|end|><|assistant|><|thinking|>El usuario quiere saber la fecha de hoy. La fecha actual es 2 de septiembre de 2026.<|final|>Hoy es 2 de septiembre de 2026.<|end|>
```

```
<|user|>What day is it today?<|end|><|assistant|><tool_call>current_date()</tool_call><|tool_result|>2026-09-02 12:30:45<|end|><|assistant|><|thinking|>The user wants to know today's date. The current date is September 2, 2026.<|final|>Today is September 2, 2026.<|end|>
```

```
<|user|>¿Cual es la fecha actual?<|end|><|assistant|><tool_call>current_date()</tool_call><|tool_result|>2026-09-02 12:30:45<|end|><|assistant|><|thinking|>El usuario quiere saber la fecha actual. Es 2 de septiembre de 2026.<|final|>La fecha actual es 2 de septiembre de 2026.<|end|>
```

**Keywords que activan la herramienta:**
- "que dia es", "hoy", "fecha actual", "que fecha es"
- "what day is it", "today", "current date", "what date is it"

---

### 3. List Directory (Listar Carpeta)

**Para que sirve:** Ver los archivos de una carpeta.

**Ejemplos de texto para entrenar:**

```
<|user|>¿Que hay en la carpeta documents?<|end|><|assistant|><tool_call>list_directory(documents)</tool_call><|tool_result|>file1.txt  file2.py  directory1/<|end|><|assistant|><|thinking|>El usuario quiere ver el contenido de la carpeta documents. Los archivos son file1.txt, file2.py y la subcarpeta directory1.<|final|>En la carpeta documents hay: file1.txt, file2.py y la subcarpeta directory1/<|end|>
```

```
<|user|>List files in current directory<|end|><|assistant|><tool_call>list_directory(.)</tool_call><|tool_result|>file1.txt  file2.py  directory1/<|end|><|assistant|><|thinking|>The user wants to list files in the current directory. There are file1.txt, file2.py and directory1/<|final|>Current directory contains: file1.txt, file2.py, and directory1/<|end|>
```

```
<|user|>Muestrame el contenido de la carpeta src<|end|><|assistant|><tool_call>list_directory(src)</tool_call><|tool_result|>main.py  utils.py  models/<|end|><|assistant|><|thinking|>El usuario quiere ver la carpeta src. Contiene main.py, utils.py y models/.<|final|>La carpeta src contiene: main.py, utils.py y models/<|end|>
```

**Keywords que activan la herramienta:**
- "que hay en", "lista archivos", "carpeta", "contenido de"
- "what's in", "list files", "directory", "contents of"

---

### 4. Read File (Leer Archivo)

**Para que sirve:** Leer el contenido de un archivo.

**Ejemplos de texto para entrenar:**

```
<|user|>Lee el archivo config.py<|end|><|assistant|><tool_call>read_file(config.py)</tool_call><|tool_result|>import os\nDEBUG = True\nSECRET_KEY = "abc123"<|end|><|assistant|><|thinking|>El usuario quiere leer config.py. El contenido es un archivo de configuracion con DEBUG=True y SECRET_KEY.<|final|>El archivo config.py contiene configuracion de Python con DEBUG=True y una clave secreta.<|end|>
```

```
<|user|>Read file README.md<|end|><|assistant|><tool_call>read_file(README.md)</tool_call><|tool_result|># My Project\n\nThis is a Python project...<|end|><|assistant|><|thinking|>The user wants to read README.md. It contains a project description.<|final|>The README.md file contains the project description and documentation.<|end|>
```

```
<|user|>¿Que dice el archivo requirements.txt?<|end|><|assistant|><tool_call>read_file(requirements.txt)</tool_call><|tool_result|>numpy==1.24.0\ntorch==2.0.0\ntransformers==4.30.0<|end|><|assistant|><|thinking|>El usuario quiere ver requirements.txt. Contiene las dependencias del proyecto.<|final|>El archivo requirements.txt lista las dependencias: numpy, torch y transformers.<|end|>
```

**Keywords que activan la herramienta:**
- "leer archivo", "contenido de", "que dice"
- "read file", "contents of", "what does the file say"

---

### 5. Web Search (Buscar en la Web)

**Para que sirve:** Buscar informacion en internet.

**Ejemplos de texto para entrenar:**

```
<|user|>Busca informacion sobre Python 3.12<|end|><|assistant|><tool_call>web_search(Python 3.12)</tool_call><|tool_result|>Python 3.12 includes new features like improved error messages, f-string improvements, and performance enhancements.<|end|><|assistant|><|thinking|>El usuario quiere buscar info sobre Python 3.12. Los resultados muestran que tiene nuevas funcionalidades.<|final|>Python 3.12 incluye nuevas funcionalidades como mensajes de error mejorados, mejoras en f-strings y optimizaciones de rendimiento.<|end|>
```

```
<|user|>Search for PyTorch documentation<|end|><|assistant|><tool_call>web_search(PyTorch documentation)</tool_call><|tool_result|>PyTorch is an open source machine learning framework...<|end|><|assistant|><|thinking|>The user wants to search for PyTorch docs. Found official documentation link.<|final|>You can find the official PyTorch documentation at pytorch.org/docs.<|end|>
```

```
<|user|>Investiga sobre transformers en IA<|end|><|assistant|><tool_call>web_search(transformers en IA)</tool_call><|tool_result|>Transformers are a type of neural network architecture used in NLP...<|end|><|assistant|><|thinking|>El usuario quiere investigar sobre transformers. Es un tipo de arquitectura de red neuronal para NLP.<|final|>Los transformers son una arquitectura de red neuronal utilizada en procesamiento de lenguaje natural, popularizada por modelos como BERT y GPT.<|end|>
```

**Keywords que activan la herramienta:**
- "busca", "buscar", "investiga", "que es"
- "search", "look up", "research", "what is"

---

### 6. Shell (Comandos del Sistema)

**Para que sirve:** Ejecutar comandos del sistema operativo.

**Ejemplos de texto para entrenar (Linux/Mac):**

```
<|user|>Ejecuta ls -la<|end|><|assistant|><tool_call>shell(ls -la)</tool_call><|tool_result|>total 32\ndrwxr-xr-x 5 user user 4096 Sep 2 12:30 .\ndrwxr-xr-x 3 user user 4096 Sep 2 12:00 ..\n-rw-r--r-- 1 user user  123 Sep 2 12:30 file.txt<|end|><|assistant|><|thinking|>El usuario quiere ejecutar ls -la. Muestra los archivos con permisos y detalles.<|final|>El comando ls -la muestra los archivos de la carpeta actual con sus permisos, tamaño y fecha.<|end|>
```

```
<|user|>Run pwd<|end|><|assistant|><tool_call>shell(pwd)</tool_call><|tool_result|>/home/user/project<|end|><|assistant|><|thinking|>The user wants to know the current directory. pwd shows /home/user/project.<|final|>The current directory is /home/user/project.<|end|>
```

**Ejemplos de texto para entrenar (Windows):**

```
<|user|>Ejecuta dir<|end|><|assistant|><tool_call>shell(dir /b)</tool_call><|tool_result|>file.txt\nmain.py\nREADME.md<|end|><|assistant|><|thinking|>El usuario quiere ejecutar dir en Windows. Lista los archivos de la carpeta.<|final|>El comando dir muestra los archivos: file.txt, main.py y README.md.<|end|>
```

```
<|user|>Run ipconfig<|end|><|assistant|><tool_call>shell(ipconfig)</tool_call><|tool_result|>Ethernet adapter: IPv4 Address: 192.168.1.100<|end|><|assistant|><|thinking|>The user wants to see network configuration. ipconfig shows IP address 192.168.1.100.<|final|>Your IP address is 192.168.1.100.<|end|>
```

**Keywords que activan la herramienta:**
- "ejecuta", "corre comando", "abre"
- "run command", "execute", "open"

---

## Como generar datos de entrenamiento

### Paso 1: Preparar el dataset

Necesitas un dataset con pares de pregunta-respuesta:

```json
{"input": "¿Cuanto es 5 + 3?", "output": "5 + 3 = 8"}
{"input": "¿Que dia es hoy?", "output": "Hoy es 2 de septiembre de 2026."}
{"input": "Lista archivos", "output": "Hay 3 archivos en la carpeta."}
```

### Paso 2: Generar datos agentic

Ejecuta el generador de datos agentic:

```bash
python main.py --prepare-data --aiml --generate-agent-data --agent-ratio 0.3
```

El `--agent-ratio 0.3` significa que el 30% de los datos tendran llamadas a herramientas.

### Paso 3: Entrenar el modelo

```bash
python main.py --train --agent-enabled --epochs 30
```

## Consejos para datos de entrenamiento

1. **Balance:** No todos los textos necesitan herramientas. Mezcla textos con y sin herramientas.

2. **Variiedad:** Usa diferentes formas de hacer la misma pregunta:
   - "¿Cuanto es 5 + 3?"
   - "Calcula 5 + 3"
   - "Suma 5 y 3"
   - "How much is 5 + 3?"

3. **Observaciones realistas:** El resultado de la herramienta debe ser realista:
   - `<tool_call>calculator(2 + 2)</tool_call>` -> `<|tool_result|>4<|end|>`
   - NO: `<|tool_result|>no se<|end|>`

4. **Idiomas:** Entrena en español Y ingles para que el modelo sea bilingue.

5. **Errores:** Incluye ejemplos de errores para que el modelo aprenda a manejarlos:
   - `<tool_call>read_file(noexiste.txt)</tool_call>` -> `<|tool_result|>Error: File not found<|end|>`

## Estructura de archivos

```
tools_config.json          # Configuracion de herramientas
├── tools                  # Herramientas disponibles
├── agent_patterns         # Patrones para detectar cuando usar herramientas
└── shell_commands         # Comandos por plataforma (windows/linux/darwin)
```

## Preguntas frecuentes

**P: ¿Cada cuanto el modelo usa una herramienta?**
R: Depende del `--agent-ratio`. Si es 0.3, el 30% de las muestras tendran herramientas.

**P: ¿Puedo agregar mis propias herramientas?**
R: Si. Agrega la herramienta en `tools_config.json` y crea la funcion en `tool_registry.py`.

**P: ¿Que pasa si el modelo no sabe usar una herramienta?**
R: El modelo aprende por patron. Si ves que no usa bien las herramientas, agrega mas ejemplos de entrenamiento.

**P: ¿Las herramientas son seguras?**
R: Algunas herramientas (shell, read_file) requieren permiso. Configura `requires_permission: true` en el JSON.
