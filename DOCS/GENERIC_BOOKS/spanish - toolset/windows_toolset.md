# Windows Toolset: PowerShell y CMD

Guia de referencia para entrenar al modelo MyIAModelChat en el uso de la linea de comandos de Microsoft Windows. La guia cubre el interprete de comandos tradicional (cmd.exe, comunmente llamado CMD) y PowerShell (Windows PowerShell 5.1 y PowerShell 7 o superior), con ejemplos practicos, equivalencias entre ambos entornos, tareas frecuentes de administracion y buenas practicas de seguridad. Ademas incluye muestras de entrenamiento completas con los tokens especiales del modelo, incluida la invocacion agentic de la herramienta shell para ejecutar comandos de PowerShell y de CMD. El contenido se basa en la documentacion oficial de Microsoft (Microsoft Learn y la referencia de producto de Windows).

## Capitulo 1: La linea de comandos de Windows

### 1.1 Conceptos basicos

La linea de comandos es una interfaz de texto en la que el usuario escribe ordenes que un interprete ejecuta y devuelve en pantalla. A diferencia de una interfaz grafica, la linea de comandos permite automatizar tareas repetitivas, ejecutar operaciones que no tienen boton en las interfaces graficas, encadenar varios procesos en un unico flujo de trabajo y generar salidas que otras herramientas pueden leer. En Windows, la linea de comandos convive con el Explorer y puede usarse para practicamente cualquier tarea de administracion del sistema.

Windows dispone de dos interpretes de comandos principales. El primero es cmd.exe, el interprete heredado de MS-DOS, que trabaja con texto plano y secuencias de ordenes ejecutadas una tras otra. El segundo es PowerShell, creado por Microsoft como interprete basado en objetos y orientado a la automatizacion de administracion de sistemas. Ambos se ejecutan en una ventana de consola y comparten conceptos como las variables de entorno, las rutas de acceso y las redirecciones, pero su sintaxis y su forma de trabajar son muy diferentes.

La linea de comandos es especialmente util cuando hay que repetir una tarea muchas veces, cuando hay que trabajar con cientos o miles de ficheros, cuando se necesita generar informes automaticos o cuando se depura un problema del sistema y se requiere informacion que la interfaz grafica no muestra. Microsoft documenta ambas herramientas en Microsoft Learn, en la referencia de comandos de Windows Server y en la guia de scripting de PowerShell.

### 1.2 CMD y PowerShell: diferencias principales

La diferencia fundamental reside en lo que devuelven los comandos. cmd.exe trabaja con cadenas de texto: la salida de un comando es una cadena que el siguiente comando puede leer, pero que hay que analizar con expresiones regulares o con comandos como findstr si se necesita extraer datos concretos. PowerShell trabaja con objetos: la salida de un cmdlet es una coleccion de objetos con propiedades, que se pueden filtrar, ordenar, agrupar y formatear sin volver a analizar el texto.

Otra diferencia es la sintaxis. CMD utiliza operadores con barra diagonal, como /s o /q, y las variables se expanden entre signos de porcentaje. PowerShell utiliza parametros con guion, como -Recurse o -Force, cmdlets con nombre de verbo y sustantivo, por ejemplo Get-ChildItem, y las variables se escriben con el signo de dolar. En CMD las rutas se expresan con barras invertidas, por ejemplo C:\Usuarios\Publico, mientras que en PowerShell se admiten tanto barras invertidas como barras normales.

CMD es un interprete heredado que sigue funcionando en todas las versiones de Windows y que resulta util para comprobaciones rapidas y para scripts antiguos. PowerShell es la herramienta recomendada por Microsoft para la automatizacion moderna, ya que ofrece tipado opcional, manejo de errores estructurado, modulos reutilizables, sesiones remotas y acceso integrado al registro, al proveedor de certificados y a WMI y CIM. En la practica, un administrador de Windows debe dominar los dos entornos y saber elegir el mas adecuado para cada tarea.

### 1.3 Consolas y terminales

El binario cmd.exe abre la consola tradicional de Windows, historicamente alojada en el proceso conhost.exe. El binario powershell.exe corresponde a Windows PowerShell 5.1, la version que se instala junto con Windows 10 y Windows 11. El binario pwsh.exe corresponde a PowerShell 7 o superior, la version multiplataforma y de codigo abierto que Microsoft distribuye de forma independiente; se instala en paralelo y no sustituye a Windows PowerShell 5.1, que permanece instalada por compatibilidad.

Windows Terminal es la aplicacion de consola moderna de Microsoft y es el terminal predeterminado en Windows 11. Windows Terminal puede alojar varias pestanas con diferentes perfiles: CMD, Windows PowerShell, PowerShell 7, Sistemas de Subsistemas de Windows para Linux o Azure Cloud Shell, y permite configurar fuentes, colores, atajos de teclado y el comando de inicio de cada perfil. Desde la linea de comandos se puede abrir una nueva pestana con el comando wt, por ejemplo wt -p "Windows PowerShell" o wt new-tab -d C:\Trabajo.

El Windows PowerShell ISE es el entorno integrado clasico de Windows PowerShell 5.1; Microsoft lo mantiene solo para compatibilidad y recomienda utilizar Visual Studio Code con la extension de PowerShell, que ofrece autocompletado, depuracion con puntos de interrupcion, ejecucion seleccionada de codigo y integracion con el analizador de estilo PSScriptAnalyzer.

### 1.4 Privilegios, elevacion y control de cuentas

Muchas operaciones de administracion requieren privilegios de administrador. Windows controla el privilegio mediante el Control de Cuentas de Usuario (UAC): incluso un usuario con cuenta de administrador trabaja con un token de usuario elevado solo cuando lo solicita. Para abrir una consola elevada se pulsa con el boton derecho sobre la aplicacion y se elige Ejecutar como administrador, o bien se usa el acceso directo Win+X y se selecciona Terminal (Administrador) en Windows 11.

En CMD puede comprobarse si la sesion esta elevada ejecutando net session: si el comando termina sin error, la consola tiene privilegios de administrador. En PowerShell puede hacerse lo mismo con la siguiente comprobacion, que evalua si el usuario actual pertenece al rol de administrador de Windows.

```powershell
[bool](([Security.Principal.WindowsPrincipal] `
  [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole( `
  [Security.Principal.WindowsBuiltInRole]::Administrator))
```

La practica recomendada es trabajar con privilegios normales para el uso diario y elevar la sesion unicamente para las tareas que lo requieren, como instalar software, modificar el registro en claves protegidas, gestionar servicios, formatear discos o leer ficheros de otras cuentas. Ejecutar siempre la consola como administrador reduce la proteccion frente a errores y frente a scripts maliciosos.

### 1.5 Codigos de pagina y codificacion de texto

La consola de Windows utiliza un codigo de pagina para representar los caracteres. En sistemas en espanol, el codigo de pagina predeterminado suele ser 850 o 1252, por lo que las letras con tilde se muestran correctamente, pero los caracteres que no pertenecen a la tabla de codigo pueden aparecer como signos de interrogacion. El comando chcp muestra el codigo de pagina activo y chcp 65001 lo cambia a UTF-8.

```cmd
chcp
chcp 65001
```

PowerShell 7 utiliza UTF-8 como codificacion predeterminada tanto para la consola como para la escritura de ficheros. Windows PowerShell 5.1, en cambio, hereda la codificacion del sistema y al escribir ficheros con Out-File -Encoding utf8 genera un fichero UTF-8 con marca de orden de bytes (BOM). Esta diferencia es importante cuando se intercambian ficheros entre versiones o cuando se procesan con otras herramientas.

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Get-Content -Path .\informe.txt -Encoding UTF8
Out-File -FilePath .\salida.txt -Encoding utf8
```

Cuando se redirige la salida de cmd.exe a un fichero, el texto se guarda con la codificacion de la consola, no con UTF-8, salvo que se cambie el codigo de pagina previamente. Para scripts por lotes que trabajan con acentos conviene guardar el fichero sin BOM y declarar chcp 65001 en la primera linea, o bien evitar los caracteres especiales en la salida.

## Capitulo 2: CMD conceptos fundamentales

### 2.1 Estructura de un comando

Un comando de cmd.exe tiene la forma general COMANDO parametros argumentos. Los parametros se escriben con barra diagonal, por ejemplo /s o /q, y se distinguen entre mayusculas y minusculas segun el comando. Muchos comandos aceptan rutas completas o rutas relativas al directorio actual, y las rutas que contienen espacios deben escribirse entre comillas dobles.

```cmd
dir "C:\Archivos de programa" /s /b
copy "C:\Mi documento.txt" "C:\Copia\documento.txt"
```

El interprete tambien admite cadenas entre comillas dobles para conservar los espacios, comodines como asterisco y signo de interrogacion para seleccionar ficheros, y caracteres de escape como el signo de intercalacion para incluir caracteres especiales en la salida.

### 2.2 Comandos internos y externos

Los comandos internos son funciones incluidas dentro del propio cmd.exe y solo funcionan en su consola: dir, cd, copy, del, type, echo, set, if, for, goto, call, pause, cls, ren, md, rd, tree, find, findstr y muchos otros. Los comandos externos son ficheros ejecutables independientes que residen en el disco, normalmente en C:\Windows\System32, y que el interprete localiza a traves de la variable de entorno PATH.

Existe una diferencia de seguridad relevante entre CMD y PowerShell a la hora de localizar programas. cmd.exe busca primero en el directorio actual y despues en el PATH, por lo que un fichero malicioso situado en la carpeta de trabajo puede ejecutarse al escribir su nombre. PowerShell solo busca en el PATH y en los alias registrados; para ejecutar algo situado en el directorio actual hay que indicarlo de forma explicita con .\ antes del nombre. Esta es una de las razones por las que PowerShell evita la ejecucion accidental desde directorios no confiables.

### 2.3 Variables de entorno

Las variables de entorno almacenan valores que todos los programas pueden leer. En CMD el valor de una variable se obtiene escribiendo su nombre entre signos de porcentaje. Windows define numerosas variables predeterminadas como %WINDIR%, %USERPROFILE%, %TEMP%, %PROGRAMFILES%, %SYSTEMROOT% o %COMPUTERNAME%.

```cmd
echo %USERPROFILE%
echo %WINDIR%
echo %TEMP%
set PATH
```

El comando set crea o modifica variables solo para la sesion actual, mientras que setx crea o modifica variables de forma permanente en el registro del usuario o del equipo. setx guarda el valor en el registro y las nuevas variables estan disponibles para los programas que se inicien despues; no modifica el entorno de la sesion en curso. setx admite los ambitos /M para equipo y /P para usuarios, y el valor no puede superar los 1024 caracteres.

```cmd
set MIPROYECTO=C:\Trabajo\Proyecto
setx MIPROYECTO C:\Trabajo\Proyecto
setx MIPROYECTO C:\Trabajo\Proyecto /M
```

CMD tambien dispone de variables dinamicas que se calculan en cada expansion: %CD% devuelve el directorio actual, %DATE% y %TIME% la fecha y la hora segun la configuracion regional, %RANDOM% un numero aleatorio entre 0 y 32767, %ERRORLEVEL% el codigo de salida del ultimo comando y %CMDCMDLINE% la linea completa con la que se inicio el interprete. El formato de %DATE% y %TIME% depende del idioma del sistema, por lo que no debe analizarse con posiciones fijas si el script debe funcionar en equipos con configuraciones regionales distintas.

### 2.4 Redirecciones y tuberias

Las redirecciones permiten mover la entrada o la salida de un comando hacia un fichero. El operador mayor que sobrescribe el fichero de destino, el operador mayor que mayor que anade una linea al final, y el operador menor que toma la entrada desde un fichero. La salida de error se captura con el flujo 2 y la salida normal con el flujo 1.

```cmd
dir > listado.txt
dir >> historial.txt
type C:\config.ini < config_base.ini
programa.exe > salida.txt 2> errores.txt
programa.exe > todo.txt 2>&1
dir > nul
```

La redireccion a nul descarta la salida, una tecnica habitual para silenciar la confirmacion de comandos. La forma 2>&1 fusiona el flujo de errores con la salida normal, de modo que ambos se guarden en el mismo fichero.

Las tuberias encadenan comandos, de forma que la salida de uno se convierte en la entrada del siguiente. Solo se transporta texto, y la mayoria de comandos de analisis como find, findstr, sort o more estan disenados para recibir esa entrada.

```cmd
dir /b *.log | findstr /i "error"
type aplicacion.log | find /c "ERROR"
netstat -ano | findstr LISTENING
tasklist | sort > procesos_ordenados.txt
dir /s /b *.txt | more
```

### 2.5 Operadores de encadenamiento

El operador && ejecuta el segundo comando solo si el primero termina correctamente, y el operador || ejecuta el segundo comando solo si el primero falla. Periten construir flujos condicionales en una sola linea sin recurrir a bloques if.

```cmd
cd C:\Proyecto && build.bat || echo Error en la compilacion
if exist backup.zip del backup.zip && echo Antiguo backup eliminado
ping -n 1 8.8.8.8 > nul && echo Red disponible || echo Sin conexion
```

### 2.6 Comodines y seleccion de ficheros

El asterisco sustituye a cualquier numero de caracteres y el signo de intercalacion sustituye a un unico caracter. Los comodines se usan en comandos como dir, del, copy, ren, type y for.

```cmd
dir *.doc
dir relatorio_?.txt
del C:\Temp\*.tmp
copy C:\Fotos\*.jpg "C:\Copia\"
ren *.txt *.bak
```

Para buscar texto dentro de los ficheros se utiliza findstr, que admite cadenas literales, expresiones regulares, busqueda recursiva y filtro por mayusculas y minusculas.

```cmd
findstr /s /i /n "contrasena" C:\Config\*.ini
findstr /r /c:"^[0-9]" registro.log
findstr /v "informativo" log.txt
```

### 2.7 Codigos de salida

Cada comando devuelve un codigo numerico de salida al terminar, accesible en CMD con %ERRORLEVEL%. El valor 0 indica exito y cualquier valor distinto de cero indica algun tipo de fallo, aunque el significado concreto depende del comando. La forma mas sencilla de comprobar el resultado es con la sentencia if errorlevel, que evalua si el codigo es mayor o igual que el numero indicado.

```cmd
programa.exe
if errorlevel 1 (
    echo El comando fallo con codigo %ERRORLEVEL%
) else (
    echo El comando termino correctamente
)

if %ERRORLEVEL% EQU 0 echo Exito
```

En PowerShell, tras ejecutar un programa externo, el codigo de salida del ultimo comando nativo se lee en la variable $LASTEXITCODE, mientras que $? indica si el ultimo cmdlet termino sin error. Distinguir ambos valores es fundamental al mezclar herramientas de CMD dentro de scripts de PowerShell.

## Capitulo 3: CMD comandos internos

### 3.1 Navegacion y consulta de ficheros

El comando dir lista el contenido de un directorio y admite opciones para buscar en subcarpetas, filtrar por atributos y ordenar el resultado. La opcion /s recorre las subcarpetas, /b muestra solo las rutas completas sin cabeceras, /a filtra por atributos, /o ordena, /h muestra ficheros ocultos y de sistema, y /q muestra el propietario.

```cmd
dir
dir /b
dir /s /b *.pdf
dir /a:h
dir /o:d
dir /ah /b
dir *.txt /b | find /c ".txt"
```

El comando cd cambia de directorio y, con la opcion /d, tambien cambia de unidad. pushd y popd gestionan una pila de directorios: pushd guarda el actual y salta a otro, y popd vuelve al anterior. pushd acepta rutas de red y automaticamente crea una unidad temporal para trabajar con ellas.

```cmd
cd C:\Windows\Temp
cd /d D:\Proyectos
pushd C:\Windows\System32
popd
pushd \\servidor\compartido
```

md crea directorios, rd los elimina, y tree muestra la estructura de carpetas en forma de arbol. rd /s /q elimina un directorio con todo su contenido sin pedir confirmacion, por lo que debe usarse con especial cuidado.

```cmd
md C:\Trabajo\Informes
rd C:\Trabajo\Borradores
rd /s /q C:\Temp\sesion
tree C:\Proyecto /f
```

type muestra el contenido de un texto, more lo pagina pantalla a pantalla, find busca cadenas exactas y findstr ofrece busqueda mas potente con expresiones regulares. sort ordena las lineas de texto recibidas por tuberia.

```cmd
type C:\Windows\win.ini
type log.txt | more
find /i "finalizada" registro.txt
sort /r listado.txt
```

attrib muestra y modifica los atributos de los ficheros: R de solo lectura, H de oculto, S de sistema y A de archivo. La opcion - quita el atributo y el signo mas lo aplica.

```cmd
attrib documento.docx
attrib +r documento.docx
attrib -h -s C:\carpeta\fichero.txt
```

### 3.2 Copia, movimiento, borrado y renombrado

copy copia ficheros y tambien puede concatenar varios ficheros en uno solo. move traslada ficheros o directorios, ren cambia el nombre sin mover el fichero de sitio y del elimina ficheros. La opcion /Y suprime la confirmacion de sobrescritura.

```cmd
copy C:\origen\informe.txt C:\destino\informe.txt
copy *.log C:\Respaldos\
copy datos1.txt + datos2.txt todo.txt
move C:\Descargas\imagen.png C:\Fotos\imagen.png
ren informe_v1.docx informe_final.docx
del /f /q C:\Temp\*.tmp
```

del admite /f para forzar el borrado de ficheros de solo lectura, /q para el modo silencioso y /s para bajar por las subcarpetas. xcopia se usa para estructuras completas de carpetas y admite opciones como /E para incluir carpetas vacias, /I para asumir que el destino es un directorio, /H para ficheros ocultos y de sistema, /Y para no preguntar y /D para copiar solo los ficheros mas recientes.

```cmd
xcopy C:\Proyecto D:\Respaldos\Proyecto /E /I /H /Y
xcopy C:\Datos D:\Copia /D /Y /I
```

En versiones recientes de Windows, robocopy es la herramienta recomendada para copias de ficheros y directorios. Admite reintentos, compresion en transito, ejecucion con varios hilos, espejo de estructuras y registro detallado de resultados.

```cmd
robocopy C:\Proyecto D:\Respaldos\Proyecto /E /Z /MT:8 /R:2 /W:5 /LOG+:C:\Log\copia.log
robocopy C:\Datos D:\Copia /MIR /XF *.tmp /XD Cache
```

### 3.3 Configuracion de la consola

echo imprime texto en pantalla y con @echo off en un script por lotes evita que cada linea se muestre antes de ejecutarse. echo. imprime una linea en blanco y redirigir echo hacia un fichero permite crear o vaciar ficheros de texto.

```cmd
@echo off
echo Preparando el informe...
echo.
echo Listo > estado.txt
type nul > fichero_vacio.txt
```

set crea variables, set /a evalua expresiones aritmeticas y set /p solicita entrada al usuario. La forma recomendada de asignar un valor es encerrar toda la asignacion entre comillas para evitar espacios accidentales al final.

```cmd
set "RUTA=C:\Trabajo"
set /a TOTAL=5+3*2
set /p USUARIO=Escribe tu nombre:
echo Hola %USUARIO%, total=%TOTAL%
echo %RUTA:Trabajo=Proyectos%
```

La ultima linea muestra la sustitucion de texto en variables: toma el valor de RUTA, busca la cadena Trabajo y la sustituye por Proyectos.

title cambia el titulo de la ventana, color cambia los colores de texto y fondo, cls borra la pantalla y prompt modifica el texto del indicador.

```cmd
title Herramienta de mantenimiento
color 0a
cls
prompt $P$G
```

path muestra o modifica la lista de directorios donde el interprete busca programas externos. La modificacion solo afecta a la sesion actual; para hacerla permanente se usa setx PATH, teniendo en cuenta el limite de 1024 caracteres de setx.

```cmd
path
path %PATH%;C:\Herramientas
setx PATH "%PATH%;C:\Herramientas"
```

### 3.4 Control de flujo: if, for, goto y call

La sentencia if evalua condiciones de existencia de ficheros, comparaciones de cadenas y valores de ERRORLEVEL. Se escriben las variantes if exist, if not exist, if defined, if errorlevel y la comparacion con dos signos de igualdad, que puede acompanarse de la opcion /i para ignorar mayusculas y minusculas.

```cmd
if exist C:\config.ini echo El fichero existe
if not exist C:\Salida md C:\Salida
if defined MIPROYECTO echo La variable esta definida
if /i "%USERINPUT%"=="si" goto continuar
if "%OS%"=="Windows_NT" echo Sistema moderno
```

Los bloques if y else abren y cierran con parentesis, y la palabra else debe quedar en la misma linea que el parentesis de cierre.

```cmd
if exist "C:\Proyecto\build.bat" (
    call C:\Proyecto\build.bat
) else (
    echo No se encontro el script de compilacion
)
```

El bucle for itera sobre listas de ficheros, directorios, rangos numericos o lineas de un fichero. En un fichero por lotes las variables de bucle se escriben con dos signos de porcentaje, mientras que en la linea de comandos interactiva se escribe un unico signo.

```bat
for %%f in (*.txt) do echo %%f
for %%d in (C:\Datos D:\Datos E:\Datos) do dir /b "%%d"
for /l %%i in (1,1,10) do echo Numero %%i
for /d %%c in (C:\Proyecto\*) do echo Carpeta: %%c
for /r C:\Logs %%f in (*.log) do echo %%f
```

La variante for /f es la mas potente: procesa la salida de un comando o las lineas de un fichero y divide cada linea en tokens. El parametro tokens indica cuantos campos se capturan, delims define los separadores, skip omite lineas iniciales y usebackq permite usar comillas para rutas con espacios.

```bat
for /f "tokens=1,2 delims=," %%a in (datos.csv) do (
    echo Campo1=%%a Campo2=%%b
)

for /f "usebackq delims=" %%a in ("C:\Mi carpeta\lista.txt") do (
    echo Linea: %%a
)

for /f "tokens=*" %%s in ('tasklist ^| find /i "notepad"') do echo %%s
```

Los modificadores de variable permiten manipular la ruta capturada: %%~fI devuelve la ruta completa, %%~dI la unidad, %%~pI el directorio, %%~nI el nombre, %%~xI la extension, %%~aI los atributos, %%~tI la fecha y %%~zI el tamano. La combinacion %%~dpI es especialmente util para obtener el directorio del script.

goto salta a una etiqueta del script y, junto con call :etiqueta, permite construir subrutinas. exit /b termina el script o la subrutina y devuelve el control al punto de llamada, mientras que exit cierra la consola completa.

```bat
goto :fin
:procesar
echo Procesando %1
exit /b 0
:fin
exit /b
```

start lanza un programa en una ventana nueva y acepta la ruta del ejecutable, los argumentos y opciones como /wait para esperar a que termine, /min para minimizar o /b para abrirlo sin ventana adicional. Las comillas vacias iniciales son necesarias cuando la ruta del programa contiene espacios.

```cmd
start "" notepad.exe
start "" /wait "C:\Programa\instalador.exe" /S
start "" /min cmd /c "C:\Scripts\proceso.bat"
start https://learn.microsoft.com
```

pause detiene la ejecucion hasta pulsar una tecla, timeout espera un numero determinado de segundos y choice presenta una seleccion fija de opciones y captura la tecla pulsada en una variable.

```cmd
pause
timeout /t 10 /nobreak
choice /c SN /m "Deseas continuar"
if %errorlevel%==1 echo Aceptado
if %errorlevel%==2 echo Cancelado
```

shift desplaza los parametros posicionales %1, %2 y demas, y %* representa todos los argumentos recibidos. call permite invocar otro fichero por lotes y devolver el control al script que lo llamo, ademas de llamar a subrutinas internas con call :etiqueta.

```bat
call otro_script.bat parametro1
call :mostrar Mensaje de prueba
exit /b

:mostrar
echo %~1
exit /b
```

### 3.5 Otras utilidades internas

ver muestra la version de Windows, vol muestra la etiqueta y el numero de serie de una unidad, date y time consultan o cambian la fecha y la hora del sistema, y cls limpia la ventana. find localiza cadenas exactas en ficheros o en la entrada estandar.

```cmd
ver
vol C:
find /i "error" C:\log\aplicacion.log
```

assoc muestra o modifica las asociaciones de extension de fichero y ftype muestra o modifica el comando que abre un tipo de fichero. En Windows moderno, la asociacion elegida por el usuario tiene prioridad, por lo que estos comandos se usan principalmente para consultas.

```cmd
assoc .txt
ftype txtfile
```

doskey recupera el historial de comandos con F7 o /history y permite definir macros de teclado, que se expanden al escribir su nombre. La macro admite los marcadores $1 a $9 para parametros y $* para todos ellos.

```cmd
doskey hist=doskey /history
doskey ip=ipconfig /all
doskey ll=dir /o:d $*
doskey /history
```

clip copia la salida de un comando al portapapeles de Windows, lo que permite pegar resultados en otros programas sin guardar ficheros intermedios.

```cmd
dir /b > listado.txt
type listado.txt | clip
systeminfo | clip
```

## Capitulo 4: CMD comandos externos utiles

### 4.1 Red y conectividad

ipconfig muestra la configuracion de red y sus opciones mas habituales son /all para el detalle completo, /release y /renew para renovar la direccion DHCP, /flushdns para vaciar la cache de resolucion de nombres y /displaydns para ver las entradas en cache.

```cmd
ipconfig
ipconfig /all
ipconfig /flushdns
ipconfig /displaydns
```

ping comprueba la conectividad con un host y admite el numero de paquetes con -n, el tamano con -l y el tiempo de espera con -w. tracert muestra el recorrido de los paquetes y pathping combina tracert con una medicion de perdidas en cada salto.

```cmd
ping -n 4 8.8.8.8
ping -l 1024 -n 3 www.microsoft.com
tracert -d www.microsoft.com
pathping -n www.microsoft.com
```

nslookup consulta servidores DNS y netstat muestra las conexiones de red y los puertos en escucha. La opcion -ano de netstat permite ver ademas el identificador de proceso asociado a cada conexion.

```cmd
nslookup www.microsoft.com
nslookup -type=mx microsoft.com
netstat -ano
netstat -ano | findstr LISTENING
netstat -abn
```

getmac muestra las direcciones MAC, arp muestra la tabla de resolucion de direcciones IP a MAC de la red local, hostname devuelve el nombre del equipo y systeminfo resume el equipo completo.

```cmd
getmac /v
arp -a
hostname
systeminfo | findstr /I "Host Name OS Name Total physical"
```

whoami muestra la cuenta actual y con /all detalla grupos, identificadores de seguridad y privilegios. es una comprobacion rapida para saber con que identidad se esta trabajando antes de ejecutar operaciones sensibles.

```cmd
whoami
whoami /groups | findstr /i "S-1-16-12288"
whoami /priv
```

### 4.2 Procesos y servicios

tasklist lista los procesos en ejecucion y admite filtros como /fi, formato de salida con /fo y muestra de los servicios asociados con /svc. taskkill termina procesos por nombre, identificador o ventana y requiere la opcion /f para forzar la finalizacion.

```cmd
tasklist
tasklist /svc
tasklist /fi "imagename eq notepad.exe"
taskkill /im notepad.exe /f
taskkill /pid 1234 /f
taskkill /fi "status eq running" /im "test.exe" /f
```

sc gestiona los servicios de Windows. sc query muestra el estado, sc qc muestra la configuracion, sc start y sc stop controlan el servicio y sc config modifica sus propiedades. En la sintaxis de sc, tras un parametro con igualdad debe escribirse un espacio antes del valor.

```cmd
sc query spooler
sc qc spooler
sc stop spooler
sc config spooler start= auto
sc config spooler binPath= "C:\Ruta\servicio.exe"
```

net es la herramienta clasica de gestion de red y cuentas. net start lista los servicios en ejecucion, net user gestiona cuentas locales, net localgroup gestiona grupos, net share gestiona recursos compartidos y net use conecta o desconecta unidades de red.

```cmd
net start
net start spooler
net stop spooler
net user
net user invitado /active:no
net localgroup Administradores
net use Z: \\servidor\compartido /user:dominio\usuario
net use Z: /delete
```

wmic es la herramienta heredada de gestion basada en WMI. Microsoft la considera en desuso y recomienda sustituirla por los cmdlets de CIM de PowerShell, como Get-CimInstance. Aun asi, sigue presente en muchas instalaciones y se encuentra frecuentemente en scripts antiguos.

```cmd
wmic os get caption /value
wmic logicaldisk get caption,freespace,size
wmic process where name="notepad.exe" get processid
```

### 4.3 Discos y almacenamiento

diskpart es una herramienta interactiva de gestion de discos que trabaja con discos, particiones, volumenes y letras. Modifica la estructura de almacenamiento del equipo, de modo que cualquier operacion de escritura puede destruir datos si se selecciona el disco equivocado; por ello conviene ejecutar primero list disk y list volume para identificar correctamente los objetos antes de continuar.

```cmd
diskpart
list disk
select disk 1
list partition
select partition 2
assign letter=E
exit
```

chkdsk comprueba la integridad del sistema de ficheros y, con /f, corrige errores logicos y con /r localiza sectores defectuosos y recupera informacion. Si el volumen esta en uso, Windows propone comprobarlo en el siguiente reinicio. fsutil ofrece informacion avanzada de volumenes.

```cmd
chkdsk C:
chkdsk D: /f /r
fsutil volume diskfree C:
fsutil fsinfo ntfsinfo C:
```

El Servicio de Limpieza de Disco se puede lanzar con cleanmgr y el optimizador de unidades con defrag. En Windows 10 y 11 existe ademas la tarea de mantenimiento automatico que ejecuta estas labores de forma programada.

```cmd
cleanmgr /sagerun:1
defrag C: /O
```

### 4.4 Mantenimiento y reparacion del sistema

sfc verifica la integridad de los ficheros protegidos del sistema y los repara si encuenta alteraciones. La opcion /scannow analiza todo el sistema y puede requerir tiempo y privilegios de administrador.

```cmd
sfc /scannow
sfc /verifyfile=C:\Windows\System32\kernel32.dll
```

DISM administra las imagenes de Windows y permite reparar la tienda de componentes de la que sfc obtiene los ficheros correctos. El flujo habitual ante un sistema inestable es ejecutar primero la comprobacion de salud, despues el analisis y, si hay correcciones pendientes, la restauracion.

```cmd
DISM /Online /Cleanup-Image /CheckHealth
DISM /Online /Cleanup-Image /ScanHealth
DISM /Online /Cleanup-Image /RestoreHealth
```

powercfg gestiona la configuracion de energia y puede generar informes detallados de bateria y de sonido. gpupdate fuerza la actualizacion de directivas de grupo y gpresult genera el informe de directivas aplicadas.

```cmd
powercfg /batteryreport /output C:\informe\bateria.html
powercfg /getactivescheme
gpupdate /force
gpresult /r
gpresult /h C:\informe\directivas.html
```

El Visor de sucesos grafico se complementa con wevtutil, que consulta y exporta eventos desde la linea de comandos.

```cmd
wevtutil qe System /c:10 /rd:true /f:text
wevtutil epl Application C:\informe\application.evtx
```

### 4.5 Usuarios, permisos y seguridad

net user crea y gestiona cuentas locales y net localgroup las agrupa. El nombre concreto de los grupos y las cuentas especiales puede variar segun el idioma de Windows, por lo que en sistemas en espanol conviene comprobar los nombres con net localgroup antes de automatizar operaciones sobre ellos.

```cmd
net user tecnico P@ssw0rd /add
net user tecnico /active:yes
net localgroup Administradores tecnico /add
net user tecnico /delete
```

icacls muestra y modifica las listas de control de acceso. Los permisos mas comunes son F para control total, M para modificar, R para lectura y W para escritura, y se aplican con /grant para conceder, /deny para denegar y /remove para retirar. takeownership cambia el propietario de un fichero o carpeta, lo que suele ser necesario antes de modificar permisos heredados de otros usuarios.

```cmd
icacls C:\Datos
icacls C:\Datos /grant Technico:(OI)(CI)F /T
icacls C:\Datos /grant "DOMAIN\Users":R
icacls C:\Datos /save permisos.txt
takeown /f C:\carpeta /r /d y
```

cipher gestiona el cifrado de ficheros del sistema de archivos EFS y tambien puede borrar de forma segura el espacio libre de una unidad. runas ejecuta un comando con las credenciales de otro usuario.

```cmd
cipher /e C:\Datos
cipher /d C:\Datos
cipher /w:C
runas /user:DOMAIN\admin cmd
```

La herramienta certutil calcula codigos hash, codifica y decodifica ficheros y gestiona certificados. Es la alternativa nativa de CMD para verificar la integridad de descargas o instaladores.

```cmd
certutil -hashfile instalador.exe SHA256
certutil -hashfile instalador.exe SHA512
certutil -encode entrada.txt salida.txt
certutil -decode salida.txt recuperado.txt
```

### 4.6 Tareas programadas

schtasks crea, consulta, ejecuta y elimina tareas programadas. Las opciones mas utilizadas son /create para crear, /query para listar, /run para ejecutar inmediatamente, /end para terminar una tarea en curso, /delete para eliminar y /change para modificar.

```cmd
schtasks /query /fo LIST /v
schtasks /create /tn "Respaldo diario" /tr "C:\Scripts\respaldo.bat" /sc daily /st 08:00 /ru administrator
schtasks /create /tn "Limpieza" /tr "cleanmgr /sagerun:1" /sc weekly /d SUN /st 09:00
schtasks /run /tn "Respaldo diario"
schtasks /end /tn "Respaldo diario"
schtasks /delete /tn "Respaldo diario" /f
```

Cuando se ejecuta en equipo con Windows Server o en un dominio, conviene indicar la cuenta de ejecucion con /ru y la contrasena con /rp. En Windows 10 y 11, el Programador de tareas grafico muestra las mismas tareas y permite revisar su historial de ejecucion.

### 4.7 Registro de Windows

reg consulta y modifica el registro de Windows. Las claves principales son HKEY_LOCAL_MACHINE abreviada HKLM para el equipo, HKEY_CURRENT_USER HKCU para el usuario, HKEY_CLASSES_ROOT HKCR para las asociaciones de ficheros y HKEY_USERS HKEYU para todos los perfiles.

```cmd
reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
reg query HKCU\Environment /v PATH
reg add "HKLM\SOFTWARE\MiEmpresa\Ajustes" /v Version /t REG_SZ /d "1.0" /f
reg add "HKCU\Environment" /v MIPROYECTO /t REG_SZ /d "C:\Trabajo" /f
reg delete "HKLM\SOFTWARE\MiEmpresa\Ajustes" /v Version /f
reg export HKLM\SOFTWARE\MiEmpresa ajustes.reg
reg import ajustes.reg
```

La opcion /f suprime la confirmacion. Los tipos de datos mas frecuentes son REG_SZ para cadenas, REG_DWORD para valores numericos de 32 bits y REG_EXPAND_SZ para cadenas con variables de entorno como %SystemRoot%. Antes de modificar el registro conviene exportar la clave afectada para poder deshacer el cambio.

### 4.8 Utilidades adicionales

donde localiza un comando en el PATH y fc compara dos ficheros, con /b para modo binario, /c para ignorar mayusculas y /N para mostrar numerode linea. comp realiza una comparacion simple byte a byte.

```cmd
where.exe powershell
where /r C:\Scripts *.bat
fc /n anterior.txt actual.txt
fc /b copia1.zip copia2.zip
```

shutdown apaga, reinicia o cierra sesion del equipo y admite un temporizador y un comentario que se muestra a los usuarios. shutdown /a cancela un apagado programado.

```cmd
shutdown /s /t 60 /c "El equipo se apagara en un minuto"
shutdown /r /t 0
shutdown /l
shutdown /h
shutdown /a
```

timeout y choice ya se describieron en el capitulo anterior; donde se destaca ahora es en la diferencia con type: donde localiza ficheros ejecutables, no contiene texto.

wmic, cacls y otros comandos clasicos estan marcados como obsoletos en la documentacion de Microsoft. Las alternativas actuales son los cmdlets de PowerShell y icacls para permisos, y Get-CimInstance para consultas de hardware y sistema.

Los comandos netsh y netsh interface son utilidades de configuracion de red por texto. netsh permite configurar direcciones estaticas, perfiles de red, cortafuegos de Windows y protocolos sin necesidad de abrir el panel de configuracion.

```cmd
netsh interface ip show config
netsh advfirewall show currentprofile
netsh wlan show profiles
netsh wlan show profile name="MiRed" key=clear
```

## Capitulo 5: Archivos por lotes

### 5.1 Estructura de un script por lotes

Un fichero por lotes es un fichero de texto con extension .bat o .cmd que contiene una secuencia de comandos que cmd.exe ejecuta en orden. La extension .cmd se usa normalmente para scripts escritos especificamente para cmd.exe, mientras que .bat se reserva por compatibilidad con MS-DOS, pero el interprete los trata practicamente igual. La extension recomendada por Microsoft para los scripts modernos de Windows es .ps1 de PowerShell.

La primera linea habitual es @echo off, que impide que cada comando se muestre antes de ejecutarse y deja la salida limpia con solo los mensajes del script. Los comentarios se escriben con rem o con dos puntos, aunque la forma con dos puntos no debe usarse dentro de bloques parentizados porque el interprete la interpreta como etiqueta.

```bat
@echo off
rem Este script genera un informe diario
:: Los comentarios ayudan a documentar el script
title Informe diario
echo Generando informe...
dir /s /b C:\Datos > C:\Informes\listado_%DATE:/=-%.txt
echo Informe creado correctamente
pause
```

Los scripts por lotes deben guardarse con codificacion compatible con la consola. Si se guardan como UTF-8 con marca de orden de bytes, el interprete puede fallar en la primera linea; lo mas seguro es guardarlos sin BOM y usar chcp 65001 si la salida contiene acentos, o evitar los caracteres especiales en las cadenas.

### 5.2 Parametros y expansiones especiales

Un script recibe parametros posicionales que se referencian como %1, %2 y asi sucesivamente; %0 es el nombre del propio script y %* contiene todos los parametros. Los modificadores de variable permiten transformar cada parametro: %~1 elimina las comillas, %~f1 convierte en ruta completa, %~dp1 aporta unidad y directorio, %~n1 el nombre y %~x1 la extension.

```bat
@echo off
rem Uso: respaldo.bat origen destino
if "%~2"=="" (
    echo Uso: %~nx0 origen destino
    exit /b 1
)
set "ORIGEN=%~1"
set "DESTINO=%~2"
echo Copiando de %ORIGEN% a %DESTINO%
robocopy "%ORIGEN%" "%DESTINO%" /E /R:2 /W:5
exit /b %ERRORLEVEL%
```

El directorio del script, expresado como %~dp0, es de gran utilidad porque permite construir rutas relativas al propio script sin importar desde que directorio se lance. Esta es la forma correcta de localizar ficheros auxiliares como plantillas, listas o ficheros de configuracion.

```bat
@echo off
set "BASE=%~dp0"
if exist "%BASE%config.ini" (
    type "%BASE%config.ini"
) else (
    echo No se encontro config.ini junto al script
)
```

### 5.3 Variables diferidas

Por defecto, cmd.exe expande todas las variables al analizar un bloque completo de parentesis, de modo que el valor de una variable modificada dentro de un bucle if o for se lee una sola vez al entrar en el bloque. Para leer el valor en cada iteracion hay que activar la expansion diferida con setlocal EnableDelayedExpansion y usar signos de exclamacion en lugar de signos de porcentaje.

```bat
@echo off
setlocal EnableDelayedExpansion
set CONTADOR=0
for %%f in (*.txt) do (
    set /a CONTADOR+=1
    echo !CONTADOR! - %%f
)
echo Total de ficheros: !CONTADOR!
endlocal
```

Sin la expansion diferida, el ejemplo anterior mostraria siempre el mismo numero, porque %CONTADOR% se sustituiria por su valor inicial antes de empezar el bucle. La expansion diferida tiene un efecto secundario: los signos de exclamacion del contenido se interpretan como inicio de variable, por lo que en textos con muchos signos de exclamacion conviene desactivarla con setlocal DisableDelayedExpansion al procesarlos.

### 5.4 Subrutinas y estructura

Las subrutinas se construyen con call :etiqueta y se cierran con exit /b. Permiten dividir un script en partes reutilizables, como funciones de registro de errores, validacion de entradas o limpieza de ficheros temporales. El primer parametro de la subrutina llega como %1 y, para recibir varios, se usan llamadas con varios argumentos.

```bat
@echo off
setlocal
call :registrar "Inicio del proceso"
call :validar C:\Datos
call :registrar "Fin del proceso"
exit /b 0

:registrar
echo [%DATE% %TIME%] %~1 >> C:\Log\proceso.log
exit /b

:validar
if exist "%~1" (
    echo Directorio valido: %~1
    exit /b 0
) else (
    echo Directorio no encontrado: %~1
    exit /b 1
)
```

El valor devuelto por exit /b se captura en %ERRORLEVEL% justo despues de la llamada, lo que permite encadenar comprobaciones de error entre subrutinas.

### 5.5 Ejemplo completo: respaldo con registro y control de errores

El siguiente ejemplo ilustra las buenas practicas habituales de un script por lotes de produccion: comprobacion de parametros, creacion de la carpeta destino, registro con fecha, uso de robocopy, traduccion del codigo de salida y codigo de salida propio del script.

```bat
@echo off
setlocal EnableExtensions
title Respaldo incremental
set "LOG=C:\Log\respaldo.log"

if not exist "%~1" (
    echo Error: no existe la carpeta de origen %~1
    exit /b 2
)
if not exist "%~2" md "%~2"

echo [%DATE% %TIME%] Inicio de respaldo de %~1 >> "%LOG%"
robocopy "%~1" "%~2" /E /Z /MT:4 /R:2 /W:5 /NP /LOG+:"%LOG%"
set "RC=%ERRORLEVEL%"

if %RC% LSS 8 (
    echo [%DATE% %TIME%] Respaldado finalizado con codigo %RC% >> "%LOG%"
    echo Respaldado completado correctamente.
    exit /b 0
) else (
    echo [%DATE% %TIME%] Respaldado fallo con codigo %RC% >> "%LOG%"
    echo Error en el respaldado. Consulta %LOG%
    exit /b 1
)
```

robocopy devuelve codigos de salida por bandera en los que los valores de 0 a 7 indican que la operacion se completo con exito o con diferencias menores, mientras que 8 o superiores indican fallos. Este comportamiento debe tenerse en cuenta al comprobar %ERRORLEVEL% tras una copia.

### 5.6 Errores frecuentes en scripts por lotes

El error mas habitual es olvidar comillar rutas con espacios, lo que provoca que el interprete trate cada palabra como un parametro distinto. El segundo es escribir %i dentro de un bucle for de un fichero por lotes, cuando corresponde %%i; en la linea de comandos interactiva funciona con un solo signo de porcentaje, pero dentro de un .bat hace falta dos.

El tercer error es confiar en %DATE% y %TIME% para generar nombres de fichero en equipos con distintas configuraciones regionales, ya que la fecha puede contener barras, comas o el caracter de dia. Una alternativa robusta es construir la marca de fecha con variables de fecha separadas o con powershell -Command desde el propio script.

El cuarto error es asumir que un comando anterior tuvo exito sin comprobar %ERRORLEVEL%. En scripts largos conviene revisar el codigo de salida tras cada operacion critica y registrar cualquier fallo. El quinto error es olvidar setlocal, que hace que las variables creadas se filtren al entorno de la consola que invoco el script y dificulten su reutilizacion.

Por ultimo, los scripts que ejecutan programas externos pueden bloquearse si el programa espera entrada. En ese caso se redirige la entrada desde un fichero vacio o se usan opciones de silencio como /q o /y segun el comando.

## Capitulo 6: PowerShell fundamentos

### 6.1 Que es PowerShell

PowerShell es un interprete de comandos y un lenguaje de script basado en la plataforma .NET, creado por Microsoft para la administracion y la automatizacion de Windows. Trabaja con objetos de .NET en lugar de texto plano, lo que permite acceder a propiedades y metodos de los resultados sin analizar cadenas. Se distribuye en dos grandes ramas: Windows PowerShell 5.1, incluida en Windows 10 y Windows 11 sobre .NET Framework, y PowerShell 7 o superior, multiplataforma y de codigo abierto sobre .NET, que se instala de forma independiente con el binario pwsh.exe.

La version activa se consulta con la variable $PSVersionTable, que contiene la version del motor, la edicion y las versiones de los componentes asociados. Esta comprobacion es recomendable al inicio de cualquier script que deba funcionar en ambas ramas, porque existen diferencias de sintaxis y de cmdlets entre ellas.

```powershell
$PSVersionTable
$PSVersionTable.PSVersion
$PSVersionTable.PSEdition
```

Entre las diferencias mas importantes, PowerShell 7 anade los operadores de encadenamiento && y ||, el operador ternario, los operadores de fusion nula ?? y ??=, valores predeterminados UTF-8 sin BOM, cmdlets adicionales de red y de terminal, y elimina componentes exclusivos de Windows PowerShell 5.1 como el cmdlet Get-WmiObject y el ISE. Windows PowerShell 5.1 sigue siendo necesaria para scripts que dependen de modulos solo disponibles en .NET Framework o de comportamientos concretos de su motor.

### 6.2 Cmdlets y convencion de nombres

Un cmdlet es un comando pequeno y consistente escrito en .NET que sigue la convencion Verbo-Sustantivo, por ejemplo Get-Service, Set-ItemProperty o New-Item. Los verbos aprobados por Microsoft se pueden listar con Get-Verb; usar verbos aprobados hace que los scripts sean predecibles para otros usuarios y aptos para el analizador de estilo.

```powershell
Get-Verb
Get-Command Get-Service
Get-Command -Noun Service
Get-Command -Verb Set
```

Los cmdlets admiten parametros con guion, valores posicionales y interruptores o switches, que son parametros booleanos que se activan simplemente escribiendolos. Un switch puede forzarse a falso con la sintaxis -Switch:$false. Los parametros pueden escribirse con dos formas equivalentes: -Nombre valor o -Nombre:valor.

```powershell
Get-ChildItem -Path C:\Windows -Filter *.exe -Recurse
Get-ChildItem -Path C:\Windows -Filter *.exe -Recurse:$false
Get-Service -Name Spooler
Get-Service Spooler
```

Los cmdlets comparten parametros comunes disponibles en practicamente todos ellos: -Verbose para mensajes detallados, -Debug para mensajes de depuracion, -ErrorAction para decidir que hacer ante un error, -ErrorVariable para capturar los errores en una variable, -WarningAction para el nivel de avisos, -OutVariable para guardar la salida en una variable y -WhatIf o -Confirm en los cmdlets que admiten cambios, para simular o confirmar la operacion antes de aplicarla.

### 6.3 El pipeline de objetos

El operador tuberia de PowerShell encadena cmdlets de forma que la salida de uno se convierte en la entrada del siguiente como objetos, no como texto. Esto permite filtrar y ordenar informacion sin volver a analizar cadenas, y es la razon principal de la potencia de PowerShell.

```powershell
Get-Service | Where-Object Status -eq 'Running' | Sort-Object DisplayName | Select-Object -First 5 Name, Status
Get-Process | Sort-Object WS -Descending | Select-Object -First 10
Get-ChildItem C:\Windows -Filter *.log -Recurse | Measure-Object
```

Cada objeto del pipeline expone propiedades con el operador punto y metodos que se invocan con parentesis. Get-Member muestra las propiedades y metodos disponibles de cualquier objeto, y es la herramienta de descubrimiento mas importante cuando se trabaja con un tipo de objeto desconocido.

```powershell
Get-Date | Get-Member
Get-Service | Get-Member -MemberType Properties
(Get-Date).AddDays(-7)
'cadena'.ToUpper()
```

El elemento automatico $_ representa el objeto actual dentro de bloques como Where-Object o ForEach-Object, y $PSItem es su sinonimo largo. Este elemento es la base de la mayoria de los filtros y transformaciones en el pipeline.

```powershell
Get-ChildItem *.txt | Where-Object { $_.Length -gt 1MB }
Get-Process | ForEach-Object { $_.Name.ToUpper() }
```

### 6.4 Providers y sistema de rutas

PowerShell expone varios recursos del sistema como si fueran unidades de disco, a traves de proveedores. El proveedor FileSystem da acceso a ficheros y carpetas, Registry al registro de Windows, Env a las variables de entorno, Cert a los certificados, Function, Variable y Alias a los elementos internos del propio entorno y Temp a una carpeta temporal de la sesion.

```powershell
Get-PSProvider
Set-Location HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion
Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion -Name ProgramFilesDir
Get-ChildItem Env: | Sort-Object Name
Set-Location C:\Windows
```

Las rutas de registro se expresan como HKLM:\ para HKEY_LOCAL_MACHINE y HKCU:\ para HKEY_CURRENT_USER, y se comportan como cualquier otra ruta del proveedor de ficheros, de modo que funcionan los mismos cmdlets Get-ChildItem, Get-Content, New-Item o Remove-Item. Este diseno uniforme es una de las caracteristicas mas apreciadas de PowerShell.

### 6.5 Sistema de ayuda

PowerShell integra un sistema de ayuda completo. Get-Help muestra la sintaxis y la descripcion de cualquier cmdlet y admite las opciones -Examples, -Detailed, -Full y -Online, que abre la pagina de Microsoft Learn correspondiente. Update-Help descarga la ayuda mas reciente de los modulos instalados y Get-Help about_* lista los temas de contexto del lenguaje.

```powershell
Get-Help Get-ChildItem
Get-Help Get-ChildItem -Examples
Get-Help about_operators
Get-Help about_automatic_variables -Full
Update-Help
```

Los temas about_ explican conceptos completos como operadores, variables automaticas, ambitos, politicas de ejecucion, analisis del interprete, comodines, expansion de cadenas, espacios de nombres de modulos y remoting. Son la referencia oficial dentro del propio entorno y complementan la documentacion de Microsoft Learn.

### 6.6 Comandos, aliases y descubrimiento

Get-Command lista todos los cmdlets, funciones, aliases y programas externos disponibles, y permite buscar por patron con -Name o por verbo y sustantivo. Get-Alias muestra los nombres cortos registrados y permite averiguar a que cmdlet apunta cada alias.

```powershell
Get-Command
Get-Command *Service*
Get-Alias
Get-Alias gci
Get-Alias -Definition Get-ChildItem
```

Entre los aliases mas comunes estan dir, ls y gci para Get-ChildItem, cd y sl para Set-Location, cls para Clear-Host, cp y copy para Copy-Item, mv y move para Move-Item, rm, del y rmdir para Remove-Item, cat, gc y type para Get-Content, ren para Rename-Item, ps para Get-Process, kill para Stop-Process, sleep para Start-Sleep, select para Select-Object, ft para Format-Table, fl para Format-List, gm para Get-Member, gcm para Get-Command, sls para Select-String, sort para Sort-Object, group para Group-Object, measure para Measure-Object, echo para Write-Output e iex para Invoke-Expression.

Existen conflictos de nombres con programas externos que conviene conocer. En Windows PowerShell 5.1, curl y wget son aliases de Invoke-WebRequest, por lo que curl no invoca el programa curl.exe; para usarlo hay que escribir curl.exe. Del mismo modo, where es un alias de Where-Object, de modo que para localizar ficheros ejecutables debe escribirse where.exe. PowerShell 7 elimino estos aliases conflictivos.

```powershell
# En Windows PowerShell 5.1
curl.exe https://api.example.com/datos
where.exe powershell
# En PowerShell 7
curl https://api.example.com/datos
where powershell
```

## Capitulo 7: PowerShell sintaxis

### 7.1 Variables

Las variables en PowerShell comienzan con el signo de dolar y pueden contener cualquier tipo de objeto de .NET, no solo cadenas o numeros. No es necesario declarar su tipo, aunque puede hacerse con una anotacion de tipo para mayor seguridad. Los nombres distinguen entre mayusculas y minusculas en su asignacion pero la busqueda de propiedades y parametros no lo hace.

```powershell
$nombre = 'Eduardo'
$edad = 34
[double]$precio = 19.99
$fecha = Get-Date
Write-Output "Hola $nombre, tienes $edad anios"
Write-Output ("Precio: {0:C2}" -f $precio)
```

Las cadenas entre comillas dobles interpretan las variables y admiten subexpresiones con el patron $( ), mientras que las cadenas entre comillas simples son literales y no interpretan nada. El caracter de escape es la comilla invertida y permite escribir caracteres literales que de otro modo tendrian significado, como el signo de dolar o la comilla doble.

```powershell
$servidor = 'SERV01'
"Servidor: $servidor"
"Fecha actual: $(Get-Date -Format yyyy-MM-dd)"
"Nombre de usuario: $env:USERNAME"
'Literal: $servidor no se interpreta'
"Escapado: costo total `$50"
$multilinea = @"
Primera linea con $servidor
Segunda linea
"@
```

Las variables con ambito se escriben con un prefijo seguido de dos puntos: $global: para el ambito global, $script: para el ambito del script, $local: para el ambito actual, $env: para las variables de entorno, $function: para las funciones, $using: dentro de bloques remotos y $private: para variables visibles solo en el bloque actual.

```powershell
$env:SystemRoot
$env:PATH -split ';'
$global:contador = 0
Set-Item -Path Env:MIPROYECTO -Value 'C:\Trabajo'
$env:MIPROYECTO
```

Entre las variables automaticas mas importantes estan $_ y $PSItem para el objeto actual, $args para los argumentos no declarados, $error para el historial de errores, $true, $false y $null para los valores logicos y nulos, $Host para el host de consola, $PWD para el directorio actual, $HOME para el perfil del usuario, $PSCommandPath y $PSScriptRoot para la ruta del script en ejecucion, $LASTEXITCODE para el codigo de salida del ultimo programa externo, $? para el resultado del ultimo cmdlet y $PSVersionTable para la version en curso.

### 7.2 Operadores

Los operadores aritmeticos son los habituales: mas, menos, asterisco para el producto, barra para la division y el simbolo de porcentaje para el resto. La division de dos enteros devuelve un decimal, de modo que para la division entera se usa el operador de division entera desplazando con [math]::Floor o con el operador -shr y -shl cuando corresponde, o simplemente con operaciones de conversion a entero.

```powershell
$suma = 5 + 3
$division = 10 / 4
$resta = 10 - 3
$potencia = [math]::Pow(2, 10)
$redondeo = [math]::Round(3.14159, 2)
$entero = [int](10 / 4)
```

Los operadores de comparacion son -eq, -ne, -gt, -ge, -lt y -le, equivalentes a igual, distinto, mayor que, mayor o igual, menor que y menor o igual. Por defecto las comparaciones de cadenas ignoran mayusculas y minusculas; para respetarlas se usa la variante -ceq, -cne y demas. Los operadores logicos son -and, -or, -not y su alias !.

```powershell
if ($estado -eq 'Activo') { 'El elemento esta activo' }
if ($edad -ge 18) { 'Mayor de edad' }
if ($nombre -ceq 'Eduardo') { 'Coincide respetando mayusculas' }
if ($activo -and $verificado) { 'Acceso permitido' }
```

Los operadores de patron evaluan cadenas. -like y -notlike comparan con comodines asterisco, signo de interrogacion y corchetes; -match y -notmatch aplican expresiones regulares y ademas rellenan la variable automatica $Matches; -replace sustituye patrones regex y su variante -creplace respeta mayusculas.

```powershell
if ('informe_2024.pdf' -like 'informe_*.pdf') { 'Coincide' }
if ($correo -match '^[^@]+@[^@]+\.[a-zA-Z]{2,}$') { $Matches }
$fecha = '2024-03-15' -replace '-', '/'
'Error 404: pagina no encontrada' -replace 'Error \d+', 'Fallo'
```

Los operadores de pertenencia son -contains e -in para comprobar si un valor esta en una coleccion, y -notcontains y -notin para el caso contrario. Ademas, -is comprueba el tipo de un objeto, -as intenta convertirlo y el operador -f formatea cadenas con posiciones marcadas por llaves.

```powershell
$frutas = @('manzana', 'pera', 'uva')
if ($frutas -contains 'pera') { 'La fruta esta en la lista' }
if ('uva' -in $frutas) { 'Tambien funciona al reves' }
$registro = '{0} - {1} - {2:C}' -f (Get-Date), 'Venta', 1250
$tipo = 42 -is [int]
```

Para el encadenamiento de comandos, Windows PowerShell 5.1 admite la redireccion numerica de flujos 1, 2, 3 y superiores y no admite los operadores && y ||, que si estan disponibles en PowerShell 7. En la version 5.1, la alternativa es encadenar cmdlets por tuberia o usar if sobre el resultado.

```powershell
Get-ChildItem -Path C:\Logs -ErrorAction SilentlyContinue > listado.txt
Get-Process -Name notepad -ErrorAction SilentlyContinue 2>&1 > salida.txt
# PowerShell 7 unicamente
Get-ChildItem C:\Temp && Get-ChildItem C:\Windows
```

### 7.3 Colecciones y objetos estructurados

Un array se crea con la arroba y las llaves y admite indexacion desde cero, operaciones de filtrado con Where-Object y medicion con la propiedad Count. Un rango de enteros se genera con la sintaxis punto entre numeros.

```powershell
$frutas = @('manzana', 'pera', 'uva')
$frutas[0]
$frutas.Count
$numeros = 1..10
$numeros | Where-Object { $_ % 2 -eq 0 }
$mezcla = @(1, 'texto', $true, (Get-Date))
```

La tabla hash almacena pares clave valor y se accede a cada elemento por su clave, ya sea con la sintaxis de punto o con corchetes. Es la estructura adecuada para parametros de cmdlets, para exportar a JSON y para construir objetos de forma inmediata.

```powershell
$usuario = @{
    Nombre = 'Eduardo'
    Edad = 34
    Activo = $true
}
$usuario.Nombre
$usuario['Edad']
$usuario.PSObject.Properties.Name
```

Los objetos personalizados se crean con [PSCustomObject] y permiten definir propiedades con orden fijo, ideal para construir salidas que se exportan a CSV o a JSON.

```powershell
$informe = [PSCustomObject]@{
    Fecha = Get-Date -Format yyyy-MM-dd
    Equipo = $env:COMPUTERNAME
    ServiciosDetenidos = (Get-Service | Where-Object Status -eq 'Stopped').Count
}
$informe | Format-List
```

### 7.4 Estructuras de control

La sentencia if evalua expresiones y admite las variantes elseif y else. Las condiciones se escriben entre llaves y pueden agruparse con parentesis cuando hay varias expresiones logicas.

```powershell
$disponibilidad = 85
if ($disponibilidad -ge 90) {
    'Nivel alto'
} elseif ($disponibilidad -ge 70) {
    'Nivel medio'
} else {
    'Nivel bajo'
}
```

La sentencia switch evalua un valor contra varios casos y admite comodines con -Wildcard, expresiones regulares con -Regex y varias entradas por patron. break sale del switch y continue pasa al siguiente elemento en casos de coleccion.

```powershell
$extension = '.txt'
switch ($extension) {
    '.txt'  { 'Texto plano' }
    '.log'  { 'Registro' }
    { $_ -like '*.doc*' } { 'Documento Word' }
    default { 'Extension desconocida' }
}
```

Los bucles disponibles son foreach, while, do while y for. El bucle foreach sobre una coleccion recorre todos los elementos y el foreach sobre el propio pipeline permite construir flujos funcionales con Select-Object, ForEach-Object o Measure-Object.

```powershell
foreach ($fruta in @('manzana', 'pera')) {
    Write-Output "Procesando $fruta"
}
$i = 0
while ($i -lt 3) {
    $i++
    Write-Output "Iteracion $i"
}
do {
    $respuesta = Read-Host 'Continuar? (s/n)'
} while ($respuesta -ne 's')
```

### 7.5 Funciones y scripts

Las funciones se definen con function y pueden recibir parametros declarados con param, usar tipado, valores por defecto, validaciones y los atributos de parametro de PowerShell. La forma avanzada, que anade los parametros comunes y la capacidad de recibir entrada por tuberia, se activa con el atributo CmdletBinding.

```powershell
function Get-Saludo {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Nombre,
        [ValidateSet('amable', 'formal')]
        [string]$Estilo = 'amable'
    )
    if ($Estilo -eq 'formal') {
        "Buenos dias, $Nombre."
    } else {
        "Hola, $Nombre!"
    }
}

Get-Saludo -Nombre Eduardo -Estilo formal
```

Las funciones se llaman como cualquier cmdlet, con parametros con guion o posicionales. Las funciones avanzadas admiten bloques Begin, Process y End, que se ejecutan una vez al inicio, una vez por cada objeto de entrada y una vez al final, lo que permite construir cmdlets propios con soporte de tuberia.

```powershell
function Add-Espacio {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline = $true)]
        [string]$Texto
    )
    begin { $contador = 0 }
    process {
        $contador++
        "$contador - $Texto"
    }
    end { "Total: $contador lineas" }
}
'a', 'b', 'c' | Add-Espacio
```

## Capitulo 8: PowerShell cmdlets esenciales

### 8.1 Gestion de ficheros y directorios

Get-ChildItem lista el contenido de un directorio o de cualquier proveedor y es el cmdlet mas utilizado. Admite -Path para la ruta, -Filter y -Include para patron de nombre, -Recurse para subcarpetas, -Depth para limitar la profundidad, -Force para incluir ocultos, -File y -Directory para filtrar por tipo y -Attributes por atributos.

```powershell
Get-ChildItem C:\Windows
Get-ChildItem -Path C:\Logs -Filter *.log -Recurse -File
Get-ChildItem C:\ -Force -Directory
Get-ChildItem C:\Proyecto -Recurse -Depth 2 -Include *.cs, *.config
Get-Item C:\Windows\notepad.exe
Test-Path C:\Windows\notepad.exe
Resolve-Path C:\Windows\..\Users
```

Get-Content lee ficheros y admite -Raw para cargar todo el contenido de una sola vez, -Tail para leer las ultimas lineas, -TotalCount para las primeras, -Encoding para la codificacion y -Wait para seguir un fichero en tiempo real. Set-Content y Add-Content escriben y anaden contenido, Out-File crea ficheros desde el pipeline y Clear-Content vacia el fichero sin eliminarlo.

```powershell
Get-Content -Path .\aplicacion.log -Tail 50
Get-Content -Path .\datos.txt -Raw
Get-Content .\config.ini -Encoding UTF8 | Where-Object { $_ -match '=' }
Set-Content -Path .\salida.txt -Value 'Primera linea'
Add-Content -Path .\salida.txt -Value 'Segunda linea'
Get-Process | Out-File -FilePath .\procesos.txt -Encoding utf8
Clear-Content .\salida.txt
```

New-Item crea ficheros y directorios con -ItemType Directory o -ItemType File y la opcion -Force crea la ruta completa aunque falten carpetas intermedias. Copy-Item copia, Move-Item mueve, Rename-Item cambia el nombre y Remove-Item elimina, con -Recurse para carpetas y -Confirm para pedir confirmacion antes de borrar.

```powershell
New-Item -ItemType Directory -Path C:\Informes\2024 -Force
New-Item -ItemType File -Path C:\Temp\registro.log -Force
Copy-Item -Path C:\Datos -Destination C:\Respaldos -Recurse
Copy-Item -Path .\plantilla.xlsx -Destination .\informe.xlsx
Move-Item -Path C:\Descargas\*.pdf -Destination C:\Documentos
Rename-Item -Path .\informe_v1.docx -NewName informe_final.docx
Remove-Item -Path C:\Temp\*.tmp -Force
Remove-Item -Path C:\Carpeta -Recurse -Confirm
```

Para operaciones con ficheros existe tambien Get-FileHash, que calcula codigos hash como SHA256 para verificar integridad, y Select-String, que busca texto dentro de ficheros equivalente a grep o findstr.

```powershell
Get-FileHash -Path .\instalador.exe -Algorithm SHA256
Select-String -Path C:\Logs\*.log -Pattern 'ERROR' -SimpleMatch
Select-String -Path C:\Logs\*.log -Pattern '^\d{4}-\d{2}-\d{2}' | Select-Object -First 10
Get-ChildItem C:\Logs\*.log | Select-String -Pattern 'critico'
```

### 8.2 Gestion de procesos

Get-Process muestra los procesos en ejecucion con propiedades como Id, Name, WS para memoria de trabajo, CPU para tiempo de procesador y StartTime. Stop-Process termina procesos, Start-Process lanza programas y Wait-Process espera a que terminen.

```powershell
Get-Process
Get-Process | Sort-Object WS -Descending | Select-Object -First 10
Get-Process -Name notepad
Stop-Process -Name notepad -Force
Stop-Process -Id 1234 -Force
Start-Process -FilePath notepad.exe -ArgumentList 'C:\nota.txt'
Start-Process -FilePath powershell -ArgumentList '-File', 'C:\Scripts\tarea.ps1' -WindowStyle Hidden
Start-Process -FilePath msiexec -ArgumentList '/i', 'paquete.msi', '/qn' -Verb RunAs
Wait-Process -Name instalador
```

La propiedad WS esta expresada en bytes, por lo que se acostumbra a dividirla por 1048576 para obtener megabytes. La sintaxis -ArgumentList admite una cadena unica o un arreglo de argumentos, y la opcion -Verb RunAs eleva el proceso con el Control de Cuentas de Usuario.

### 8.3 Gestion de servicios

Get-Service lista los servicios con su nombre, titulo y estado, y Start-Service, Stop-Service, Restart-Service, Suspend-Service y Resume-Service los controlan. Set-Service modifica propiedades como el tipo de inicio y New-Service registra un servicio nuevo.

```powershell
Get-Service
Get-Service | Where-Object Status -eq 'Stopped'
Get-Service -Name Spooler | Format-List *
Start-Service -Name Spooler
Stop-Service -Name Spooler -Force
Restart-Service -Name Spooler -Force
Set-Service -Name Spooler -StartupType Disabled
New-Service -Name MiServicio -BinaryPathName 'C:\Ruta\servicio.exe' -DisplayName 'Mi Servicio' -StartupType Automatic
```

El estado de un servicio se expone con la propiedad Status y el tipo de inicio con StartType. Para filtrar servicios criticos se suele construir una lista de nombres y usar Where-Object con -notin, un patron habitual de comprobacion en scripts de monitorizacion.

```powershell
$criticos = @('WinDefend', 'wuauserv', 'Dnscache', 'LanmanWorkstation')
Get-Service -Name $criticos | Where-Object Status -ne 'Running' |
    Select-Object Name, Status, StartType
```

### 8.4 Salida y formato

La salida de un cmdlet viaja por el pipeline como objetos y solo se convierte a texto al llegar a la consola o a un fichero. Format-Table genera tablas y admite -AutoSize para ajustar el ancho, -Property para elegir columnas, -Wrap para partir lineas largas y propiedades calculadas con etiqueta y expresion. Format-List genera listas de propiedades y es util cuando hay muchos campos o cadenas largas.

```powershell
Get-Service | Format-Table Name, Status, StartType -AutoSize
Get-Process | Format-Table Name, @{n='MB';e={[math]::Round($_.WS/1MB)}} -AutoSize
Get-Service Spooler | Format-List *
Get-ChildItem *.txt | Format-Wide -Column 3
```

Las salidas a fichero se realizan con Out-File, Export-Csv e Export-Clixml. Export-Csv crea un fichero con valores separados por comas que puede abrirse en hoja de calculo, y admite -NoTypeInformation para evitar la cabecera inicial, -Delimiter para cambiar el separador y -Encoding para la codificacion. ConvertTo-Json y ConvertFrom-Json permiten intercambiar datos con APIs web.

```powershell
Get-Service | Export-Csv -Path .\servicios.csv -NoTypeInformation -Encoding UTF8
Import-Csv .\servicios.csv | Where-Object Status -eq 'Running'
Get-Process | Select-Object Name, WS | ConvertTo-Json -Depth 3
$datos = Get-Content .\datos.json -Raw | ConvertFrom-Json
Get-Service | ConvertTo-Clixml -Path .\servicios.xml
Import-Clixml .\servicios.xml
```

Out-Null descarta la salida, util para cmdlets como New-Item o Remove-Item cuyo resultado no interesa mostrar, y Out-String convierte objetos en cadenas de texto cuando realmente se necesita texto, por ejemplo para enviarlos a otro programa. Tee-Object guarda una copia en un fichero mientras continua el pipeline.

```powershell
New-Item -ItemType Directory -Path C:\Temp\salida -Force | Out-Null
Get-Service | Tee-Object -FilePath .\servicios_copia.txt | Where-Object Status -eq 'Running'
$texto = Get-Service | Out-String
```

### 8.5 Utilidades generales

PowerShell incluye cmdlets para tareas cotidianas como Get-Date para la fecha y hora, Set-Date para cambiarla con privilegios de administrador, Get-Random para numeros aleatorios, Start-Sleep para esperas, Read-Host para entrada de usuario, Clear-Host para limpiar la consola y Measure-Object para contar y agregar valores numericos o de texto.

```powershell
Get-Date
Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
(Get-Date).AddDays(-30)
Get-Random -Minimum 1 -Maximum 100
Start-Sleep -Seconds 5
$nombre = Read-Host 'Introduce tu nombre'
Get-Content .\datos.txt | Measure-Object -Line -Word -Character
Get-Process | Measure-Object -Property WS -Sum -Average -Maximum
Clear-Host
```

Measure-Object admite -Sum, -Average, -Minimum y -Maximum para propiedades numericas y -Line, -Word y -Character para texto, devolviendo objetos con las estadisticas calculadas. Invoke-Item abre un fichero o carpeta con su aplicacion predeterminada, equivalente a escribir la ruta en el Explorer.

```powershell
Invoke-Item .\informe.pdf
Invoke-Item C:\Windows
Get-History
Invoke-History -Id 5
```

## Capitulo 9: Filtrado y transformacion en el pipeline

### 9.1 Where-Object

Where-Object filtra objetos del pipeline y admite tres sintaxis. La forma clasica utiliza un scriptblock con $_, la forma simplificada compara una propiedad con un operador y un valor, y la forma con operadores logicos permite combinar varias condiciones.

```powershell
Get-Process | Where-Object { $_.WS -gt 100MB }
Get-Process | Where-Object WS -GT 100MB
Get-Service | Where-Object { $_.Status -eq 'Running' -and $_.StartType -eq 'Automatic' }
Get-ChildItem *.log | Where-Object LastWriteTime -LT (Get-Date).AddDays(-7)
Get-Process | Where-Object Name -Like '*chrome*'
```

Los operadores admitidos en la forma simplificada incluyen -EQ, -NE, -GT, -GE, -LT, -LE, -LIKE, -MATCH, -CONTAINS e -IN. Cuando la condicion es compleja, por ejemplo una combinacion con -and y -or o una llamada a un metodo, se recurre al scriptblock con $_.

### 9.2 Select-Object

Select-Object elige propiedades, extrae elementos concreto y construye proyecciones personalizadas. Las opciones -First y -Last toman los primeros o ultimos elementos, -Skip omite elementos, -Property selecciona campos, -ExpandProperty devuelve el valor de una propiedad sin envolverlo y -Unique elimina duplicados.

```powershell
Get-Process | Select-Object -First 5
Get-Process | Sort-Object WS -Descending | Select-Object -First 10 -Property Name, WS
Get-Service | Select-Object -ExpandProperty Name
Get-ChildItem -Recurse -File | Select-Object -Unique Extension
Get-Content .\datos.txt | Select-Object -Skip 10 -First 20
```

Las propiedades calculadas se definen con una tabla que contiene Name o N para la etiqueta y Expression o E para la expresion que genera el valor. Es la forma habitual de anadir columnas derivadas como megabytes, gigabytes o fechas formateadas.

```powershell
Get-Process |
    Select-Object Name,
        @{n='MB';e={[math]::Round($_.WS/1MB,2)}},
        @{n='CPU';e={$_.CPU}}
Get-ChildItem -File |
    Select-Object Name,
        @{n='TamanoKB';e={[math]::Round($_.Length/1KB,2)}},
        @{n='Modificado';e={$_.LastWriteTime}}
```

Select-Object es tambien la alternativa a Group-Object para paginar resultados grandes y, combinado con Where-Object, resuelve la mayoria de las consultas habituales sobre colecciones de objetos.

### 9.3 Ordenacion, agrupacion y medicion

Sort-Object ordena por una o varias propiedades y admite -Descending para invertir el orden. Group-Object agrupa elementos por el valor de una propiedad y devuelve grupos con Count y Group, y Measure-Object calcula estadisticas sobre colecciones numericas o de texto.

```powershell
Get-ChildItem -File | Sort-Object Length -Descending | Select-Object -First 10
Get-Service | Sort-Object Status, Name
Get-Process | Group-Object Name | Sort-Object Count -Descending | Select-Object -First 5
Get-ChildItem -Recurse -File | Group-Object Extension | Sort-Object Count -Descending
Get-ChildItem *.log | Measure-Object -Property Length -Sum
```

Compare-Object compara dos colecciones y senala los elementos que estan solo en una de ellas, lo que permite detectar cambios entre dos listados, dos configuraciones o dos instantaneas de un directorio.

```powershell
$antes = Get-ChildItem C:\Config -File | Select-Object -ExpandProperty Name
$ahora = Get-ChildItem C:\Config -File | Select-Object -ExpandProperty Name
Compare-Object -ReferenceObject $antes -DifferenceObject $ahora
```

### 9.4 Ejemplos combinados

Los patrones mas frecuentes combinan obtencion, filtro, transformacion y salida en un unico flujo. Listar los ficheros mas grandes de un arbol de carpetas, por ejemplo, se resuelve obteniendo los ficheros, filtrando por tamano, ordenando de forma descendente y proyectando solo las columnas interesantes.

```powershell
Get-ChildItem C:\Datos -Recurse -File |
    Where-Object { $_.Length -gt 10MB } |
    Sort-Object Length -Descending |
    Select-Object FullName, @{n='MB';e={[math]::Round($_.Length/1MB,2)}}
```

Un informe de servicios con el numero de servicios automaticos detenidos se construye filtrando por tipo de inicio, excluyendo los que estan en ejecucion y contando el resultado.

```powershell
Get-Service |
    Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -ne 'Running' } |
    Select-Object Name, DisplayName, Status |
    Format-Table -AutoSize
$detenidos = Get-Service |
    Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -ne 'Running' } |
    Measure-Object
$detenidos.Count
```

Para resumir el uso de disco por directorio se agrupan los ficheros por su carpeta contenedora y se suman los largos, un patron que requiere agregar la propiedad Length de cada grupo.

```powershell
Get-ChildItem C:\Datos -Recurse -File |
    Group-Object DirectoryName |
    Select-Object @{n='Carpeta';e={$_.Name}},
                  @{n='Ficheros';e={$_.Count}},
                  @{n='TotalMB';e={[math]::Round(($_.Group | Measure-Object Length -Sum).Sum/1MB,2)}} |
    Sort-Object TotalMB -Descending |
    Format-Table -AutoSize
```

Estos ejemplos muestran la diferencia fundamental respecto a CMD: mientras que en el interprete clasico habria que capturar la salida de dir en un fichero, analizarla con findstr y recalcular los totales con comandos de texto, en PowerShell los datos conservan su estructura durante todo el proceso y cada paso opera sobre propiedades tipadas.

## Capitulo 10: Scripts, politicas de ejecucion y modulos

### 10.1 Ejecutar scripts y politica de ejecucion

Los scripts de PowerShell tienen extension .ps1 y se ejecutan llamando al fichero con el operador de invocacion &, escribiendo su ruta relativa o absoluta. La forma habitual desde el propio directorio del script es .\nombre.ps1, porque PowerShell no busca en el directorio actual sino unicamente en el PATH.

```powershell
& .\informe.ps1
.\informe.ps1 -Servidor SERV01
. .\utilidades.ps1
powershell -NoProfile -File C:\Scripts\informe.ps1
```

La sentencia con el punto delante ejecuta el script en el ambito actual, operacion conocida como dot sourcing, de modo que sus funciones y variables quedan disponibles en la sesion despues de terminar. Es el mecanismo tipico para cargar un modulo propio o un fichero de configuracion.

La politica de ejecucion controla las condiciones bajo las que PowerShell permite lanzar scripts. Microsoft aclara en su documentacion que la politica de ejecucion no es un sistema de seguridad que impida la ejecucion maliciosa, sino un mecanismo que reduce el riesgo de ejecucion accidental de scripts no verificados. Los valores disponibles son Restricted, que impide todos los scripts y es el predeterminado en equipos cliente con Windows PowerShell 5.1, AllSigned, que exige todos los scripts firmados digitalmente, RemoteSigned, que permite los scripts locales y exige firma o zona de confianza para los descargados, Unrestricted, que los ejecuta avisando de los descargados, Bypass, que no muestra avisos, y Undefined, que no fija valor y permite que se herede el de un ambito superior.

```powershell
Get-ExecutionPolicy
Get-ExecutionPolicy -List
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

La precedencia de los ambitos se aplica en este orden: MachinePolicy, UserPolicy, Process, CurrentUser y LocalMachine, de modo que el valor efectivo es el primero definido empezando por la izquierda. El valor mas practico para el uso diario es RemoteSigned aplicado al ambito CurrentUser, que no requiere privilegios de administrador. Los ficheros descargados de internet conservan el marcador de zona Mark of the Web y pueden desbloquearse con Unblock-File o desde las propiedades del fichero.

```powershell
Unblock-File -Path .\script_descargado.ps1
Get-Item .\script_descargado.ps1 -Stream * | Select-Object Stream
powershell -ExecutionPolicy Bypass -File .\script.ps1
```

### 10.2 Parametros de script

Un script con parametros declara el bloque param en su primera linea efectiva y puede definir tipos, valores por defecto, validaciones y parametros obligatorios. Los mismos atributos de parametro que en las funciones se aplican aqui, y los parametros se pasan al script por nombre o por posicion.

```powershell
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Ruta,

    [switch]$Forzar,

    [ValidateRange(1, 100)]
    [int]$Dias = 7,

    [ValidateSet('CSV', 'JSON', 'TXT')]
    [string]$Formato = 'TXT'
)

$fechaLimite = (Get-Date).AddDays(-$Dias)
Get-ChildItem -Path $Ruta -Recurse -File |
    Where-Object { $_.LastWriteTime -lt $fechaLimite } |
    Format-Table Name, LastWriteTime -AutoSize

if ($Forzar) { Write-Output 'Modo forzado activo' }
```

La validacion automatica se produce antes de ejecutar el cuerpo del script: si un valor no cumple ValidateSet, ValidateRange o ValidateNotNullOrEmpty, el script se detiene con un error de parametro. El interruptor -Forzar llega como un booleano y no admite valor a menos que se escriba explicitamente -Forzar:$true.

### 10.3 Manejo de errores

PowerShell distingue entre errores no terminantes, que continuan la ejecucion y se registran en $error, y errores terminantes, que detienen la sentencia actual. Los cmdlets generan errores no terminantes por defecto y el parametro -ErrorAction permite elevar el comportamiento a Stop para capturarlos con try y catch.

```powershell
try {
    $contenido = Get-Content -Path C:\NoExiste\fichero.txt -ErrorAction Stop
} catch [System.IO.DirectoryNotFoundException] {
    Write-Error "El directorio no existe: $($_.Exception.Message)"
} catch {
    Write-Error "Error inesperado: $_"
} finally {
    Write-Output 'Proceso finalizado'
}
```

La variable automatica $_ dentro del catch contiene el error actual y su propiedad Exception explica el fallo. El bloque finally se ejecuta siempre, incluso si hubo error, y es el lugar indicado para cerrar recursos o registrar la finalizacion del proceso.

El prefijo de sesion $ErrorActionPreference controla el comportamiento por defecto de todos los cmdlets; sus valores mas usados son Continue, que muestra el error y sigue, Stop, que convierte cualquier error en terminante, y SilentlyContinue, que omite la visualizacion. En scripts robustos se combina el establecimiento temporal del prefijo con try y catch.

```powershell
$ErrorActionPreference = 'Stop'
try {
    Get-Service -Name NombreQueNoExiste
} catch {
    Write-Warning "No se pudo obtener el servicio: $_"
} finally {
    $ErrorActionPreference = 'Continue'
}
```

Tras ejecutar un programa externo como cmd.exe, robocopy o git, hay que comprobar $LASTEXITCODE porque los errores nativos no generan excepciones de PowerShell de forma automatica en Windows PowerShell 5.1. Esta distincion es una de las causas mas frecuentes de scripts que reportan exito cuando el comando externo fallo.

```powershell
& robocopy.exe C:\Origen D:\Destino /E
if ($LASTEXITCODE -ge 8) {
    throw "robocopy fallo con codigo $LASTEXITCODE"
}
```

El historial de errores se consulta con $error, que guarda los ultimos errores con el mas reciente en la posicion cero, y cada cmdlet puede capturar sus propios errores con -ErrorVariable. throw genera un error terminante propio y trap permite definir un manejador de errores a nivel de script.

### 10.4 Modulos

Un modulo es un conjunto de cmdlets, funciones y variables empaquetados en un fichero .psm1 con su manifiesto .psd1. Get-Module -ListAvailable muestra los modulos instalados en el sistema, Import-Module los carga en la sesion y Get-Command -Module lista los comandos que aporta cada modulo.

```powershell
Get-Module -ListAvailable | Sort-Object Name
Get-Module
Import-Module ActiveDirectory
Get-Command -Module ActiveDirectory
Remove-Module ActiveDirectory
```

PowerShell carga automaticamente los modulos cuando se invoca un cmdlet que pertenece a uno no cargado, caracteristica conocida como autoimportacion de modulos. Para evitar esa carga implicita y mejorar el rendimiento, se puede desactivar con $PSModuleAutoLoadingPreference.

Los modulos adicionales se instalan con PowerShellGet desde el repositorio de PowerShell Gallery, preferiblemente en el ambito del usuario para no requerir elevacion. Microsoft distribuye asi modulos como PSWindowsUpdate, PSScriptAnalyzer o la extension de telemetria de PowerShell.

```powershell
Install-Module -Name PSScriptAnalyzer -Scope CurrentUser
Update-Module -Name PSScriptAnalyzer
Find-Module -Name '*WindowsUpdate*'
Save-Module -Name MyModule -Path C:\Modulos
$env:PSModulePath -split ';'
```

Los modulos propios se crean con Export-ModuleMember para controlar que funciones quedan visibles y se cargan con dot sourcing o con Import-Module apuntando al fichero .psm1. El manifiesto .psd1 documenta version, dependencias y cmdlets exportados.

### 10.5 Perfiles y sesion

El perfil de PowerShell se ejecuta cada vez que se inicia una sesion y es el lugar habitual para personalizar el entorno: alias propios, variables globales, funcion de limpieza o importacion de modulos. La variable $PROFILE apunta al fichero de perfil del host actual y existen varios niveles segun el host y el ambito.

```powershell
$PROFILE
$PROFILE.CurrentUserCurrentHost
$PROFILE.AllUsersCurrentHost
Test-Path $PROFILE
New-Item -Path $PROFILE -ItemType File -Force
notepad $PROFILE
```

En Windows PowerShell 5.1, el perfil del usuario se guarda habitualmente en Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1, mientras que PowerShell 7 usa Documents\PowerShell\Microsoft.PowerShell_profile.ps1, de modo que ambas versiones mantienen perfiles separados. El perfil tambien esta sujeto a la politica de ejecucion y puede requerir el desbloqueo del fichero.

Para registrar una sesion completa se usa Start-Transcript, que guarda todo lo mostrado en la consola, y Stop-Transcript para finalizarlo. Es util para documentar intervenciones de administracion o para depurar scripts ejecutados en segundo plano.

```powershell
Start-Transcript -Path C:\Log\sesion_$(Get-Date -Format yyyyMMdd_HHmmss).txt
Get-Service | Where-Object Status -eq 'Stopped'
Stop-Transcript
```

### 10.6 Depuracion y analisis

La depuracion se apoya en puntos de interrupcion, en el paso a paso y en variables de contexto. Set-PSDebug -Step activa la confirmacion paso a paso de cada sentencia y Set-PSDebug -Trace permite trazar la ejecucion con niveles crecientes de detalle. En Visual Studio Code, los puntos de interrupcion se colocan en el margen izquierdo y las teclas de control del depurador permiten avanzar sentencia a sentencia.

```powershell
Set-StrictMode -Version Latest
Set-PSDebug -Trace 1
Get-Process -Name 'no_existe' -ErrorAction SilentlyContinue
Set-PSDebug -Off
```

Set-StrictMode -Version Latest activa el modo estricto, que convierte en error el acceso a variables no definidas y a propiedades inexistentes, una practica recomendable al inicio de cualquier script serio porque convierte errores silenciosos en fallos visibles.

El analizador de estilo PSScriptAnalyzer comprueba el cumplimiento de buenas practicas de escritura, detecta cmdlets poco eficientes, nomenclatura incorrecta y errores comunes, y puede autocorregir parte de los hallazgos.

```powershell
Invoke-ScriptAnalyzer -Path .\informe.ps1
Invoke-ScriptAnalyzer -Path .\informe.ps1 -Severity Warning
Invoke-ScriptAnalyzer -Path .\informe.ps1 -Fix
```

## Capitulo 11: Administracion de Windows con PowerShell

### 11.1 Informacion del sistema

Los cmdlets de CIM proporcionan informacion de hardware y sistema. Win32_OperatingSystem resume el sistema operativo, Win32_Processor la CPU, Win32_LogicalDisk los discos logicos, Win32_PhysicalMemory la memoria fisica y Win32_NetworkAdapterConfiguration la configuracion de red.

```powershell
Get-CimInstance Win32_OperatingSystem |
    Select-Object Caption, Version, BuildNumber, LastBootUpTime, TotalVisibleMemorySize
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
    Select-Object DeviceID, @{n='TotalGB';e={[math]::Round($_.Size/1GB,1)}},
                  @{n='FreeGB';e={[math]::Round($_.FreeSpace/1GB,1)}}
Get-CimInstance Win32_PhysicalMemory |
    Measure-Object -Property Capacity -Sum |
    Select-Object @{n='RAM_GB';e={[math]::Round($_.Sum/1GB,1)}}
```

El filtro -Filter usa la sintaxis WQL, equivalente a la clausula WHERE de las consultas WMI. La alternativa clasica Get-WmiObject esta disponible solo en Windows PowerShell 5.1 y se considera obsoleta; en PowerShell 7 existe unicamente Get-CimInstance, por lo que los scripts modernos deben usar la API de CIM.

Los volumenes y discos fisicos se gestionan con los cmdlets del modulo Storage, que ofrecen una vista mas moderna que Win32_LogicalDisk.

```powershell
Get-Volume | Where-Object DriveType -eq 'Fixed' |
    Select-Object DriveLetter, FileSystemLabel, @{n='GBLibres';e={[math]::Round($_.SizeRemaining/1GB,1)}}
Get-Disk | Select-Object Number, FriendlyName, OperationalStatus, @{n='SizeGB';e={[math]::Round($_.Size/1GB)}}
Get-PhysicalDisk | Select-Object FriendlyName, MediaType, HealthStatus
Get-ComputerInfo | Select-Object WindowsProductName, OsVersion, CsManufacturer, CsModel
```

### 11.2 Usuarios y grupos locales

El modulo Microsoft.PowerShell.LocalAccounts gestiona cuentas locales con Get-LocalUser, New-LocalUser, Set-LocalUser, Disable-LocalUser, Enable-LocalUser y Remove-LocalUser, y los grupos con Get-LocalGroup, Add-LocalGroupMember y Get-LocalGroupMember.

```powershell
Get-LocalUser | Select-Object Name, Enabled, LastLogon, PasswordLastSet
Get-LocalUser -Name invitado | Format-List *
$contrasena = Read-Host 'Contrasena nueva' -AsSecureString
New-LocalUser -Name tecnico -Password $contrasena -FullName 'Tecnico de soporte' -Description 'Cuenta de mantenimiento'
Add-LocalGroupMember -Group Administrators -Member tecnico
Disable-LocalUser -Name tecnico
Get-LocalGroupMember -Group Administrators |
    Select-Object Name, PrincipalSource, ObjectClass
```

La contrasena se crea con ConvertTo-SecureString -AsPlainText -Force cuando proviene de una cadena literal, aunque en scripts productivos se prefiere la entrada protegida con Read-Host -AsSecureString o la obtencion desde un vault. Las cuentas del dominio se gestionan con el modulo ActiveDirectory de RSAT, que aporta Get-ADUser, Get-ADGroup y New-ADUser.

### 11.3 Programacion de tareas

El modulo ScheduledTasks expone el Programador de tareas de Windows. Get-ScheduledTask lista las tareas, Get-ScheduledTaskInfo muestra su historial y estado, y las tareas nuevas se componen con una accion, un disparador y una configuracion que se registran con Register-ScheduledTask.

```powershell
Get-ScheduledTask | Where-Object State -eq 'Ready' |
    Select-Object TaskName, TaskPath, State | Format-Table -AutoSize
Get-ScheduledTask -TaskName 'Respaldo diario' | Get-ScheduledTaskInfo

$accion = New-ScheduledTaskAction -Execute 'PowerShell.exe' `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\Scripts\respaldo.ps1"'
$disparador = New-ScheduledTaskTrigger -Daily -At 08:00
$ajustes = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Respaldo diario' -Action $accion -Trigger $disparador -Settings $ajustes -Description 'Respaldo incremental diario'
```

Las tareas existentes se ejecutan con Start-ScheduledTask, se desactivan con Disable-ScheduledTask y se eliminan con Unregister-ScheduledTask. El historial de ejecucion de cada tarea se consulta con Get-ScheduledTaskInfo, que devuelve la ultima ejecucion, el ultimo resultado y la proxima programacion.

### 11.4 Sucesos de Windows

El Visor de sucesos se consulta con Get-WinEvent, que admite -LogName para elegir el registro, -MaxEvents para limitar resultados y -FilterHashtable para filtrar por nivel, origen, identificador o intervalo de fechas.

```powershell
Get-WinEvent -LogName System -MaxEvents 20 |
    Select-Object TimeCreated, Id, LevelDisplayName, ProviderName |
    Format-Table -AutoSize

Get-WinEvent -FilterHashtable @{
    LogName   = 'System'
    Level     = 1, 2, 3
    StartTime = (Get-Date).AddHours(-24)
} -MaxEvents 100 |
    Select-Object TimeCreated, Id, Message |
    Format-Table -AutoSize
```

Los niveles 1, 2 y 3 corresponden a Critico, Error y Advertencia. Para exportar resultados a CSV se anade una seleccion de propiedades y Export-Csv, y para analizar por origen se usa Group-Object sobre la propiedad ProviderName. El cmdlet Get-EventLog es la version clasica, limitada al historial corto de los registros tradicionales, mientras que Get-WinEvent accede a todos los registros de canal de Windows.

### 11.5 Remoting y sesiones remotas

PowerShell remoting se basa en WinRM, protocolo que escucha en los puertos 5985 HTTP y 5986 HTTPS. Enable-PSRemoting configura el equipo para recibir sesiones remotas y WinRM quickconfig es el equivalente desde la linea de comandos.

```powershell
Enable-PSRemoting -Force
winrm quickconfig
Get-Service WinRM | Start-Service
```

Para equipos en grupo de trabajo es necesario declarar los hosts de confianza con Set-Item o con winrm set. En entornos de dominio, la autenticacion y la autorizacion se resuelven con las directivas predeterminadas.

```powershell
Set-Item -Path WSMan:\localhost\Client\TrustedHosts -Value 'SRV01,SRV02' -Force
Get-Item WSMan:\localhost\Client\TrustedHosts
```

Enter-Pssession abre una sesion interactiva con otro equipo y Exit-Pssession la cierra, mientras que Invoke-Command ejecuta un bloque de script de forma remota sin abrir sesion interactiva, que es la forma recomendada para automatizacion porque admite varios equipos a la vez.

```powershell
Enter-PSSession -ComputerName SRV01
Get-Service
Exit-PSSession

Invoke-Command -ComputerName SRV01 -ScriptBlock {
    Get-Service | Where-Object Status -eq 'Stopped' | Select-Object Name, Status
}

$equipos = 'SRV01', 'SRV02', 'SRV03'
Invoke-Command -ComputerName $equipos -ScriptBlock {
    Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
        Select-Object PSComputerName, DeviceID, FreeSpace
}
```

Los bloques remotos reciben parametros con -ArgumentList y se referencian con param dentro del scriptblock. Las variables de la sesion local no viajan al equipo remoto salvo que se declaren con el prefijo $using:, mecanismo disponible en remoting desplegado y en Jobs.

```powershell
$limite = 10GB
Invoke-Command -ComputerName SRV01 -ScriptBlock {
    param($limiteLocal)
    Get-ChildItem C:\Logs -Recurse -File | Where-Object Length -gt $limiteLocal
} -ArgumentList $limite
```

Las credenciales se obtienen con Get-Credential, que presenta un dialogo de entrada protegido, y se pasan al parametro -Credential de los cmdlets remotos. Microsoft recomienda evitar credenciales en texto plano dentro de scripts y utilizar cuentas dedicadas con privilegios minimos para el remoting.

### 11.6 Red con PowerShell

El modulo NetTCPIP expone la configuracion de red con cmdlets como Get-NetIPAddress, Get-NetAdapter, Get-NetTCPConnection y Set-DnsClientServerAddress, y Test-NetConnection realiza pruebas de conectividad con resolucion DNS, ruta y puerto.

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
    Select-Object InterfaceAlias, IPAddress, PrefixLength
Get-NetAdapter | Select-Object Name, Status, LinkSpeed
Get-NetTCPConnection -State Listen |
    Select-Object LocalAddress, LocalPort, OwningProcess |
    Sort-Object LocalPort
Test-NetConnection www.microsoft.com -Port 443
Resolve-DnsName www.microsoft.com
```

Test-NetConnection devuelve un objeto con propiedades como TcpTestSucceeded, RemoteAddress y HopLimit, que permite construir comprobaciones de salud de servicios web sin salir de PowerShell. Para peticiones HTTP, Invoke-WebRequest obtiene paginas y Invoke-RestMethod datos de APIs JSON.

```powershell
$respuesta = Invoke-WebRequest -Uri 'https://learn.microsoft.com' -UseBasicParsing
$respuesta.StatusCode
$respuesta.Content.Length

$datos = Invoke-RestMethod -Uri 'https://api.example.com/v1/estado'
$datos | ConvertTo-Json -Depth 5
```

En Windows PowerShell 5.1, Invoke-WebRequest utiliza por defecto el motor de Internet Explorer para analizar el contenido y falla en equipos sin su primera ejecucion completada; por eso se anade -UseBasicParsing, parametro que en PowerShell 7 es innecesario porque siempre se usa el analizador propio.

### 11.7 Windows Update y mantenimiento

PowerShell puede gestionar Windows Update con el modulo PSWindowsUpdate instalado desde PowerShell Gallery, o bien mediante el servicio COM de actualizacion, sin dependencias externas.

```powershell
Install-Module -Name PSWindowsUpdate -Scope CurrentUser
Get-WindowsUpdate
Install-WindowsUpdate -AcceptAll -AutoReboot
```

Los equivalentes sin modulos incluyen la reiniciacion del servicio wuauserv, la ejecucion de DISM para la tienda de componentes y el lanzamiento de la tarea de mantenimiento predeterminada de Windows con la herramienta de limpieza.

```powershell
Restart-Service -Name wuauserv -Force
DISM.exe /Online /Cleanup-Image /RestoreHealth
Start-ScheduledTask -TaskName '\Microsoft\Windows\DiskCleanup\SilentCleanup'
```

## Capitulo 12: Tareas practicas frecuentes

### 12.1 Listar los ficheros mas grandes

En CMD se combinan dir con sort y find, y en PowerShell el pipeline de objetos lo resuelve sin analizar texto.

```cmd
dir /s /b *.zip
for /r C:\ %f in (*.zip) do @echo %~zf %f
```

```powershell
Get-ChildItem C:\ -Recurse -File -Filter *.zip -ErrorAction SilentlyContinue |
    Sort-Object Length -Descending |
    Select-Object -First 10 FullName, @{n='MB';e={[math]::Round($_.Length/1MB,2)}}
```

### 12.2 Buscar un texto en todos los ficheros

```cmd
findstr /s /i /n "Error de conexion" C:\Logs\*.log
```

```powershell
Get-ChildItem C:\Logs -Recurse -Include *.log, *.txt |
    Select-String -Pattern 'Error de conexion' |
    Select-Object -First 20 Path, LineNumber, Line
```

### 12.3 Renombrar ficheros en bloque

```cmd
ren documento.docx documento_v2.docx
for %f in (*.txt) do ren "%f" "viejo_%f"
```

```powershell
Get-ChildItem *.jpg |
    Rename-Item -NewName { 'foto_2024_' + $_.Name }
Get-ChildItem C:\Descargas\*.pdf |
    ForEach-Object { Rename-Item -Path $_.FullName -NewName ($_.BaseName + '_leido' + $_.Extension) }
```

### 12.4 Copiar solo los ficheros recientes

```cmd
robocopy C:\Datos D:\Copia /E /maxage:7
xcopy C:\Datos D:\Copia /E /D /Y /I
```

```powershell
$limite = (Get-Date).AddDays(-7)
Get-ChildItem C:\Datos -Recurse -File |
    Where-Object LastWriteTime -GT $limite |
    ForEach-Object {
        $destino = Join-Path 'D:\Copia' $_.FullName.Substring(3)
        Copy-Item -Path $_.FullName -Destination $destino -Force
    }
```

### 12.5 Informe de espacio en disco

```cmd
wmic logicaldisk get caption,freespace,size
fsutil volume diskfree C:
```

```powershell
Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
    Select-Object DeviceID,
        @{n='TotalGB';e={[math]::Round($_.Size/1GB,1)}},
        @{n='LibreGB';e={[math]::Round($_.FreeSpace/1GB,1)}},
        @{n='Usado%';e={[math]::Round(($_.Size-$_.FreeSpace)/$_.Size*100,1)}} |
    Format-Table -AutoSize
```

### 12.6 Detectar servicios criticos detenidos

```cmd
sc query WinDefend | findstr STATE
sc query wuauserv | findstr STATE
```

```powershell
$criticos = 'WinDefend', 'wuauserv', 'Dnscache', 'EventLog', 'Spooler'
$estado = Get-Service -Name $criticos |
    Select-Object Name, DisplayName, Status, StartType
$estado | Format-Table -AutoSize
$detenidos = $estado | Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -ne 'Running' }
if ($detenidos) {
    $detenidos | ForEach-Object { Start-Service -Name $_.Name }
}
```

### 12.7 Procesos que mas memoria consumen

```cmd
tasklist /fi "memusage gt 100000"
taskkill /im nombre.exe /f
```

```powershell
Get-Process |
    Sort-Object WS -Descending |
    Select-Object -First 10 Id, Name, @{n='MB';e={[math]::Round($_.WS/1MB)}}, @{n='CPU';e={$_.CPU}}
Get-Process |
    Where-Object { $_.WS -gt 100MB } |
    Select-Object Name, @{n='MB';e={[math]::Round($_.WS/1MB)}}
```

### 12.8 Consultar puertos en escucha

```cmd
netstat -ano | findstr LISTENING
netstat -an | findstr :443
```

```powershell
Get-NetTCPConnection -State Listen |
    Select-Object LocalAddress, LocalPort, @{n='Proceso';e={(Get-Process -Id $_.OwningProcess).Name}} |
    Sort-Object LocalPort |
    Format-Table -AutoSize
```

### 12.9 Contar lineas y palabras de un fichero

```cmd
find /c /v "" fichero.txt
for %f in (*.log) do @find /c /v "%f"
```

```powershell
Get-Content .\fichero.txt | Measure-Object -Line -Word -Character
(Get-Content .\fichero.txt -Raw).Split("`n").Count
Get-ChildItem *.log | ForEach-Object {
    $m = Get-Content $_.FullName | Measure-Object -Line
    [PSCustomObject]@{ Fichero = $_.Name; Lineas = $m.Lines }
}
```

### 12.10 Filtrar y exportar datos de un CSV

```cmd
findstr /i "pendiente" datos.csv > pendientes.csv
```

```powershell
Import-Csv .\tickets.csv |
    Where-Object { $_.Estado -eq 'Pendiente' -and $_.Prioridad -in 'Alta', 'Critica' } |
    Sort-Object Fecha |
    Export-Csv .\pendientes.csv -NoTypeInformation -Encoding UTF8

Import-Csv .\ventas.csv |
    Group-Object Region |
    Select-Object Name, Count, @{n='Total';e={($_.Group | Measure-Object -Property Importe -Sum).Sum}} |
    Export-Csv .\resumen_region.csv -NoTypeInformation
```

### 12.11 Vigilar un fichero de registro en tiempo real

CMD no dispone de una espera activa sobre un fichero, por lo que la alternativa nativa es PowerShell, incluso dentro de una consola de CMD.

```cmd
powershell -NoProfile -Command "Get-Content 'C:\Logs\aplicacion.log' -Tail 20 -Wait"
```

```powershell
Get-Content -Path C:\Logs\aplicacion.log -Tail 20 -Wait |
    Where-Object { $_ -match 'ERROR|CRITICAL' }
```

### 12.12 Limpiar temporales antiguos

```cmd
del /f /q %TEMP%\*.* 2>nul
for /d %d in ("%TEMP%\*") do rd /s /q "%d"
```

```powershell
$limite = (Get-Date).AddDays(-7)
$elementos = Get-ChildItem $env:TEMP -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt $limite }
$elementos | Measure-Object -Property Length -Sum |
    Select-Object Count, @{n='LiberadoMB';e={[math]::Round($_.Sum/1MB,2)}}
$elementos | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
```

La limpieza de temporales debe excluir siempre los ficheros en uso, por eso se aplica -ErrorAction SilentlyContinue y se comprueba primero el tamano que se liberaria antes de borrar nada.

### 12.13 Verificar la integridad de una descarga

```cmd
certutil -hashfile instalador.exe SHA256
fc hash_oficial.txt hash_obtenido.txt
```

```powershell
Get-FileHash .\instalador.exe -Algorithm SHA256 |
    Select-Object Algorithm, Hash, Path
$esperado = 'A1B2C3D4E5F6...'
$obtenido = (Get-FileHash .\instalador.exe -Algorithm SHA256).Hash
if ($obtenido -eq $esperado) { 'Hash correcto' } else { 'El fichero no coincide' }
```

### 12.14 Generar un informe de equipo

```cmd
systeminfo > informe.txt
ipconfig /all >> informe.txt
tasklist >> informe.txt
```

```powershell
$informe = [PSCustomObject]@{
    Fecha          = Get-Date -Format 'yyyy-MM-dd HH:mm'
    Equipo         = $env:COMPUTERNAME
    Sistema        = (Get-CimInstance Win32_OperatingSystem).Caption
    Version        = (Get-CimInstance Win32_OperatingSystem).Version
    Arranque       = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
    CPU            = (Get-CimInstance Win32_Processor).Name
    Nucleos        = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
    ServiciosStop  = (Get-Service | Where-Object Status -eq 'Stopped').Count
    Procesos       = (Get-Process).Count
}
$informe | Format-List
$informe | ConvertTo-Html -Title 'Informe de equipo' | Out-File .\informe.html
```

### 12.15 Comprobar un servicio web

```cmd
curl -s -o nul -w "%{http_code}" https://www.microsoft.com
powershell -NoProfile -Command "(Invoke-WebRequest -Uri 'https://www.microsoft.com' -UseBasicParsing).StatusCode"
```

```powershell
try {
    $r = Invoke-WebRequest -Uri 'https://www.microsoft.com' -UseBasicParsing -TimeoutSec 10
    [PSCustomObject]@{ Url = 'https://www.microsoft.com'; Estado = $r.StatusCode; TiempoMs = $r.Headers['x-ms-request-id'] }
} catch {
    Write-Warning "Fallo la peticion: $_"
}
Test-NetConnection www.microsoft.com -Port 443 -InformationLevel Quiet
```

### 12.16 Crear una tarea programada y monitorizar su resultado

```cmd
schtasks /create /tn "Informe semanal" /tr "powershell -NoProfile -File C:\Scripts\informe.ps1" /sc weekly /d MON /st 07:30 /ru SYSTEM
schtasks /query /tn "Informe semanal" /fo LIST /v
```

```powershell
$accion = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -File "C:\Scripts\informe.ps1"'
$disparador = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 07:30
Register-ScheduledTask -TaskName 'Informe semanal' -Action $accion -Trigger $disparador -User 'SYSTEM' -RunLevel Highest
Get-ScheduledTask -TaskName 'Informe semanal' | Get-ScheduledTaskInfo
```

### 12.17 Exportar errores de sucesos a CSV

```powershell
Get-WinEvent -FilterHashtable @{ LogName = 'Application'; Level = 2; StartTime = (Get-Date).AddDays(-1) } -MaxEvents 500 |
    Select-Object TimeCreated, ProviderName, Id,
        @{n='Mensaje';e={($_.Message -replace '\s+', ' ').Substring(0, [Math]::Min(200, $_.Message.Length))}} |
    Export-Csv .\errores_aplicacion.csv -NoTypeInformation -Encoding UTF8
```

## Capitulo 13: Equivalencias entre CMD y PowerShell

La siguiente tabla resume las operaciones mas frecuentes y su traduccion de un entorno a otro. Los aliases de la columna de PowerShell muestran que muchos nombres de CMD se conservan como nombres cortos, lo que facilita la transicion.

| Operacion | CMD | PowerShell |
|---|---|---|
| Listar contenido | dir, ls | Get-ChildItem (dir, ls, gci) |
| Cambiar de directorio | cd, chdir | Set-Location (cd, sl, chdir) |
| Crear directorio | md, mkdir | New-Item -ItemType Directory (md) |
| Eliminar ficheros | del, erase | Remove-Item (del, rm, rmdir) |
| Renombrar | ren, rename | Rename-Item (ren, rename) |
| Copiar | copy, xcopy | Copy-Item (copy, cp) |
| Mover | move | Move-Item (move, mv) |
| Leer fichero | type, more | Get-Content (type, cat, gc) |
| Escribir fichero | echo texto > f.txt | Set-Content, Out-File |
| Vaciar fichero | type nul > f.txt | Clear-Content, New-Item -Force |
| Buscar texto | find, findstr | Select-String (sls) |
| Ordenar | sort | Sort-Object (sort) |
| Filtrar | findstr patron | Where-Object (where, ?) |
| Seleccionar campos | no equivalente directo | Select-Object (select) |
| Contar | find /c /v "" | Measure-Object (measure) |
| Limpiar pantalla | cls | Clear-Host (cls, clear) |
| Echo de salida | echo | Write-Output (echo) |
| Variables de entorno | set, %VAR% | $env:VAR, Get-ChildItem Env: |
| Crear variable | set VAR=valor | $var = valor |
| Esperar | pause, timeout | Read-Host, Start-Sleep |
| Procesos | tasklist | Get-Process (ps) |
| Terminar proceso | taskkill | Stop-Process (kill) |
| Servicios | net start, sc | Get-Service, Start-Service |
| Lanzar programa | start | Start-Process |
| Tareas programadas | schtasks | Get-ScheduledTask, Register-ScheduledTask |
| Registro | reg | Proveedor Registry (HKLM:\) |
| Red | ipconfig, ping | Get-NetIPAddress, Test-NetConnection |
| Comprobacion de integridad | certutil -hashfile | Get-FileHash |
| Codigos hash de ficheros | certutil | Get-FileHash -Algorithm |
| Ejecutar PowerShell desde CMD | powershell -Command | ya en PowerShell |
| Ejecutar CMD desde PowerShell | no aplica | cmd /c comando, & cmd.exe |

Tambien conviene recordar las diferencias de sintaxis entre ambos entornos. CMD usa barras diagonales para parametros y signos de porcentaje para variables, mientras que PowerShell usa guiones para parametros y signos de dolar para variables. CMD trabaja con texto y PowerShell con objetos. CMD evalua las variables al analizar el bloque completo y PowerShell las evalua al ejecutar la sentencia, por lo que no existe en PowerShell el equivalente a la expansion diferida de setlocal EnableDelayedExpansion.

En cuanto a la interopabilidad, desde CMD se invoca PowerShell con powershell -Command o powershell -File, y desde PowerShell se invoca CMD con cmd /c o simplemente escribiendo el comando externo, siempre comprobando $LASTEXITCODE despues. Para scripts de migracion, lo habitual es reescribir la logica en PowerShell y conservar las llamadas puntuales a herramientas externas de CMD que no tengan equivalente.

## Capitulo 14: Buenas practicas y seguridad

### 14.1 Principios generales

Trabaja siempre con rutas entre comillas cuando puedan contener espacios, tanto en CMD como en PowerShell. En CMD, la asignacion de variables debe escribirse con la variable y el valor entre comillas dobles para evitar espacios accidentales. En PowerShell, prefiere rutas construidas con Join-Path y comprobaciones con Test-Path antes de operar sobre ficheros.

No analices salidas de texto cuando puedas trabajar con objetos. Si CMD es el unico camino, minimo filtra con findstr o find y evita depender de posiciones fijas de columnas, que cambian con el idioma y la version de Windows. En PowerShell, explora primero con Get-Member para conocer las propiedades disponibles y selecciona solo las necesarias con Select-Object.

Especifica el comportamiento ante errores de forma explicita. En CMD, comprueba %ERRORLEVEL% tras cada operacion critica y registra los fallos. En PowerShell, usa -ErrorAction donde corresponda, envuelve las operaciones criticas en try y catch y comprueba $LASTEXITCODE tras los programas externos.

Manten los scripts legibles: usa nombres descriptivos, separa la logica en funciones o subrutinas, documenta los parametros y guarda el registro de las operaciones en ficheros de log con fecha y hora. La transcripcion de sesiones con Start-Transcript es recomendable en intervenciones de administracion sensibles.

### 14.2 Seguridad y privilegios

Eleva la sesion unicamente cuando la tarea lo exija y cierra la consola elevada al terminar. Revisa siempre el contenido de cualquier script antes de ejecutarlo, especialmente si se ha descargado de internet, y desbloquealo con Unblock-File si procede. La politica de ejecucion RemoteSigned para el ambito del usuario combina comodidad y control sin requerir privilegios de administrador.

Evita Invoke-Expression e iex con datos procedentes de fuentes no confiables, porque evaluan cadenas como codigo y convierten un dato manipulable en una ejecucion arbitraria. Del mismo modo, no construyas comandos concatenando entradas de usuario sin validar: usa parametros tipados y comprobaciones ValidateSet o ValidatePattern.

No guardes contrasenas en claro en scripts ni en tareas programadas. Usa Read-Host -AsSecureString, Get-Credential o cuentas de servicio con las credenciales almacenadas por Windows. En CMD, el parametro /rp de schtasks guarda la contrasena en el registro de tareas y debe evitarse en entornos con requisitos de seguridad.

En remoting, declara hosts de confianza de forma concreta y evita el comodin asterisco salvo en laboratorios. Utiliza cuentas dedicadas con privilegios minimos, limita los puertos de WinRM y revisa periodicamente las sesiones y las directivas con Get-PSSessionConfiguration y winrm get.

Para verificar la integridad de ficheros descargados, compara el codigo hash publicado por el fabricante con certutil en CMD o con Get-FileHash en PowerShell. Para scripts de organizacion, considera la politica AllSigned y la firma con Set-AuthenticodeSignature, que garantiza que el contenido no ha sido alterado desde su firma.

```powershell
Set-AuthenticodeSignature -FilePath .\script.ps1 -Certificate (Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert)
```

Habilita la auditoria de cambios en las zonas criticas: exporta antes las claves de registro que vayas a modificar, guarda copia de los ficheros de configuracion, registra las acciones en un log central y utiliza Compare-Object o fc para documentar las diferencias entre el estado anterior y el nuevo.

### 14.3 Codificacion y compatibilidad

Escribe y guarda los ficheros de PowerShell en UTF-8. En Windows PowerShell 5.1, si el script contiene acentos y la codificacion es incorrecta, las cadenas se leeran mal; la solucion habitual es guardar el fichero con BOM o evitar los caracteres especiales en las cadenas criticas. En PowerShell 7, UTF-8 sin BOM es el comportamiento predeterminado.

En scripts por lotes, declara chcp 65001 si la salida debe ser UTF-8 y guarda el fichero sin marca de orden de bytes. Ten en cuenta que las redirecciones de cmd.exe heredan la codificacion de la consola, no la del sistema de ficheros.

Cuando un script deba funcionar en Windows PowerShell 5.1 y en PowerShell 7, evita los operadores exclusivos de la version 7 como && y ||, el operador ternario y ??, comprueba $PSVersionTable al inicio y evita cmdlets retirados como Get-WmiObject. Prueba el script en ambas versiones antes de desplegarlo.

Los nombres localizados cambian segun el idioma de Windows: los grupos como Administrators, los servicios y las rutas de tareas del Programador pueden aparecer traducidos. Para automatizacion multiidioma, identifica los objetos por sus propiedades estables o por identificadores SID, y comprueba siempre el nombre real con Get-LocalGroup o net localgroup antes de referirte a el en un script.

### 14.4 Errores frecuentes

En CMD, el error mas clasico es ejecutar un comando del directorio actual sin querer, porque cmd.exe busca primero ahi; en PowerShell, el error equivalente es olvidar el prefijo .\ al lanzar un script del directorio actual. Otro error habitual es no comprobar %ERRORLEVEL%, confiar en %DATE% con formato regional o expandir variables dentro de bloques parentizados sin activar la expansion diferida.

En PowerShell, los errores mas frecuentes son asumir que los programas externos generan excepciones, usar el alias curl sin el sufijo .exe en Windows PowerShell 5.1, escribir rutas con corchetes en Get-ChildItem sin -LiteralPath porque los corchetes se interpretan como comodines, y olvidar que -eq distingue o no mayusculas segun el operador elegido: -eq ignora mayusculas y -ceq las respeta.

El uso de -LiteralPath en lugar de -Path evita que los caracteres comodines de la ruta, como corchetes o llaves, se interpreten erroneamente. Es una practica recomendable en cualquier script que procese ficheros cuyo nombre puede contener caracteres especiales.

```powershell
Get-Content -LiteralPath 'C:\Datos\informe [final].txt'
Remove-Item -LiteralPath 'C:\Temp\registro {viejo}.log' -Force
```

Por ultimo, no ejecutes en produccion scripts sin probar en un entorno de control: valida los parametros, simula los cambios con -WhatIf cuando el cmdlet lo admita, revisa la salida con -Verbose y documenta el resultado esperado. La combinacion de practicas de ingenieria con el dominio de las dos herramientas de linea de comandos de Windows reduce los incidentes y convierte la automatizacion en una ventaja fiable.

## Capitulo 15: Formatos de entrenamiento con tokens especiales

### 15.1 Objetivo de esta parte

Los capitulos anteriores aportan el conocimiento de PowerShell y de CMD en texto plano, que el convertidor de markdown envuelve como muestra de problema y respuesta. Esta segunda parte del libro aporta ademas muestras de entrenamiento completas con los tokens especiales que define el modelo MyIAModelChat, de modo que el modelo tambien aprenda el flujo agentic: recibir una peticion del usuario, invocar la herramienta shell con un comando, leer el resultado y responder con razonamiento incluido.

Cada bloque de texto de los capitulos 16 a 19 es una muestra de entrenamiento completa y el convertidor la detecta como ya formateada, de modo que la copia sin envolverla y cada bloque se convierte en un fichero de entrenamiento independiente. Las muestras se escriben con el token de idioma al inicio, seguido de los turnos de usuario y de asistente cerrados con el token de fin de turno.

### 15.2 Formato agentic con la herramienta shell

El formato agentic encadena cuatro partes: el turno de usuario con la peticion, un turno de asistente que invoca la herramienta shell con el comando entre parentesis envuelto en los delimitadores de llamada de herramienta del modelo, el resultado de la herramienta con el prefijo de resultado, y un turno final de asistente que contiene el razonamiento y la respuesta al usuario. La herramienta shell recibe el comando como unico parametro posicional y en Windows el executor lo ejecuta con powershell.exe, por lo que los comandos de CMD se envuelven con cmd /c entre comillas para que lleguen enteros al interprete de CMD. Las salidas simuladas de las herramientas imitan la consola real y no superan los quinientos caracteres que el executor admite antes de truncar el resultado.

```text
<|es|><|user|>Cual es la politica de ejecucion de PowerShell?<|end|><|assistant|>``shell(Get-ExecutionPolicy)``<|tool_result|>RemoteSigned<|end|><|assistant|><|thinking|>La politica de ejecucion es una configuracion del equipo. Uso el cmdlet Get-ExecutionPolicy, que devuelve el valor activo de la sesion.<|final|>La politica activa es RemoteSigned: los scripts locales se ejecutan sin restricciones y los descargados de internet necesitan firma o desbloqueo.<|end|>
```

### 15.3 Formato de chat sin herramientas

El formato de chat es el mas simple: un turno de usuario y un turno de asistente con la respuesta directa. Se usa para preguntas de conocimiento que no requieren ejecutar nada en el equipo, como explicar que hace un comando o comparar dos herramientas.

```text
<|es|><|user|>Que diferencia hay entre CMD y PowerShell?<|end|><|assistant|>CMD es el interprete clasico de Windows y trabaja con texto plano: sus comandos devuelven cadenas que hay que analizar con findstr u otras utilidades. PowerShell trabaja con objetos de .NET que se filtran y ordenan por propiedades, admite scripts con manejo de errores y modulos, y es la herramienta recomendada por Microsoft para la automatizacion de la administracion.<|end|>
```

### 15.4 Formato de razonamiento

El formato de razonamiento presenta la necesidad como problema, desarrolla el pensamiento paso a paso entre el token de razonamiento y el token final, y cierra con la respuesta. Es el formato que entrena la cadena de pensamiento de la que el modelo dispone tambien en la inferencia.

```text
<|es|><|problem|>Necesito ver los procesos que mas memoria ocupan en el equipo.<|thinking|>La peticion se resuelve con un cmdlet de PowerShell. Get-Process devuelve objetos con la propiedad WS, que contiene la memoria de trabajo en bytes; ordenando de forma descendente y tomando los primeros obtengo los procesos mas pesados.<|final|>Ejecuta: Get-Process | Sort-Object WS -Descending | Select-Object -First 10 Name, WS. Para ver megabytes, anade una propiedad calculada que divida WS entre 1048576.
```

### 15.5 Convenciones de las muestras

Todas las muestras de los capitulos siguientes siguen cuatro convenciones. Primero, van en espanol y con el token de idioma espanol al principio, ya que este libro se integra como fuente markdown en espanol. Segundo, los comandos estan escritos para ejecutarse con Windows PowerShell 5.1 a traves de la herramienta shell, y las ordenes de CMD van siempre dentro de cmd /c con comillas. Tercero, las salidas de las herramientas reproducen el formato de la consola real de Windows, con las columnas de las tablas de PowerShell alineadas. Cuarto, las muestras se mantienen compactas para que quepan en la ventana de contexto del modelo durante el entrenamiento.

## Capitulo 16: Muestras agentic de PowerShell

Los bloques siguientes son muestras completas del formato agentic en las que la herramienta shell ejecuta cmdlets y programas de PowerShell. Agrupadas por area, cubren servicios, procesos, discos, ficheros, red, registro, tareas programadas y consultas de sistema.

### 16.1 Servicios y procesos

```text
<|es|><|user|>Que servicios de Windows estan detenidos?<|end|><|assistant|>``shell(Get-Service | Where-Object Status -eq 'Stopped' | Select-Object -First 6 Name, Status)``<|tool_result|>Name          Status
----          ------
wuauserv      Stopped
Themes        Stopped
LanmanServer  Stopped
RasMan        Stopped<|end|><|assistant|><|thinking|>Para listar servicios detenidos uso Get-Service y filtro con Where-Object sobre la propiedad Status. Selecciono Name y Status y limito la salida a seis filas para no saturar el resultado.<|final|>Hay varios servicios detenidos, entre ellos wuauserv de Windows Update, Themes y LanmanServer. Si necesitas poner alguno en marcha, usa Start-Service -Name nombre.<|end|>
```

```text
<|es|><|user|>Cuantos procesos hay en ejecucion?<|end|><|assistant|>``shell((Get-Process).Count)``<|tool_result|>147<|end|><|assistant|><|thinking|>Get-Process devuelve la lista de procesos y Count da el total. Es la forma mas directa de obtener el numero sin analizar texto.<|final|>En este momento hay 147 procesos en ejecucion. Puedes verlos ordenados con Get-Process | Sort-Object WS -Descending si quieres revisarlos por memoria.<|end|>
```

```text
<|es|><|user|>Cuales son los cuatro procesos que mas memoria ocupan?<|end|><|assistant|>``shell(Get-Process | Sort-Object WS -Descending | Select-Object -First 4 Name, WS)``<|tool_result|>Name               WS
----               --
firefox            812345344
chrome             624836608
Code               412345344
explorer           189454336<|end|><|assistant|><|thinking|>Ordeno con Sort-Object sobre la propiedad WS de forma descendente y tomo los cuatro primeros con Select-Object -First. WS esta expresada en bytes.<|final|>Los mas pesados son firefox con unos 775 MB, chrome con 596 MB, Code con 393 MB y explorer con 181 MB. Divide cada valor entre 1048576 para ver los megabytes exactos.<|end|>
```

```text
<|es|><|user|>Reinicia el servicio de impresion y comprueba que vuelve a estar en marcha.<|end|><|assistant|>``shell(Restart-Service -Name Spooler -Force; (Get-Service Spooler).Status)``<|tool_result|>Running<|end|><|assistant|><|thinking|>Reinicio el servicio con Restart-Service y en la misma linea consulto su estado con Get-Service para confirmar que ha vuelto a ejecutarse.<|final|>El servicio Spooler se ha reiniciado correctamente y su estado es Running, es decir, en ejecucion.<|end|>
```

```text
<|es|><|user|>Termina la calculadora si esta abierta.<|end|><|assistant|>``shell(Get-Process calc -ErrorAction SilentlyContinue | Stop-Process -Force; 'calculadora finalizada')``<|tool_result|>calculadora finalizada<|end|><|assistant|><|thinking|>Get-Process con SilentlyContinue evita el error si la calculadora no esta abierta, la tuberia la pasa a Stop-Process y el ultimo valor deja un mensaje de confirmacion.<|final|>He finalizado el proceso de la calculadora; el comando no falla aunque no estuviera abierta.<|end|>
```

```text
<|es|><|user|>Cuando se arranco por ultima vez este equipo?<|end|><|assistant|>``shell((Get-CimInstance Win32_OperatingSystem).LastBootUpTime)``<|tool_result|>29/10/2026 08:12:44<|end|><|assistant|><|thinking|>La fecha de ultimo arranque esta en la clase Win32_OperatingSystem. Uso Get-CimInstance y accedo a LastBootUpTime directamente sobre el objeto.<|final|>El equipo se arranco por ultima vez el 29 de octubre de 2026 a las 08:12. Si quieres el tiempo encendido, resta esa fecha a Get-Date.<|end|>
```

```text
<|es|><|user|>Tengo servicios automaticos caidos?<|end|><|assistant|>``shell(Get-Service | Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -ne 'Running' } | Select-Object Name, Status)``<|tool_result|>Name     Status
----     ------
Bonjour  Stopped
gupdate  Stopped<|end|><|assistant|><|thinking|>Filtro dos condiciones a la vez: StartType Automatic y Status distinto de Running. Uso scriptblock con $_ porque combino dos propiedades con -and.<|final|>Si, tienes dos servicios automaticos caidos: Bonjour y gupdate. Puedes relanzarlos con Start-Service -Name nombre.<|end|>
```

```text
<|es|><|user|>Deja el servicio de impresion con inicio automatico y confirmalo.<|end|><|assistant|>``shell(Set-Service -Name Spooler -StartupType Automatic; (Get-Service Spooler).StartType)``<|tool_result|>Automatic<|end|><|assistant|><|thinking|>Set-Service modifica el tipo de inicio y en la misma linea releo la propiedad StartType para confirmar el cambio.<|final|>El servicio Spooler queda configurado con inicio Automatic; el cambio es inmediato en el registro de servicios.<|end|>
```

### 16.2 Discos y ficheros

```text
<|es|><|user|>Cuanto espacio libre queda en el disco C:?<|end|><|assistant|>``shell(Get-Volume -DriveLetter C | Select-Object DriveLetter, FileSystemLabel, @{n='LibreGB';e={[math]::Round($_.SizeRemaining/1GB,1)}})``<|tool_result|>DriveLetter FileSystemLabel LibreGB
----------- --------------- -------
C           Sistema         446.9<|end|><|assistant|><|thinking|>Get-Volume da el volumen de la unidad C y calculo el espacio libre en gigabytes con una propiedad calculada sobre SizeRemaining.<|final|>En la unidad C: quedan 446,9 GB libres. Para revisar todos los discos usa Get-Volume | Where-Object DriveType -eq 'Fixed'.<|end|>
```

```text
<|es|><|user|>Crea la carpeta C:\Respaldos\2024 si no existe y muestra su ruta.<|end|><|assistant|>``shell(New-Item -ItemType Directory -Path C:\Respaldos\2024 -Force | Select-Object -ExpandProperty FullName)``<|tool_result|>C:\Respaldos\2024<|end|><|assistant|><|thinking|>New-Item con -Force crea la carpeta aunque ya exista y tambien las intermedias; ExpandProperty devuelve solo la ruta.<|final|>Carpeta creada o verificada en C:\Respaldos\2024; con -Force el comando es seguro aunque el directorio ya existiera.<|end|>
```

```text
<|es|><|user|>Copia el informe a la carpeta de respaldos y dime su tamano.<|end|><|assistant|>``shell(Copy-Item C:\Datos\informe.txt C:\Respaldos\informe.txt -Force; (Get-Item C:\Respaldos\informe.txt).Length)``<|tool_result|>24517<|end|><|assistant|><|thinking|>Copy-Item realiza la copia con -Force para sobrescribir y despues mido el fichero con Get-Item y la propiedad Length en bytes.<|final|>El fichero se ha copiado a C:\Respaldos\informe.txt y pesa 24517 bytes, unos 24 KB.<|end|>
```

```text
<|es|><|user|>Que ficheros de log de mas de 1 MB hay en C:\Windows\Logs?<|end|><|assistant|>``shell(Get-ChildItem C:\Windows\Logs -Recurse -Filter *.log -ErrorAction SilentlyContinue | Where-Object Length -gt 1MB | Sort-Object Length -Descending | Select-Object -First 5 FullName, Length)``<|tool_result|>FullName                         Length
--------                         ------
C:\Windows\Logs\CBS\CBS.log      5242880
C:\Windows\Logs\DISM\dism.log    2097152<|end|><|assistant|><|thinking|>Recorro la carpeta en subcarpetas con -Recurse, filtro por extension log, selecciono los mayores de 1 MB con Where-Object y los ordeno por tamano descendente.<|final|>Los mas grandes son CBS.log con 5 MB y dism.log con 2 MB. Puedes vaciarlos con Clear-Content si necesitas liberar espacio.<|end|>
```

```text
<|es|><|user|>Cual es el hash SHA256 del instalador?<|end|><|assistant|>``shell(Get-FileHash C:\Descargas\instalador.exe -Algorithm SHA256 | Select-Object -ExpandProperty Hash)``<|tool_result|>A1B2C3D4E5F60718293A4B5C6D7E8F90123456789ABCDEF0123456789ABCDEF0<|end|><|assistant|><|thinking|>Get-FileHash calcula el resumen criptografico del fichero y con ExpandProperty obtengo solo la cadena del hash sin la tabla.<|final|>El hash SHA256 del instalador es A1B2C3D4E5F60718293A4B5C6D7E8F90123456789ABCDEF0123456789ABCDEF0. Comprueba siempre la descarga contra el valor publicado por el fabricante.<|end|>
```

```text
<|es|><|user|>Muestrame las tres ultimas lineas del log de la aplicacion.<|end|><|assistant|>``shell(Get-Content C:\Logs\aplicacion.log -Tail 3)``<|tool_result|>2026-10-29 09:01:12 INFO servicio iniciado
2026-10-29 09:03:45 WARN reintento de conexion
2026-10-29 09:04:02 ERROR fallo al leer la cola<|end|><|assistant|><|thinking|>Get-Content con -Tail lee solo las ultimas lineas sin cargar el fichero completo, que es lo mas eficiente para registros grandes.<|final|>Las tres ultimas lineas muestran el inicio del servicio, un aviso de reintento y un error al leer la cola a las 09:04.<|end|>
```

```text
<|es|><|user|>Cuantos ficheros hay en mi carpeta de descargas?<|end|><|assistant|>``shell((Get-ChildItem $env:USERPROFILE\Descargas -File -ErrorAction SilentlyContinue).Count)``<|tool_result|>47<|end|><|assistant|><|thinking|>Uso la variable de entorno USERPROFILE para no depender del nombre del usuario y -File para contar solo ficheros, no carpetas.<|final|>Hay 47 ficheros en la carpeta Descargas. Si quieres el desglose por tipo, agrupa con Group-Object sobre la propiedad Extension.<|end|>
```

```text
<|es|><|user|>Exporta los servicios a un CSV y dime cuantos registros tiene.<|end|><|assistant|>``shell(Get-Service | Export-Csv C:\Temp\servicios.csv -NoTypeInformation -Encoding UTF8; (Import-Csv C:\Temp\servicios.csv).Count)``<|tool_result|>203<|end|><|assistant|><|thinking|>Export-Csv guarda la tabla con -NoTypeInformation para evitar la cabecera inicial y luego vuelvo a leerlo para validar cuantos registros se escribieron.<|final|>Se han exportado 203 servicios a C:\Temp\servicios.csv; puedes abrirlo en Excel o filtrarlo con Import-Csv.<|end|>
```

### 16.3 Red

```text
<|es|><|user|>Responde el puerto 443 de www.microsoft.com?<|end|><|assistant|>``shell(Test-NetConnection www.microsoft.com -Port 443 -InformationLevel Quiet)``<|tool_result|>True<|end|><|assistant|><|thinking|>Test-NetConnection comprueba la conexion TCP al puerto indicado y con InformationLevel Quiet devuelve un simple True o False.<|final|>Si, el puerto 443 de www.microsoft.com responde correctamente y el servidor web esta accesible.<|end|>
```

```text
<|es|><|user|>Que direcciones IP tiene este equipo?<|end|><|assistant|>``shell(Get-NetIPAddress -AddressFamily IPv4 | Select-Object InterfaceAlias, IPAddress)``<|tool_result|>InterfaceAlias IPAddress
-------------- ---------
Wi-Fi           192.168.1.45
Loopback        127.0.0.1<|end|><|assistant|><|thinking|>Get-NetIPAddress con AddressFamily IPv4 filtra las direcciones version 4 y muestro la interfaz junto a la direccion.<|final|>El equipo tiene 192.168.1.45 en la interfaz Wi-Fi y 127.0.0.1 de loopback; si usas cable aparecera otra interfaz Ethernet.<|end|>
```

```text
<|es|><|user|>Que puertos estan en escucha en este equipo?<|end|><|assistant|>``shell(Get-NetTCPConnection -State Listen | Select-Object -First 4 LocalPort, OwningProcess)``<|tool_result|>LocalPort OwningProcess
--------- -------------
135       1024
445       4
5040      3096<|end|><|assistant|><|thinking|>Get-NetTCPConnection con estado Listen devuelve los puertos abiertos y el PID propietario de cada conexion entrante.<|final|>Estan en escucha los puertos 135 y 445 de servicios de red de Windows y el 5040 de un proceso del sistema. Para ver de que proceso es cada puerto, une OwningProcess con Get-Process -Id.<|end|>
```

```text
<|es|><|user|>Haz un ping a 8.8.8.8 y comprueba que responde.<|end|><|assistant|>``shell(Test-Connection 8.8.8.8 -Count 2 -Quiet)``<|tool_result|>True<|end|><|assistant|><|thinking|>Test-Connection con -Quiet devuelve un valor logico en lugar de texto, lo que evita analizar la salida del ping clasico.<|final|>El servidor 8.8.8.8 responde al equipo: la comprobacion de conexion ha terminado sin errores.<|end|>
```

```text
<|es|><|user|>Vacia la cache de resolucion DNS.<|end|><|assistant|>``shell(ipconfig /flushdns)``<|tool_result|>Configuracion IP de Windows
Cache de resolucion de nombres vaciada correctamente.<|end|><|assistant|><|thinking|>ipconfig es un ejecutable externo que PowerShell invoca directamente y /flushdns limpia las entradas en cache del cliente DNS.<|final|>Cache DNS vaciada; las proximas consultas de nombres volveran a ir al servidor DNS.<|end|>
```

```text
<|es|><|user|>Cual es la IP de www.microsoft.com?<|end|><|assistant|>``shell(Resolve-DnsName www.microsoft.com | Select-Object -First 1 Name, IPAddress)``<|tool_result|>Name                IPAddress
----                ---------
www.microsoft.com   20.205.243.166<|end|><|assistant|><|thinking|>Resolve-DnsName consulta el servidor DNS y con Select-Object tomo la primera respuesta del dominio.<|final|>www.microsoft.com resuelve en 20.205.243.166; quita el First si quieres ver todas las respuestas.<|end|>
```

```text
<|es|><|user|>A que red esta conectado el equipo?<|end|><|assistant|>``shell(Get-NetConnectionProfile | Select-Object Name, InterfaceAlias, IPv4Connectivity)``<|tool_result|>Name           InterfaceAlias IPv4Connectivity
----           -------------- ----------------
Red domestica  Wi-Fi           Internet<|end|><|assistant|><|thinking|>Get-NetConnectionProfile da el perfil de red activo y su conectividad IPv4, suficiente para saber si hay salida a internet.<|final|>El equipo esta conectado a la red Red domestica por Wi-Fi con conectividad IPv4 a Internet.<|end|>
```

### 16.4 Sistema, registro y tareas

```text
<|es|><|user|>Que version de Windows tiene este equipo?<|end|><|assistant|>``shell(Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber)``<|tool_result|>Caption                   Version      BuildNumber
-------                   -------      -----------
Microsoft Windows 10 Home 10.0.19045   19045<|end|><|assistant|><|thinking|>Win32_OperatingSystem contiene la edicion y la version interna; Caption es el nombre comercial que muestra Windows.<|final|>El equipo usa Microsoft Windows 10 Home, version 10.0.19045 con compilacion 19045.<|end|>
```

```text
<|es|><|user|>Que programas arrancan con Windows?<|end|><|assistant|>``shell(Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run | Format-List)``<|tool_result|>SecurityHealth : %windir%\system32\SecurityHealthSystray.exe
OneDrive       : C:\Users\eduardo\AppData\Local\Microsoft\OneDrive\OneDrive.exe /background<|end|><|assistant|><|thinking|>La clave Run de HKLM contiene los programas de inicio de todos los usuarios y Format-List muestra cada valor con su nombre y comando.<|final|>Arrancan con Windows SecurityHealth del centro de seguridad y OneDrive. Puedes quitar cualquiera de ellos borrando su valor de esa clave del registro.<|end|>
```

```text
<|es|><|user|>Crea una tarea programada que ejecute el respaldo cada dia a las 8.<|end|><|assistant|>``shell(Register-ScheduledTask -TaskName 'Respaldo diario' -Action (New-ScheduledTaskAction -Execute powershell.exe -Argument '-File C:\Scripts\respaldo.ps1') -Trigger (New-ScheduledTaskTrigger -Daily -At 08:00) | Select-Object TaskName, State)``<|tool_result|>TaskName         State
-------          -----
Respaldo diario  Ready<|end|><|assistant|><|thinking|>Compongo la tarea con una accion de PowerShell y un disparador diario, la registro con Register-ScheduledTask y el estado Ready confirma que quedo activa.<|final|>Tarea Respaldo diario creada y en estado Ready, se ejecutara todos los dias a las 08:00. Puedes revisarla con Get-ScheduledTask -TaskName 'Respaldo diario'.<|end|>
```

```text
<|es|><|user|>Que usuarios locales tiene el equipo y cuales estan habilitados?<|end|><|assistant|>``shell(Get-LocalUser | Select-Object Name, Enabled, LastLogon)``<|tool_result|>Name      Enabled LastLogon
----      ------- ---------
eduardo   True    29/10/2026 08:15:02
invitado  False<|end|><|assistant|><|thinking|>Get-LocalUser lista las cuentas locales con su estado y ultimo inicio de sesion, suficiente para revisar cuentas sin abrir graficos.<|final|>El equipo tiene la cuenta eduardo habilitada con ultimo inicio hoy y la cuenta invitado esta deshabilitada.<|end|>
```

```text
<|es|><|user|>Hay errores en el registro de sucesos del sistema en las ultimas 24 horas?<|end|><|assistant|>``shell(Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2,3; StartTime=(Get-Date).AddDays(-1)} -MaxEvents 5 | Select-Object TimeCreated, Id, LevelDisplayName)``<|tool_result|>TimeCreated          Id    LevelDisplayName
-----------          --    ----------------
29/10/2026 10:21:03  7036  Error
29/10/2026 09:47:55  10016 Error<|end|><|assistant|><|thinking|>FilterHashtable filtra por registro, nivel y fecha de inicio en una sola consulta; los niveles 1 a 3 son critico, error y advertencia.<|final|>Si, hay eventos de error recientes: uno del servicio a las 10:21 y otro de permisos de activacion a las 09:47. Revisa sus mensajes con Format-List Message.<|end|>
```

```text
<|es|><|user|>Cual es la fecha y hora actuales del sistema?<|end|><|assistant|>``shell(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')``<|tool_result|>2026-10-29 11:03:27<|end|><|assistant|><|thinking|>Get-Date devuelve un objeto DateTime y con el formato especifico obtengo una cadena estable y legible.<|final|>La fecha y hora actuales del sistema son 2026-10-29 11:03:27.<|end|>
```

```text
<|es|><|user|>Que modulos de PowerShell hay instalados?<|end|><|assistant|>``shell(Get-Module -ListAvailable | Sort-Object Name | Select-Object -First 5 Name, Version)``<|tool_result|>Name            Version
----            -------
ActiveDirectory 1.0.1.0
Appx            2.0.5.0
Bits            1.0.0.1<|end|><|assistant|><|thinking|>Get-Module -ListAvailable muestra los modulos presentes en PSModulePath aunque no esten cargados; ordeno y limito la salida a cinco filas.<|final|>Hay multitud de modulos instalados, entre ellos ActiveDirectory, Appx y Bits. Carga cualquiera con Import-Module NombreParaUsarlo.<|end|>
```

## Capitulo 17: Muestras agentic de CMD

Los bloques siguientes invocan la herramienta shell con ordenes de CMD envueltas en cmd /c, tal y como debe hacerse para que lleguen al interprete de CMD de Windows sin ser reinterpretadas por PowerShell.

### 17.1 Ficheros y directorios

```text
<|es|><|user|>Lista el contenido de C:\Windows\Temp solo con nombres.<|end|><|assistant|>``shell(cmd /c "dir /b C:\Windows\Temp")``<|tool_result|>diagnostic_1.tmp
~DF1A2B3C4D.tmp<|end|><|assistant|><|thinking|>dir /b escribe solo los nombres sin cabeceras y el envoltorio cmd /c hace que la orden se ejecute en el interprete de CMD.<|final|>El directorio contiene dos ficheros temporales. Usa dir /ah si tambien quieres ver los ocultos.<|end|>
```

```text
<|es|><|user|>Cual es la direccion IPv4 de este equipo?<|end|><|assistant|>``shell(cmd /c "ipconfig | findstr IPv4")``<|tool_result|>   IPv4. . . . . . . . . . . . . : 192.168.1.45<|end|><|assistant|><|thinking|>Filtro la salida de ipconfig con findstr sobre la etiqueta IPv4, que es comun a todas las versiones de Windows.<|final|>La direccion IPv4 del equipo es 192.168.1.45 en la interfaz principal.<|end|>
```

```text
<|es|><|user|>Muestra la estructura de carpetas de C:\Proyecto.<|end|><|assistant|>``shell(cmd /c "tree C:\Proyecto /f")``<|tool_result|>C:\PROYECTO
+---docs
+---src
    +---modulos<|end|><|assistant|><|thinking|>tree dibuja la estructura como arbol de directorios y /f anade la lista de ficheros de cada carpeta.<|final|>C:\Proyecto tiene las carpetas docs y src, y dentro de src la subcarpeta modulos.<|end|>
```

```text
<|es|><|user|>Busca la palabra error en todos los logs de C:\Logs.<|end|><|assistant|>``shell(cmd /c "findstr /s /i /n error C:\Logs\*.log")``<|tool_result|>C:\Logs\app.log:12: [10:32:11] ERROR de conexion
C:\Logs\app.log:87: [10:35:02] error reintento fallido<|end|><|assistant|><|thinking|>findstr con /s recorre las subcarpetas, /i ignora mayusculas y /n numera las lineas, que es la busqueda recursiva clasica de CMD.<|final|>Hay dos coincidencias en app.log: un error de conexion a las 10:32 y un reintento fallido a las 10:35.<|end|>
```

```text
<|es|><|user|>Crea la carpeta C:\Temp\salida y comprueba que existe.<|end|><|assistant|>``shell(cmd /c "mkdir C:\Temp\salida && dir /ad /b C:\Temp")``<|tool_result|>salida<|end|><|assistant|><|thinking|>mkdir crea el directorio y con el operador de encadenamiento solo se lista si la creacion tuvo exito; dir /ad muestra unicamente carpetas.<|final|>Carpeta creada y verificada: C:\Temp\salida aparece en el listado de directorios.<|end|>
```

```text
<|es|><|user|>Copia el informe con xcopy y dime si copio algo.<|end|><|assistant|>``shell(cmd /c "xcopy C:\Datos\informe.txt C:\Respaldos\ /Y")``<|tool_result|>1 fichero(s) copiado(s).<|end|><|assistant|><|thinking|>xcopy copia el fichero con /Y para no preguntar por la sobrescritura y su mensaje final indica el numero de ficheros copiados.<|final|>Se ha copiado 1 fichero a C:\Respaldos sin pedir confirmacion gracias a /Y.<|end|>
```

```text
<|es|><|user|>El informe A es identico al informe B?<|end|><|assistant|>``shell(cmd /c "fc /b C:\Datos\informe_a.txt C:\Datos\informe_b.txt")``<|tool_result|>Comparando los ficheros C:\DATOS\INFORME_A.TXT y C:\DATOS\INFORME_B.TXT
C:\DATOS\INFORME_A.TXT y C:\DATOS\INFORME_B.TXT son diferentes<|end|><|assistant|><|thinking|>fc compara los dos ficheros byte a byte con /b y la ultima linea indica si hubo diferencias o no.<|final|>Los ficheros no son identicos: fc encuentra diferencias entre informe_a.txt e informe_b.txt.<|end|>
```

### 17.2 Red y procesos

```text
<|es|><|user|>Muestra los puertos en escucha con las herramientas de CMD.<|end|><|assistant|>``shell(cmd /c "netstat -ano | findstr LISTENING")``<|tool_result|>TCP    0.0.0.0:135            0.0.0.0:0              LISTENING     1024
TCP    0.0.0.0:445            0.0.0.0:0              LISTENING     4
TCP    [::]:135               [::]:0                 LISTENING     1024<|end|><|assistant|><|thinking|>netstat -ano muestra ademas del estado el identificador de proceso; findstr selecciona unicamente las lineas de escucha.<|final|>Los puertos 135 y 445 estan en escucha con los PID 1024 y 4. Consulta cada proceso con tasklist /fi "PID eq 1024".<|end|>
```

```text
<|es|><|user|>Comprueba con ping si 8.8.8.8 responde.<|end|><|assistant|>``shell(cmd /c "ping -n 2 8.8.8.8")``<|tool_result|>Haciendo ping a 8.8.8.8 con 32 bytes de datos:
Respuesta desde 8.8.8.8: bytes=32 tiempo=14ms TTL=117
Respuesta desde 8.8.8.8: bytes=32 tiempo=13ms TTL=117

Estadisticas de ping para 8.8.8.8:
    Paquetes: enviados = 2, recibidos = 2, perdidos = 0 (0 perdidos),<|end|><|assistant|><|thinking|>ping -n 2 envia dos paquetes y la estadistica final confirma si hay perdidas; la salida en espanol corresponde a Windows en espanol.<|final|>8.8.8.8 responde: dos paquetes recibidos con un tiempo de 13 y 14 ms y cero perdidas.<|end|>
```

```text
<|es|><|user|>Hay procesos de python en marcha?<|end|><|assistant|>``shell(cmd /c "tasklist | findstr python")``<|tool_result|>python.exe                   12484 Console                    1     92.516 KB<|end|><|assistant|><|thinking|>tasklist lista los procesos y findstr filtra por nombre; una unica linea indica un unico proceso en ejecucion.<|final|>Si, hay un proceso python.exe con PID 12484 usando unos 90 MB de memoria.<|end|>
```

```text
<|es|><|user|>Cierra el bloc de notas.<|end|><|assistant|>``shell(cmd /c "taskkill /im notepad.exe /f")``<|tool_result|>Correcto: se termino el proceso "notepad.exe" con PID 11012.<|end|><|assistant|><|thinking|>taskkill /im termina por nombre de imagen y /f fuerza el cierre; el mensaje de exito confirma que el proceso existia y ha terminado.<|final|>He cerrado el bloc de notas: el proceso notepad.exe con PID 11012 ha terminado.<|end|>
```

```text
<|es|><|user|>Cual es el estado del servicio de impresion?<|end|><|assistant|>``shell(cmd /c "sc query spooler")``<|tool_result|>SERVICE_NAME: spooler
TYPE               : 10  WIN32_OWN_PROCESS
STATE              : 4  RUNNING
<|end|><|assistant|><|thinking|>sc query devuelve el tipo y el estado numerico del servicio; el valor 4 significa en ejecucion.<|final|>El servicio spooler esta en ejecucion (estado 4 RUNNING), la impresion funciona con normalidad.<|end|>
```

```text
<|es|><|user|>Que direccion MAC tiene la tarjeta de red?<|end|><|assistant|>``shell(cmd /c "getmac /v | findstr /i Conectado")``<|tool_result|>Conector de red Ethernet 00-1A-2B-3C-4D-5E      Conectado<|end|><|assistant|><|thinking|>getmac /v muestra la direccion fisica de cada adaptador con su estado y findstr se queda con los conectados.<|final|>El adaptador de ethernet conectado tiene la direccion MAC 00-1A-2B-3C-4D-5E.<|end|>
```

```text
<|es|><|user|>Termina el proceso con PID 11012.<|end|><|assistant|>``shell(cmd /c "taskkill /PID 11012 /F")``<|tool_result|>Correcto: se termino el proceso con PID 11012.<|end|><|assistant|><|thinking|>taskkill con /PID apunta a un proceso concreto y /F fuerza el cierre sin esperar a que termine por si mismo.<|final|>Proceso con PID 11012 terminado correctamente.<|end|>
```

```text
<|es|><|user|>Cual es la direccion IP del servidor DNS que usa este equipo?<|end|><|assistant|>``shell(cmd /c "nslookup www.microsoft.com")``<|tool_result|>Servidor:   dns.google
Address:  192.168.1.1

Nombre:    www.microsoft.com
Address:  20.205.243.166<|end|><|assistant|><|thinking|>nslookup muestra primero el servidor DNS que respondio y despues la resolucion del dominio consultado.<|final|>El equipo usa el servidor DNS 192.168.1.1 y www.microsoft.com resuelve en 20.205.243.166.<|end|>
```

### 17.3 Sistema y mantenimiento

```text
<|es|><|user|>Que version exacta de Windows corre aqui?<|end|><|assistant|>``shell(cmd /c "ver")``<|tool_result|>Microsoft Windows [Version 10.0.19045.3693]<|end|><|assistant|><|thinking|>ver devuelve la version del nucleo de Windows tal como la reporta el interprete de CMD.<|final|>El equipo ejecuta Windows 10 con el build 19045.3693.<|end|>
```

```text
<|es|><|user|>Cuanta memoria fisica tiene y cuanta queda libre?<|end|><|assistant|>``shell(cmd /c "systeminfo | findstr /i memoria")``<|tool_result|>Memoria fisica total:      16.280 MB
Memoria fisica disponible:  5.120 MB<|end|><|assistant|><|thinking|>systeminfo tarda bastante, por eso filtro con findstr /i sobre la palabra memoria para traer unicamente las dos lineas que interesan.<|final|>El equipo tiene 16 GB de RAM fisica y unos 5 GB libres en el momento de la consulta.<|end|>
```

```text
<|es|><|user|>Cuantos bytes libres quedan en las unidades?<|end|><|assistant|>``shell(cmd /c "wmic logicaldisk get caption,freespace")``<|tool_result|>Caption  FreeSpace
C:       479800000000
D:       120000000000<|end|><|assistant|><|thinking|>wmic logicaldisk devuelve el espacio libre en bytes de cada unidad en una tabla simple, adecuada para lectura automatica.<|final|>La unidad C: dispone de unos 479 GB libres y la D: de unos 120 GB.<|end|>
```

```text
<|es|><|user|>Cual es el hash SHA256 del instalador con la herramienta de CMD?<|end|><|assistant|>``shell(cmd /c "certutil -hashfile C:\Descargas\instalador.exe SHA256")``<|tool_result|>SHA256 hash of instalador.exe:
a1 b2 c3 d4 e5 f6 07 18 29 3a 4b 5c 6d 7e 8f 90
12 34 56 78 9a bc de f0 12 34 56 78 9a bc de f0
CertUtil: -hashfile command completed successfully.<|end|><|assistant|><|thinking|>certutil -hashfile calcula el resumen del fichero con el algoritmo indicado y devuelve el hash en bytes separados por espacios.<|final|>El hash SHA256 del instalador es a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0. Comparalo con el valor oficial antes de instalar.<|end|>
```

```text
<|es|><|user|>Que programas de inicio tiene el equipo en el registro?<|end|><|assistant|>``shell(cmd /c "reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run")``<|tool_result|>HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
    SecurityHealth    REG_SZ    %windir%\system32\SecurityHealthSystray.exe
    OneDrive          REG_SZ    C:\Users\eduardo\AppData\Local\Microsoft\OneDrive\OneDrive.exe /background<|end|><|assistant|><|thinking|>reg query recupera los valores de la clave Run de la maquina, donde Windows guarda los programas que arrancan con el sistema.<|final|>Arrancan con Windows SecurityHealth y OneDrive. Puedes eliminar un valor con reg delete seguido de la clave, /v nombre y /f.<|end|>
```

```text
<|es|><|user|>Que tareas programadas hay relacionadas con respaldos?<|end|><|assistant|>``shell(cmd /c "schtasks /query /fo csv | findstr /i respaldo")``<|tool_result|>"\Respaldo diario","29/10/2026 08:00:00","Listo"<|end|><|assistant|><|thinking|>schtasks /fo csv saca el listado en formato separado por comas y findstr filtra por el nombre de la tarea.<|final|>La tarea Respaldo diario esta programada para el 29/10/2026 a las 08:00 y figura como Listo.<|end|>
```

```text
<|es|><|user|>Actualiza las directivas de grupo de este equipo.<|end|><|assistant|>``shell(cmd /c "gpupdate /force")``<|tool_result|>Actualizando la directiva...
Se aplico correctamente la directiva de equipo.
Se aplico correctamente la directiva de usuario.<|end|><|assistant|><|thinking|>gpupdate /force vuelve a aplicar las directivas de grupo sin esperar al siguiente inicio de sesion.<|final|>Las directivas de grupo de equipo y de usuario se han vuelto a aplicar correctamente.<|end|>
```

```text
<|es|><|user|>Crea una carpeta temporal y comprueba que el listado la muestra.<|end|><|assistant|>``shell(cmd /c "mkdir C:\Temp\salida2 && dir /ad /b C:\Temp")``<|tool_result|>salida
salida2<|end|><|assistant|><|thinking|>mkdir crea la carpeta y el encadenamiento con doble ampersand ejecuta el listado solo si la creacion termino bien.<|final|>Carpeta C:\Temp\salida2 creada y confirmada en el listado junto a la carpeta salida.<|end|>
```

## Capitulo 18: Muestras de chat sin herramientas

Estas muestras usan el formato de chat: un turno de usuario y un turno de asistente con la respuesta. Sirven para preguntas de conocimiento que no exigen ejecutar nada en el equipo.

### 18.1 Preguntas sobre CMD

```text
<|es|><|user|>Para que sirve el comando dir en CMD?<|end|><|assistant|>dir lista el contenido de un directorio con sus ficheros, carpetas y atributos. Las opciones mas usadas son /s para recorrer subcarpetas, /b para mostrar solo rutas, /ah para ocultos, /o para ordenar y /w en formato ancho.<|end|>
```

```text
<|es|><|user|>Que diferencia hay entre copy y xcopy?<|end|><|assistant|>copy copia ficheros sueltos y tambien puede concatenar varios ficheros en uno. xcopy copia ademas estructuras completas de carpetas y admite opciones como /E para subcarpetas vacias, /H para ocultos y de sistema, /Y para no preguntar y /D solo los mas recientes.<|end|>
```

```text
<|es|><|user|>Como busco un texto dentro de varios ficheros con CMD?<|end|><|assistant|>Con findstr. La forma habitual es findstr /s /i /n "texto" C:\Logs\*.log, donde /s recorre subcarpetas, /i ignora mayusculas y /n numera las lineas encontradas. Admite expresiones regulares con /r y cadenas literales con /c.<|end|>
```

```text
<|es|><|user|>Que es %~dp0 en un script por lotes?<|end|><|assistant|>Es la expansion del directorio donde se encuentra el propio script, incluida la unidad. Permite construir rutas relativas al script con seguridad aunque se lance desde cualquier carpeta, por ejemplo set "BASE=%~dp0" y despues usar "%BASE%config.ini".<|end|>
```

```text
<|es|><|user|>Como copio una carpeta completa con robocopy?<|end|><|assistant|>La orden basica es robocopy origen destino /E, donde /E incluye las subcarpetas aunque esten vacias. Para trabajos serios se anaden /R:2 y /W:5 de reintentos, /MT:8 con varios hilos y /LOG+ para registrar la actividad. Cuidado con /MIR, que crea un espejo y borra del destino los ficheros que ya no esten en el origen.<|end|>
```

```text
<|es|><|user|>Que hace tasklist /svc?<|end|><|assistant|>tasklist lista los procesos en ejecucion y con /svc anade la columna de servicios de Windows asociados a cada proceso. Es util para averiguar que servicio sustenta un proceso svchost.exe, por ejemplo con tasklist /svc /fi "imagename eq svchost.exe".<|end|>
```

```text
<|es|><|user|>Como cambio la letra de una unidad?<|end|><|assistant|>Desde CMD se hace con diskpart: ejecuta diskpart, selecciona el volumen con select volume N, cambia la letra con assign letter=Z y sal con exit. Powershell ofrece los cmdlets Get-Volume y Add-Partition con la misma funcion sobre el proveedor de almacenamiento.<|end|>
```

```text
<|es|><|user|>Que es ERRORLEVEL?<|end|><|assistant|>Es el codigo de salida que deja el ultimo comando ejecutado en CMD. El valor 0 indica exito y los demas algun tipo de fallo, con significado propio de cada comando. Se comprueba con if errorlevel 1 o comparando %ERRORLEVEL% con EQU, GTR o LTR dentro de scripts por lotes.<|end|>
```

```text
<|es|><|user|>Para que sirven assoc y ftype?<|end|><|assistant|>assoc relaciona las extensiones de fichero con un tipo de archivo, por ejemplo .txt=txtfile, y ftype relaciona cada tipo de archivo con el comando que lo abre. Son herramientas de consulta; en Windows moderno la asociacion elegida por el usuario desde Configuracion tiene prioridad sobre estas claves.<|end|>
```

```text
<|es|><|user|>Como programo una tarea con schtasks?<|end|><|assistant|>Con schtasks /create seguido del nombre con /tn, la orden con /tr, la frecuencia con /sc daily u otra y la hora con /st. Por ejemplo schtasks /create /tn "Limpieza" /tr "cleanmgr /sagerun:1" /sc weekly /d SUN /st 09:00. Se consulta con /query, se lanza con /run y se borra con /delete /f.<|end|>
```

### 18.2 Preguntas sobre PowerShell

```text
<|es|><|user|>Que hace el cmdlet Get-ChildItem?<|end|><|assistant|>Get-ChildItem lista el contenido de un directorio o de cualquier proveedor de rutas, como el registro con HKLM:\. Admite -Filter y -Include para patron de nombre, -Recurse para subcarpetas, -Depth para limitar la profundidad, -File y -Directory para filtrar por tipo y -Force para incluir los ocultos. Es el equivalente a dir y sus aliases son dir, ls y gci.<|end|>
```

```text
<|es|><|user|>Como funciona el pipeline de objetos de PowerShell?<|end|><|assistant|>El operador tuberia encadena cmdlets de modo que la salida de uno se convierte en la entrada del siguiente como objetos de .NET, no como texto. Eso permite filtrar con Where-Object, ordenar con Sort-Object y proyectar con Select-Object sin volver a analizar cadenas, que es la principal ventaja sobre CMD.<|end|>
```

```text
<|es|><|user|>Que diferencia hay entre -eq y -ceq?<|end|><|assistant|>Ambos comparan, pero -eq ignora las mayusculas y minusculas mientras que -ceq las respeta. En Windows, donde los nombres de ficheros y servicios pueden variar en capitalizacion, -eq es el comportamiento por defecto y -ceq solo hace falta cuando distinguir mayusculas es importante.<|end|>
```

```text
<|es|><|user|>Que es la politica de ejecucion y cual es la recomendada?<|end|><|assistant|>La politica de ejecucion controla bajo que condiciones PowerShell lanza scripts. Los valores principales son Restricted, RemoteSigned, AllSigned, Unrestricted y Bypass. Microsoft aclara que no es un sistema de seguridad sino una proteccion contra ejecuciones accidentales; la recomendada para el uso diario es RemoteSigned en el ambito CurrentUser, que no requiere privilegios de administrador.<|end|>
```

```text
<|es|><|user|>Por que en PowerShell 5.1 el comando curl no es curl.exe?<|end|><|assistant|>Porque curl y wget son aliases de Invoke-WebRequest en Windows PowerShell 5.1, asi que invocan el cmdlet de PowerShell y no el programa. Para ejecutar el programa incluido en Windows escribe curl.exe con la extension. PowerShell 7 elimino esos aliases conflicto y siempre que se escriba curl llama al ejecutable.<|end|>
```

```text
<|es|><|user|>Que variable contiene el directorio actual en PowerShell?<|end|><|assistant|>$PWD guarda el directorio actual y $HOME la carpeta del perfil del usuario. Las variables de entorno se leen con el prefijo $env:, por ejemplo $env:TEMP o $env:USERPROFILE. En CMD el equivalente es %CD% y en los scripts por lotes el directorio del script se obtiene con %~dp0.<|end|>
```

```text
<|es|><|user|>Como ejecuto un script bloqueado por la politica de ejecucion?<|end|><|assistant|>La solucion limpia es desbloquear el fichero con Unblock-File -Path .\script.ps1 o fijar RemoteSigned para el usuario con Set-ExecutionPolicy. Para un unico lanzamiento puntual tambien puede usarse powershell -ExecutionPolicy Bypass -File .\script.ps1, sin cambiar la configuracion permanente del equipo.<|end|>
```

```text
<|es|><|user|>Que diferencia hay entre Get-Content -Tail y -TotalCount?<|end|><|assistant|>-Tail lee las ultimas N lineas del fichero, ideal para seguir un registro en vivo con la opcion -Wait. -TotalCount lee las primeras N lineas. Ambos son mucho mas eficientes que cargar el fichero completo con -Raw cuando solo hace falta una parte.<|end|>
```

```text
<|es|><|user|>Para que sirve Get-Member?<|end|><|assistant|>Get-Member muestra las propiedades, metodos y eventos del objeto que recibe por el pipeline. Es la herramienta de descubrimiento para saber que datos ofrece un cmdlet cuando no conoces el tipo de objeto que devuelve, por ejemplo Get-Date | Get-Member para ver los metodos AddDays o ToString.<|end|>
```

```text
<|es|><|user|>Que es $LASTEXITCODE y por que importa?<|end|><|assistant|>Es la variable que guarda el codigo de salida del ultimo programa externo ejecutado, como cmd, git o robocopy. PowerShell no convierte esos codigos en excepciones, asi que despues de cada comando nativo hay que comprobar $LASTEXITCODE si se quiere detectar un fallo; $? solo informa del resultado de los cmdlets de PowerShell.<|end|>
```

## Capitulo 19: Muestras de razonamiento

Estas muestras usan el formato de razonamiento: la necesidad del usuario como problema, el pensamiento paso a paso y la respuesta final con el comando recomendado. Ensenan al modelo a elegir entre CMD y PowerShell antes de responder.

### 19.1 Administracion del sistema

```text
<|es|><|problem|>Necesito liberar espacio en disco eliminando los temporales de mas de una semana.<|thinking|>Los temporales estan en la variable de entorno TEMP. El camino seguro es primero medir cuantos ficheros antiguos hay con Where-Object sobre LastWriteTime y despues borrar con Remove-Item; asi se evita eliminar ficheros en uso omitiendo los errores con SilentlyContinue.<|final|>Primero comprueba el volumen: Get-ChildItem $env:TEMP -Recurse -File -Force | Where-Object LastWriteTime -lt (Get-Date).AddDays(-7) | Measure-Object -Property Length -Sum. Cuando confirmes el tamano, anade Remove-Item -Recurse -Force -ErrorAction SilentlyContinue al final de la tuberia.
```

```text
<|es|><|problem|>Quiero hacer un respaldo incremental de C:\Datos a un disco externo.<|thinking|>Para estructuras de carpetas completas, robocopy es la herramienta recomendada porque copia solo los cambios, reintenta ante fallos y deja un registro. /E incluye todas las subcarpetas, /XJ salta los enlaces simbolicos y /R y /W controlan reintentos y espera.<|final|>Ejecuta: robocopy C:\Datos E:\Respaldos /E /XJ /R:2 /W:5 /LOG+:E:\Respaldos\copia.log. Los codigos de salida de 0 a 7 indican exito; 8 o mas significan fallos que debes revisar en el log.
```

```text
<|es|><|problem|>Necesito saber que proceso esta usando el puerto 3389 del servidor de escritorio remoto.<|thinking|>Primero busco la conexion que escucha en ese puerto con Get-NetTCPConnection, que devuelve el PID propietario, y despues resuelvo el PID a nombre de proceso con Get-Process. En CMD el mismo flujo seria netstat -ano y tasklist con filtro de PID.<|final|>Usa: Get-NetTCPConnection -LocalPort 3389 -State Listen | Select-Object OwningProcess | ForEach-Object { Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue } . La alternativa de CMD es netstat -ano | findstr 3389 seguido de tasklist /fi.
```

```text
<|es|><|problem|>Quiero renombrar todas las imagenes de una carpeta anadiendo un prefijo de viaje.<|thinking|>Get-ChildItem selecciona los ficheros por extension y Rename-Item acepta un scriptblock como nombre nuevo, lo que permite construir el prefijo dentro de la propia tuberia sin escribir un bucle. El scriptblock se evalua una vez por fichero con $_ disponible.<|final|>Ejecuta: Get-ChildItem C:\Fotos -Filter *.jpg | Rename-Item -NewName { 'viaje_2024_' + $_.Name }. Primero puedes probar con Select-Object @{n='Nuevo';e={'viaje_2024_' + $_.Name}} para ver el resultado sin renombrar nada.
```

```text
<|es|><|problem|>Necesito comprobar si la base de datos responde por el puerto 1433.<|thinking|>Para un puerto concreto de un host concreto, Test-NetConnection es la herramienta adecuada porque hace la apertura TCP y la resolucion DNS en una sola llamada y puede devolver un simple valor logico con InformationLevel Quiet.<|final|>Usa: Test-NetConnection servidorbd -Port 1433 -InformationLevel Quiet. Si devuelve False, revisa con Test-NetConnection servidorbd la ruta de red o pide el firewall con Get-NetFirewallRule.
```

```text
<|es|><|problem|>Quiero contar cuantos errores ha registrado el sistema en la ultima semana.<|thinking|>Get-WinEvent con FilterHashtable permite filtrar por registro y rango de fechas en una sola consulta, y Measure-Object cuenta los objetos que pasan por el tuberia. Nivel 2 es error y 1 es critico, con lo que la combinacion cubre los fallos reales.<|final|>Ejecuta: Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2; StartTime=(Get-Date).AddDays(-7)} | Measure-Object | Select-Object -ExpandProperty Count. Puedes agrupar por ProviderName con Group-Object para ver que componente falla mas.
```

```text
<|es|><|problem|>Necesito crear un usuario local con contrasena y ponerlo en el grupo de administradores.<|thinking|>New-LocalUser exige una contrasena en forma segura, por lo que hay que construirla con ConvertTo-SecureString o pedirla con Read-Host -AsSecureString. Despues Add-LocalGroupMember completa la alta. El cambio requiere consola elevada.<|final|>Primero: $clave = Read-Host 'Contrasena' -AsSecureString. Luego: New-LocalUser -Name tecnico -Password $clave -FullName 'Tecnico de soporte' y Add-LocalGroupMember -Group Administrators -Member tecnico. Verifica con Get-LocalGroupMember -Group Administrators.
```

```text
<|es|><|problem|>Necesito revisar la integridad de los ficheros protegidos de Windows.<|thinking|>El comando clasico es sfc /scannow, pero si el almacen de componentes esta corrupto, sfc no encontrara copias validas. El orden correcto es primero reparar la tienda con DISM y despues lanzar sfc, y ambas tareas necesitan privilegios de administrador y pueden tardar varios minutos.<|final|>Ejecuta primero DISM /Online /Cleanup-Image /RestoreHealth y cuando termine sfc /scannow. En PowerShell puedes invocarlos como shell(cmd /c "DISM /Online /Cleanup-Image /RestoreHealth") y revisar el resultado en el informe de CBS.log.
```

```text
<|es|><|problem|>Quiero que una tarea se ejecute cada semana con PowerShell en lugar de con schtasks.<|thinking|>El modulo ScheduledTasks expone el Programador de tareas con objetos: una accion, un disparador y una configuracion. Register-ScheduledTask combina las tres piezas y devuelve el estado de la tarea creada, mas legible que la sintaxis larga de schtasks.<|final|>Construye la tarea asi: $accion = New-ScheduledTaskAction -Execute powershell.exe -Argument '-File C:\Scripts\informe.ps1'; $disparador = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 07:30; Register-ScheduledTask -TaskName 'Informe semanal' -Action $accion -Trigger $disparador.
```

### 19.2 Ficheros, respaldo y red

```text
<|es|><|problem|>Quiero exportar los eventos de error de la aplicacion a un CSV para un informe.<|thinking|>Get-WinEvent con FilterHashtable trae los eventos del canal elegido, pero sus propiedades incluyen textos largos, asi que conviene proyectar solo fecha, identificador y un mensaje truncado antes de exportar. Export-Csv con UTF8 asegura que el informe se abre bien en cualquier equipo.<|final|>Usa: Get-WinEvent -FilterHashtable @{LogName='Application'; Level=2; StartTime=(Get-Date).AddDays(-1)} | Select-Object TimeCreated, Id, @{n='Mensaje';e={$_.Message.Substring(0,200)}} | Export-Csv C:\Informes\errores.csv -NoTypeInformation -Encoding UTF8.
```

```text
<|es|><|problem|>Necesito ejecutar un mismo comando en varios equipos del laboratorio.<|thinking|>PowerShell remoting usa WinRM y permite pasar una lista de equipos a Invoke-Command con un bloque de script, mucho mas comodo que abrir sesiones una a una. Antes hay que habilitar el remoting y declarar los equipos de confianza si no hay dominio.<|final|>Habilita con Enable-PSRemoting -Force y ejecuta: Invoke-Command -ComputerName PC1,PC2,PC3 -ScriptBlock { Get-Service | Where-Object Status -eq 'Stopped' | Select-Object PSComputerName, Name }. Recuerda aprobar los hosts con Set-Item WSMan:\localhost\Client\TrustedHosts si trabajas en grupo de trabajo.
```

```text
<|es|><|problem|>Necesito vaciar la cache DNS y renovar la direccion IP de este equipo.<|thinking|>Ambas tareas las cubre ipconfig: /flushdns limpia la cache de nombres y /renew renueva la concesion DHCP. Se pueden encadenar en una sola llamada a traves de cmd /c y el doble ampersand, que solo lanza la segunda si la primera tuvo exito.<|final|>Ejecuta: shell(cmd /c "ipconfig /flushdns && ipconfig /renew"). Comprueba el resultado con ipconfig /all y si la red no vuelve, revisa el adaptador con Get-NetAdapter -Name * | Where-Object Status -eq 'Up'.
```

```text
<|es|><|problem|>Quiero saber si un fichero descargado esta corrompido.<|thinking|>La forma fiable es comparar el hash publicado por el fabricante con el calculado localmente. Tanto Get-FileHash como certutil sirven, y el algoritmo debe coincidir exactamente con el oficial, normalmente SHA256. Cualquier diferencia de un solo byte cambia todo el resumen.<|final|>Calcula: Get-FileHash .\fichero.zip -Algorithm SHA256 y compara la cadena con la del sitio oficial. En CMD: cmd /c "certutil -hashfile fichero.zip SHA256". Si no coinciden, vuelve a descargar el fichero antes de ejecutarlo.
```

```text
<|es|><|problem|>Necesito restringir una carpeta para que solo la use mi usuario.<|thinking|>En NTFS los permisos se gestionan con icacls. Lo correcto es primero romper la herencia con /inheritance:r para que no herede los permisos de la carpeta padre, despues conservar la lectura para el sistema y por ultimo conceder control total al usuario indicado. El cambio puede requerir ser propietario con takeown.<|final|>Usa: icacls C:\Privada /inheritance:r /grant %USERNAME%:F. Si deniega el acceso, ejecuta antes takeown /f C:\Privada y vuelve a intentarlo. Revisa el resultado con icacls C:\Privada.
```

```text
<|es|><|problem|>Necesito ver que carpetas ocupan mas disco en la unidad C:.<|thinking|>No existe un comando unico que sume por carpeta, pero el patron de PowerShell es recorrer las carpetas de primer nivel, medir cada una con Measure-Object y presentar el total ordenado. Para bajar de profundidad se puede repetir con -Recurse, aunque tarda mas.<|final|>Ejecuta: Get-ChildItem C:\ -Directory -ErrorAction SilentlyContinue | ForEach-Object { $m = Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum; [PSCustomObject]@{ Carpeta = $_.Name; GB = [math]::Round($m.Sum/1GB,2) } } | Sort-Object GB -Descending | Select-Object -First 10.
```

```text
<|es|><|problem|>Quiero bloquear la consola para que solo yo pueda elevarla desde el equipo compartido.<|thinking|>La elevacion depende del Control de Cuentas de Usuario y de la pertenencia al grupo de administradores. La medida practica es sacar a los demas usuarios del grupo, mantener su cuenta como estandar y revisar el historial de sucesos de seguridad para detectar intentos de elevation.<|final|>Comprueba el grupo con Get-LocalGroupMember -Group Administradores, retira cuentas con Remove-LocalGroupMember -Group Administrators -Member nombre y revisa los sucesos 4672 y 4688 con Get-WinEvent -FilterHashtable @{LogName='Security'; Id=4672,4688}.
```

## Fuentes de referencia

El contenido de esta guia se basa en la documentacion oficial de Microsoft disponible en Microsoft Learn, en particular los temas de referencia de comandos de Windows Server para cmd.exe y sus utilidades, las guias de scripting de PowerShell sobre about_execution_policies, about_operators, about_automatic_variables, about_scopes, about_wildcards, about_parsing, about_remote y about_cim_cmdlets, la referencia de cmdlets de los modulos Microsoft.PowerShell.Management, Microsoft.PowerShell.Utility, Microsoft.PowerShell.Security, NetTCPIP, ScheduledTasks, Storage y Microsoft.PowerShell.LocalAccounts, y los articulos de Windows Terminal y de politicas de ejecucion de PowerShell. La documentacion de producto de Windows incluida en el propio sistema, accesible con Get-Help y con los temas about_, complementa estas referencias.

