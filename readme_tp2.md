# TP2 - Procesamiento de Imágenes

Este repositorio contiene la resolución del Trabajo Práctico N° 2 de la materia Procesamiento de Imágenes. El proyecto se divide en dos grandes problemas resueltos mediante scripts en Python utilizando la librería OpenCV.

- **Problema 1:** Detección, segmentación y clasificación de monedas y dados, incluyendo el conteo de los valores y los puntos.
- **Problema 2:** Detección automática de patentes vehiculares y la posterior segmentación de sus caracteres.

---

## 📋 Requisitos Previos

Para ejecutar los scripts correctamente, es necesario tener instalado **Python 3.x**. Se recomienda utilizar un entorno virtual para evitar conflictos con otras librerías del sistema.

### Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/santialmironn/procesamiento-imagenes-tp2.git
   cd procesamiento-imagenes-tp2
   ```

2. **Crear y activar el entorno virtual:**
   - En Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - En Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Las dependencias incluyen `opencv-python`, `numpy` y `matplotlib`).*

---

## 📁 Estructura del Directorio

Antes de ejecutar los scripts, asegúrese de que la estructura del proyecto cuente con los archivos de entrada necesarios:

- `monedas.jpg`: Imagen original requerida para el Problema 1.
- `Patentes/`: Carpeta que debe contener las 12 imágenes de patentes (`img01.png` a `img12.png`) requeridas para el Problema 2.
- `Problema1.py` y `Problema2.py`: Scripts principales de ejecución.
- `requirements.txt`: Archivo de configuración de dependencias.
- Informe final en formato PDF.

---

## 🚀 Instrucciones de Ejecución

Ambos scripts están diseñados para ejecutarse de manera independiente. Al correrlos, **generarán automáticamente una carpeta llamada `resultados/`** en el directorio raíz, donde se guardarán todas las imágenes de las etapas intermedias y los resultados finales, tal como se documenta en el informe PDF.

### Ejecutar el Problema 1 (Monedas y Dados)

Desde la terminal, posicionado en la raíz del proyecto, ejecute:

```bash
python Problema1.py
```

**¿Qué hace este script?**
- **A.** Segmenta los objetos de la imagen `monedas.jpg`.
- **B.** Separa los dados de las monedas utilizando descriptores geométricos, y clasifica las monedas por su valor nominal evaluando su tamaño y color.
- **C.** Cuenta los puntos de los dados detectados y calcula el valor total del dinero.

### Ejecutar el Problema 2 (Detección de Patentes)

Desde la terminal, ejecute:

```bash
python Problema2.py
```

**¿Qué hace este script?**
- **A.** Procesa el lote de 12 imágenes de la carpeta `Patentes/`, detectando y recortando la zona correspondiente a la chapa patente.
- **B.** Segmenta cada uno de los caracteres (letras y números) de forma independiente.

---

**Autores del Trabajo Práctico:**
*Justo Borras*
*Santino Almiron Nanni*
*Wenceslao Tejerina*