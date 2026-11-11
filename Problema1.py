import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- Carpeta de resultados -----------------------------------------------------
carpeta_resultados = 'resultados/problema1'
os.makedirs(carpeta_resultados, exist_ok=True)


# --- Cargo imagen -------------------------------------------------------------
img = cv2.imread('monedas.jpg')

if img is None:
    print('Error al cargar monedas.jpg')
    exit()

img = cv2.resize(img, (1024, 709))
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('Dimensiones:', img.shape)
print('Tipo de dato:', img.dtype)


# --- Suavizado ----------------------------------------------------------------
img_suav = cv2.GaussianBlur(img_gris, (3, 3), 0)


# --- Gradiente morfologico -----------------------------------------------------
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_grad = cv2.morphologyEx(img_suav, cv2.MORPH_GRADIENT, se)


# --- Umbralado ----------------------------------------------------------------
# Umbral 13 elegido experimentalmente: permite conservar
# los bordes de bajo contraste y detectar los 19 objetos.
_, img_bin = cv2.threshold(img_grad, 13, 255, cv2.THRESH_BINARY)


# --- Clausura -----------------------------------------------------------------
# Unimos pequenos cortes en los contornos, sin juntar objetos cercanos.
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_clau = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, se)


# --- Deteccion de contornos ----------------------------------------------------
contornos, _ = cv2.findContours(img_clau, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Revisamos primero los contornos mas grandes para evitar contar dos veces
# una misma moneda cuando aparecen un contorno externo y otro interior.
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

img_segmentada = np.zeros_like(img_gris)
img_segmentada_final = np.zeros_like(img_gris)
img_resultado = img_RGB.copy()
objetos = []

for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    area = cv2.contourArea(contorno)

    if 350 < area < 9000 and 28 < min(w, h) and max(w, h) < 130 and 0.5 < w/h < 1.8:

        # Evitamos repetir una region ya detectada.
        cx = x + w/2
        cy = y + h/2
        repetido = False

        for x0, y0, w0, h0, _ in objetos:
            if x0 <= cx <= x0 + w0 and y0 <= cy <= y0 + h0:
                repetido = True
                break

        if not repetido:
            objetos.append((x, y, w, h, contorno))  # guardo el contorno para minAreaRect (U7).
            cv2.drawContours(img_segmentada, [contorno], -1, 255, -1)

            # Aplicamos una clausura grande SOLO sobre este objeto.
            # Esto evita que se unan monedas cercanas entre si.
            margen = 45
            mascara_local = np.zeros((h + 2*margen, w + 2*margen), np.uint8)
            contorno_local = contorno.copy()
            contorno_local[:, :, 0] = contorno_local[:, :, 0] - x + margen
            contorno_local[:, :, 1] = contorno_local[:, :, 1] - y + margen

            cv2.drawContours(mascara_local, [contorno_local], -1, 255, -1)

            se_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
            mascara_local = cv2.morphologyEx(mascara_local, cv2.MORPH_CLOSE, se_grande)

            # Copiamos al resultado solamente la zona correspondiente al objeto.
            region = mascara_local[margen:margen+h, margen:margen+w]
            img_segmentada_final[y:y+h, x:x+w] = cv2.bitwise_or(
                img_segmentada_final[y:y+h, x:x+w], region
            )

            cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (255, 0, 0), 2)

contador = len(objetos)
print('Objetos detectados:', contador)

# --- Muestro resultados --------------------------------------------------------
plt.figure(figsize=(15, 10))

plt.subplot(331)
plt.imshow(img_RGB)
plt.title('Imagen original')
plt.xticks([]), plt.yticks([])

plt.subplot(332)
plt.imshow(img_gris, cmap='gray')
plt.title('Escala de grises')
plt.xticks([]), plt.yticks([])

plt.subplot(333)
plt.imshow(img_suav, cmap='gray')
plt.title('Suavizado')
plt.xticks([]), plt.yticks([])

plt.subplot(334)
plt.imshow(img_grad, cmap='gray')
plt.title('Gradiente morfologico')
plt.xticks([]), plt.yticks([])

plt.subplot(335)
plt.imshow(img_bin, cmap='gray')
plt.title('Umbralado')
plt.xticks([]), plt.yticks([])

plt.subplot(336)
plt.imshow(img_clau, cmap='gray')
plt.title('Clausura (3x3)')
plt.xticks([]), plt.yticks([])

plt.subplot(337)
plt.imshow(img_segmentada, cmap='gray')
plt.title('Máscara con contornos')
plt.xticks([]), plt.yticks([])

plt.subplot(338)
plt.imshow(img_segmentada_final, cmap='gray')
plt.title('Máscara final (clausura individual)')
plt.xticks([]), plt.yticks([])

plt.subplot(339)
plt.imshow(img_resultado)
plt.title(f'Objetos detectados: {contador}')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_A_segmentacion.png'), dpi=130)
plt.show(block=False)


# =============================================================================
# --- Problema 1.B: separo monedas y dados por su forma -------------------------
# =============================================================================

# Combinamos dos descriptores geometricos de la Unidad 7:
# - Rectangularidad girada: area / area de cv2.minAreaRect.
# - Circularidad: 4*pi*area / perimetro**2.
# Las monedas son aproximadamente circulares; los dados no.

img_monedas = np.zeros_like(img_gris)
img_dados = np.zeros_like(img_gris)
img_clasificada = img_RGB.copy()

monedas = []
dados = []
rect_monedas, circ_monedas = [], []
rect_dados, circ_dados = [], []

for x, y, w, h, contorno in objetos:

    # Extraigo la region del objeto de la mascara final del punto A.
    region = img_segmentada_final[y:y+h, x:x+w]

    # Calculo el area y la rectangularidad.
    area = np.sum(region > 0)

    # Unidad 7: rectangularidad invariante a la orientacion.
    # Uso el rectangulo minimo girado (minAreaRect), no boundingRect.
    rect_girado = cv2.minAreaRect(contorno)
    lado1, lado2 = rect_girado[1]
    rectangularidad_girada = area / (lado1 * lado2) if lado1 * lado2 > 0 else 0

    # Unidad 7: mantengo circularidad como segundo descriptor.
    # Un dado con contorno incompleto puede dar rect. girada similar a una moneda.
    contornos_region, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno_region = max(contornos_region, key=cv2.contourArea)
    perimetro = cv2.arcLength(contorno_region, True)
    area_contorno = cv2.contourArea(contorno_region)
    circularidad = 4 * np.pi * area_contorno / perimetro**2 if perimetro > 0 else 0

    # umbrales 0.90 y 0.82 elegidos a mitad de los intervalos medidos:
    # monedas: rect. girada ~0.79-0.82 y circ. ~0.85-0.90;
    # dados: rect. girada ~0.96 / 0.78 y circ. ~0.72-0.76.
    # El OR evita perder el dado cuya silueta quedo incompleta.
    if rectangularidad_girada > 0.90 or circularidad < 0.82:
        # Dado: guardo su region y marco su recuadro en rojo.
        dados.append((x, y, w, h))
        rect_dados.append(rectangularidad_girada)
        circ_dados.append(circularidad)
        img_dados[y:y+h, x:x+w][region > 0] = 255
        cv2.rectangle(img_clasificada, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(f'Dado  | x={x:3d}, y={y:3d}, rect_girada={rectangularidad_girada:.3f}, circ={circularidad:.3f}')
    else:
        # Moneda: guardo su region y marco su recuadro en verde.
        monedas.append((x, y, w, h))
        rect_monedas.append(rectangularidad_girada)
        circ_monedas.append(circularidad)
        img_monedas[y:y+h, x:x+w][region > 0] = 255
        cv2.rectangle(img_clasificada, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(f'Moneda | x={x:3d}, y={y:3d}, rect_girada={rectangularidad_girada:.3f}, circ={circularidad:.3f}')

print('Cantidad de monedas:', len(monedas))
print('Cantidad de dados:', len(dados))


# --- Muestro resultados del punto B -------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(img_RGB)
plt.title('Imagen original')
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(img_monedas, cmap='gray')
plt.title(f'Monedas: {len(monedas)}')
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(img_dados, cmap='gray')
plt.title(f'Dados: {len(dados)}')
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(img_clasificada)
plt.title('Monedas (verde) y dados (rojo)')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_B_separacion_resultado.png'), dpi=130)
plt.show(block=False)  # solo bloquea la ultima figura.

# --- Comparo descriptores geometricos (Unidad 7) ---------------------------
plt.figure(figsize=(7, 5))
plt.scatter(rect_monedas, circ_monedas, label='Monedas', marker='o')
plt.scatter(rect_dados, circ_dados, label='Dados', marker='s')
# Limites de decision: agrego etiquetas para identificar cada umbral.
plt.axvline(0.90, linestyle='--', color='gray', label='Límite rectangularidad: 0.90')
plt.axhline(0.82, linestyle=':', color='black', label='Límite circularidad: 0.82')
plt.xlabel('Rectangularidad girada (minAreaRect)')
plt.ylabel('Circularidad')
plt.title('Rectangularidad girada vs. circularidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_B_descriptores.png'), dpi=130)
plt.show(block=False)


# =============================================================================
# --- Problema 1.B: Clasificacion por valor de moneda ------------------------
# =============================================================================

# Se tienen en cuenta el diametro (Unidad 7) y la saturacion del
# centro y del borde (Unidad 5, espacio HSV).
# Los valores de los umbrales se ajustaron a monedas.jpg.
# Los valores nominales fueron verificados para esta imagen.

img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)

mascara_1peso = np.zeros_like(img_gris)
mascara_50cent = np.zeros_like(img_gris)
mascara_10cent = np.zeros_like(img_gris)
img_valores = img_RGB.copy()

cantidad_1peso = 0
cantidad_50cent = 0
cantidad_10cent = 0

# Colores RGB compartidos por el conteo y la figura final.
# 10 centavos: azul; 1 peso: verde; 50 centavos: rojo.
colores_monedas = {'10 centavos': (0, 0, 255),
                   '1 peso': (0, 255, 0),
                   '50 centavos': (255, 0, 0)}

# Guardo tipo y recuadro de cada moneda para la figura final.
clasificacion = []

for x, y, w, h in monedas:

    # --- Analizo tamaño ------------------------------------------------------
    diametro = (w + h) / 2

    # --- Analizo color -------------------------------------------------------
    canal_S = img_HSV[y:y+h, x:x+w, 1]
    centro = canal_S[h//3:2*h//3, w//3:2*w//3]
    anillo = canal_S[h//10:h//5, w//3:2*w//3]

    saturacion_centro = np.mean(centro)
    saturacion_anillo = np.mean(anillo)
    diferencia_S = saturacion_centro - saturacion_anillo

    region = img_segmentada_final[y:y+h, x:x+w]

    # Las monedas pequeñas tienen aproximadamente 78 píxeles de diámetro.
    # En esta imagen corresponden a monedas de 10 centavos.
    if diametro < 85:
        tipo = '10 centavos'
        cantidad_10cent += 1
        mascara_10cent[y:y+h, x:x+w][region > 0] = 255

    # En las de $1, el centro es dorado y el anillo exterior plateado.
    elif diferencia_S > 30:
        tipo = '1 peso'
        cantidad_1peso += 1
        mascara_1peso[y:y+h, x:x+w][region > 0] = 255

    # En las de 50 centavos, el color dorado es similar en toda la moneda.
    else:
        tipo = '50 centavos'
        cantidad_50cent += 1
        mascara_50cent[y:y+h, x:x+w][region > 0] = 255

    # Uso los mismos colores RGB en las dos figuras de clasificación.
    color = colores_monedas[tipo]

    # Conservo la clasificación sin volver a calcularla al final.
    clasificacion.append((x, y, w, h, tipo))

    print(f'{tipo:23s} | diametro={diametro:5.1f} | '
          f'S centro={saturacion_centro:5.1f} | '
          f'S borde={saturacion_anillo:5.1f}')

    cv2.rectangle(img_valores, (x, y), (x+w, y+h), color, 2)
    # Marcamos cada grupo con un color diferente.

print('\n--- Conteo por grupo ---')
print('Monedas bimetalicas de $1:', cantidad_1peso)
print('Monedas doradas de 50 centavos:', cantidad_50cent)
print('Monedas de 10 centavos:', cantidad_10cent)
print('Total monedas:', cantidad_1peso + cantidad_50cent + cantidad_10cent)

# --- Muestro resultados -------------------------------------------------------
plt.figure(figsize=(14, 9))

plt.subplot(231)
plt.imshow(img_RGB)
plt.title('Imagen original')
plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(img_monedas, cmap='gray')
plt.title(f'Monedas detectadas: {len(monedas)}')
plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(img_valores)
plt.title('Clasificacion: 10c azul | 1 peso verde | 50c rojo', fontsize=10)
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(mascara_1peso, cmap='gray')
plt.title(f'Bimetalicas (1 peso): {cantidad_1peso}')
plt.xticks([]), plt.yticks([])

plt.subplot(235)
plt.imshow(mascara_50cent, cmap='gray')
plt.title(f'Doradas (50 c): {cantidad_50cent}')
plt.xticks([]), plt.yticks([])

plt.subplot(236)
plt.imshow(mascara_10cent, cmap='gray')
plt.title(f'10 centavos: {cantidad_10cent}')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_B_conteo.png'), dpi=150)
plt.show(block=False)  # solo bloquea la ultima figura.


# =============================================================================
# --- Problema 1.C: Conteo de puntos en los dados ------------------------------
# =============================================================================

# Los dados ya fueron detectados y separados en el punto B.
# Los ordenamos de izquierda a derecha.

dados = sorted(dados, key=lambda d: d[0])

img_resultado_dados = img_RGB.copy()
resultados_dados = []

plt.figure(figsize=(18, 8))

for i, (x, y, w, h) in enumerate(dados):

    # --- Recorto el dado ----------------------------------------------
    img_dado = img_RGB[y:y+h, x:x+w]
    img_dado_gris = cv2.cvtColor(img_dado, cv2.COLOR_RGB2GRAY)

    # --- Suavizado ----------------------------------------------------
    img_dado_suav = cv2.GaussianBlur(img_dado_gris, (3, 3), 0)

    # --- umbralado inverso automatico de Otsu (U3) -------------
    # Otsu elige el umbral automaticamente (126 y 136 en estos dados).
    umbral_dado, img_puntos = cv2.threshold(
        img_dado_suav, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    print(f'Dado {i+1}: umbral Otsu = {umbral_dado:.0f}')

    # --- Deteccion de contornos ---------------------------------------
    contornos_puntos, _ = cv2.findContours(
        img_puntos, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    img_puntos_filtrados = np.zeros_like(img_dado_gris)
    img_dado_marcado = img_dado.copy()

    cantidad_puntos = 0

    # --- Analizo cada componente --------------------------------------
    for contorno in contornos_puntos:

        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        if perimetro == 0:
            continue

        # Circularidad (Unidad 7)
        circularidad = 4 * np.pi * area / (perimetro ** 2)

        px, py, pw, ph = cv2.boundingRect(contorno)
        relacion = pw / ph

        # mantengo los filtros originales: el area 70-250 px² es
        # el rango observado para los puntos tras Otsu (aprox. 134-160 px²).
        if 70 < area < 250 and circularidad > 0.70 and 0.65 < relacion < 1.4:

            cantidad_puntos += 1

            cv2.drawContours(
                img_puntos_filtrados, [contorno], -1, 255, -1
            )

            cx = px + pw // 2
            cy = py + ph // 2

            cv2.circle(
                img_dado_marcado, (cx, cy), 9, (255, 0, 0), 2
            )

    resultados_dados.append(cantidad_puntos)

    print(f'Dado {i+1}: {cantidad_puntos} puntos')

    # --- Marco el dado en la imagen original -------------------------
    cv2.rectangle(
        img_resultado_dados,
        (x, y), (x+w, y+h),
        (255, 0, 0), 2
    )

    # --- Muestro todas las etapas --------------------------------------
    fila = i * 6

    plt.subplot(2, 6, fila + 1)
    plt.imshow(img_dado)
    plt.title(f'Dado {i+1} - Original')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 6, fila + 2)
    plt.imshow(img_dado_gris, cmap='gray')
    plt.title('Escala de grises')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 6, fila + 3)
    plt.imshow(img_dado_suav, cmap='gray')
    plt.title('Suavizado')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 6, fila + 4)
    plt.imshow(img_puntos, cmap='gray')
    plt.title('Umbralado inverso')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 6, fila + 5)
    plt.imshow(img_puntos_filtrados, cmap='gray')
    plt.title('Puntos filtrados')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 6, fila + 6)
    plt.imshow(img_dado_marcado)
    plt.title(f'Valor: {cantidad_puntos}')
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_C_etapas.png'), dpi=150)
plt.show(block=False)


# --- Resultado final --------------------------------------------------
plt.figure(figsize=(10, 7))

plt.imshow(img_resultado_dados)

plt.title('Dados detectados: ' + ', '.join(
    f'Dado {i+1} = {valor}'
    for i, valor in enumerate(resultados_dados)
))

plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_C_resultado.png'), dpi=150)
plt.show(block=False)  # La figura final integrada es la que bloquea.

# Las dos figuras del punto C se guardan para utilizarlas en el informe.


# =============================================================================
# --- resultado final integrado de los puntos A, B y C -----------------
# =============================================================================
# Se usan las mascaras ya calculadas: no se vuelve a detectar ni clasificar.
img_resultado_final = img_RGB.copy()

# Se reutiliza el diccionario colores_monedas definido en el punto B.

for x, y, w, h, tipo in clasificacion:
    # extraigo el contorno de la mascara final, en coordenadas locales.
    region = img_segmentada_final[y:y+h, x:x+w].copy()
    contornos_moneda, _ = cv2.findContours(
        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos_moneda:
        # El contorno mayor representa la moneda; offset lo ubica en img_RGB.
        contorno_moneda = max(contornos_moneda, key=cv2.contourArea)
        cv2.drawContours(img_resultado_final, [contorno_moneda], -1,
                         colores_monedas[tipo], 2, offset=(x, y))

    # Etiquetas compactas, centradas sobre cada moneda para evitar solapamientos.
    # Mantengo el tipo completo en la clasificacion; solo abrevio el texto del dibujo.
    etiquetas_monedas = {'10 centavos': '10c', '1 peso': '1 peso', '50 centavos': '50c'}
    texto = etiquetas_monedas[tipo]
    (ancho_texto, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
    x_texto = max(0, min(img_resultado_final.shape[1] - ancho_texto - 2,
                         x + (w - ancho_texto)//2))
    cv2.putText(img_resultado_final, texto, (x_texto, max(15, y-7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colores_monedas[tipo], 2)

# marco los dados en magenta con el conteo existente del punto C.
for (x, y, w, h), valor in zip(dados, resultados_dados):
    region = img_segmentada_final[y:y+h, x:x+w].copy()
    contornos_dado, _ = cv2.findContours(
        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos_dado:
        contorno_dado = max(contornos_dado, key=cv2.contourArea)
        cv2.drawContours(img_resultado_final, [contorno_dado], -1,
                         (255, 0, 255), 2, offset=(x, y))
    # Centro la etiqueta sobre el dado, igual que con las monedas.
    texto = f'Dado: {valor}'
    (ancho_texto, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
    x_texto = max(0, min(img_resultado_final.shape[1] - ancho_texto - 2,
                         x + (w - ancho_texto)//2))
    cv2.putText(img_resultado_final, texto, (x_texto, max(15, y-7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 2)

# calculo el valor total a partir de los conteos ya obtenidos.
total_pesos = cantidad_10cent*0.10 + cantidad_1peso*1.00 + cantidad_50cent*0.50
print(f'Total de monedas: {total_pesos:.2f} pesos')

valores_dados = ', '.join(str(valor) for valor in resultados_dados)
titulo_final = (f'10c: {cantidad_10cent} | 1 peso: {cantidad_1peso} | '
                f'50c: {cantidad_50cent} | Total: {total_pesos:.2f} pesos | '
                f'Dados: {valores_dados}')

plt.figure(figsize=(15, 9))
plt.imshow(img_resultado_final)
plt.title(titulo_final)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'TP2_Problema1_resultado_final.png'), dpi=150)
plt.show()  # unica figura con plt.show() bloqueante al final.
