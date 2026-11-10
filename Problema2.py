import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# --- PROBLEMA 2.A: DETECCION AUTOMATICA DE PATENTES ----------------------------
# =============================================================================

carpeta_imagenes = 'Patentes'
carpeta_resultados = 'resultados/problema2'
os.makedirs(carpeta_resultados, exist_ok=True)
# Cambiar a True para abrir tambien en pantalla las etapas de las 12 imagenes.
# Con False se muestra el ejemplo img05 y se guardan TODAS las figuras.
mostrar_todas_etapas = False


# --- umbrales reducidos y criterio geometrico unico -------------------
# Otsu y cinco niveles fijos (90,110,135,150,170). Son intensidades probadas
# de oscuras a claras para las letras blancas sobre placa oscura.
# Se redujo la busqueda original de 14 umbrales a seis posibilidades.
umbrales_caracteres = [0, 90, 110, 135, 150, 170]


def segmentar_caracteres(gris):
    alto, ancho = gris.shape
    # Busco seis componentes de alturas parecidas, alineadas y con la separacion
    # mas grande entre las letras (3) y los numeros (3).
    for umbral in umbrales_caracteres:
        tipo = cv2.THRESH_BINARY
        if umbral == 0:
            tipo += cv2.THRESH_OTSU  # 0 significa elegir umbral con Otsu.
        _, binaria = cv2.threshold(gris, umbral, 255, tipo)

        for apertura in [0, 2]:
            # Apertura vertical 1x2: elimina puentes horizontales de 1 pixel
            # entre una letra y el marco; solo se usa si la opcion simple falla.
            if apertura == 0:
                procesada = binaria.copy()
            else:
                se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, apertura))
                procesada = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, se)

            n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(procesada, 8)
            caracteres = []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                # Rangos relativos heredados del script: las letras ocupan aprox.
                # 27%-94% de la altura del recorte, y menos de 25% de su ancho.
                # Los valores de area/forma descartan bordes y puntos aislados.
                if (0.27*alto < h < 0.94*alto and
                    max(2, 0.055*ancho) < w < 0.25*ancho and
                    5 < area < 0.19*ancho*alto and
                    0.03*ancho < x < 0.98*ancho and w/h < 1.1):
                    caracteres.append((int(x), int(y), int(w), int(h), i))

            if len(caracteres) != 6:
                continue

            caracteres.sort(key=lambda c: c[0])
            alturas = np.array([c[3] for c in caracteres])
            centros_y = np.array([c[1]+c[3]/2 for c in caracteres])
            espacios = [caracteres[j+1][0] - (caracteres[j][0]+caracteres[j][2])
                        for j in range(5)]

            # validacion geometrica sin puntuacion con pesos arbitrarios.
            # 0.65: la menor altura al menos 65% de la mayor (permite perspectiva).
            # 0.45: tolerancia vertical por inclinacion de algunas placas (~14 px/36).
            # +2: el espacio entre el 3ro y 4to debe superar a los otros en 2 px.
            if min(alturas) < 0.65 * max(alturas):
                continue
            if max(centros_y) - min(centros_y) > 0.45 * alto:
                continue
            if espacios[2] < max(espacios[0], espacios[1],
                                espacios[3], espacios[4]) + 2:
                continue

            return caracteres, procesada, etiquetas, umbral, apertura

    # Si no hay seis regiones validas, devuelvo Otsu para diagnosticar.
    _, diagnostico = cv2.threshold(gris, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [], diagnostico, np.zeros_like(gris, dtype=np.int32), None, None


# --- recorte con margen vertical proporcional --------------------------
def recortar_patente(img, caja, proporcion=0.10):
    x, y, w, h = caja
    # Margen horizontal 2 px (del codigo original); vertical 10% del alto.
    # Este margen permite incluir el borde completo aun si la placa esta inclinada.
    margen_h = 2
    margen_v = max(1, round(proporcion * h))
    return img[max(0,y-margen_v):min(img.shape[0],y+h+margen_v),
               max(0,x-margen_h):min(img.shape[1],x+w+margen_h)]


# --- busqueda de la patente sin puntaje arbitrario ---------------------
def detectar_patente(img):
    # una region candidata solo se acepta como patente si tiene
    # dimensiones compatibles, seis componentes alineados y de altura
    # parecida (tres letras y tres numeros) y suficiente cobertura horizontal.
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (3, 3), 0)
    alto, ancho = gris.shape

    # Cuatro pruebas con solo DOS operaciones morfologicas de Unidad 6.
    # El Black-Hat equivale a CLAUSURA menos imagen, que antes se hacia a mano.
    # 21x7: resalta caracteres oscuros/claros contra el marco en esta escala.
    # 9x9 y 3x3: resaltan detalles claros de diferente tamaño.
    # 13x3/17x3: unen letras alineadas en una region horizontal candidata.
    # Los umbrales 40/35 fueron elegidos en las pruebas sobre estas 12 fotos;
    # 0 representa Otsu. Se mantiene la opcion final solo como alternativa.
    opciones = [
        (cv2.MORPH_BLACKHAT, (21, 7), 40, (13, 3)),
        (cv2.MORPH_TOPHAT, (9, 9), 40, (17, 3)),
        (cv2.MORPH_TOPHAT, (3, 3), 35, (17, 3)),
        (cv2.MORPH_TOPHAT, (3, 3), 0, (17, 3))]

    mejor_caja = None
    mayor_area = 0
    mejores_etapas = None
    candidatos = 0

    for operacion, tamano, umbral, tam_cierre in opciones:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, tamano)
        morfologica = cv2.morphologyEx(suavizada, operacion, se)
        tipo = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if umbral == 0 else 0)
        _, binaria = cv2.threshold(morfologica, umbral, 255, tipo)
        se_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, tam_cierre)
        clausura = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, se_cierre)

        contornos, _ = cv2.findContours(clausura, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            # Dimensiones de las placas en las 12 fotos (640x360):
            # ancho 45-155 px; alto 19-62 px; proporcion 1.7-7.5.
            if not (45 <= w <= 155 and 19 <= h <= 62 and 1.7 < w/h < 7.5):
                continue
            # descarto regiones cercanas a los 12 px superiores
            # o a los 20 px inferiores de la foto: en estas 12 imagenes la
            # patente nunca esta alli (pueden ser capot o paragolpes).
            if y < 12 or y+h > alto-20:
                continue

            recorte = recortar_patente(img, (x,y,w,h))
            region = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            caracteres, _, _, _, _ = segmentar_caracteres(region)
            if len(caracteres) != 6:
                continue
            cobertura = (caracteres[-1][0] + caracteres[-1][2] - caracteres[0][0]) / region.shape[1]
            # En placas validas los 6 caracteres cubren >=70% del recorte;
            # los falsos candidatos de carroceria suelen cubrir <60%.
            if cobertura < 0.70:
                continue

            candidatos += 1
            # elijo el rectangulo MAS GRANDE entre los que cumplen
            # el criterio de seis componentes, para no cortar bordes de placa.
            if w*h > mayor_area:
                mayor_area = w*h
                mejor_caja = (x, y, w, h)
                mejores_etapas = (gris, suavizada, morfologica, binaria, clausura)

    return mejor_caja, candidatos, mejores_etapas


# --- recuperar trazos unidos al marco con Top-Hat horizontal (U6) -----
def recuperar_caracteres_angostos(gris, caracteres, umbral):
    # Se aplica SOLO cuando un componente resulta demasiado angosto respecto
    # de los restantes: puede ser un caracter unido al marco (ej. img03).
    # la regla exige que exista una region mas ancha del Top-Hat
    # que contenga horizontalmente al caracter angosto. Un '1' o 'I' aislado
    # normalmente no tendra esa region y, por lo tanto, no se reemplaza.
    reparaciones = {}
    if len(caracteres) != 6:
        return caracteres, reparaciones, None

    anchos = np.array([c[2] for c in caracteres])
    mediana = np.median(anchos)
    if mediana <= 0:
        return caracteres, reparaciones, None

    # Kernel horizontal de longitud similar al ancho mediano de un caracter,
    # con minimo 5 px para tener un elemento estructural significativo.
    largo = max(5, 2*(int(mediana)//2)+1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (largo, 1))
    modo = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if umbral==0 else 0)
    _, bin_base = cv2.threshold(gris, umbral, 255, modo)
    sin_lineas = cv2.morphologyEx(bin_base, cv2.MORPH_TOPHAT, kernel)
    n, etiquetas_nuevas, stats, _ = cv2.connectedComponentsWithStats(sin_lineas, 8)

    for j, (x, y, w, h, etiqueta) in enumerate(caracteres):
        # 70% de la mediana: identifica letras visiblemente partidas.
        if w >= 0.70*mediana:
            continue
        candidatos = []
        for k in range(1, n):
            xx, yy, ww, hh, area = stats[k]
            # 0.80-1.80 de la mediana: admito un ancho similar pero mas completo.
            # 0.75-1.40 de altura: tolerancia a perspectiva y pixeles faltantes.
            # Distancia minima 1 px al borde: evita elegir el propio marco.
            if (xx <= x and xx+ww >= x+w-1 and
                0.80*mediana < ww < 1.80*mediana and
                0.75*h < hh < 1.4*h and
                xx > 1 and yy > 1 and
                xx+ww < gris.shape[1]-1 and yy+hh < gris.shape[0]-1):
                candidatos.append((int(area), int(xx), int(yy), int(ww), int(hh), k))
        if candidatos:
            _, xx, yy, ww, hh, k = max(candidatos)
            caracteres[j] = (xx, yy, ww, hh, etiqueta)
            reparaciones[etiqueta] = (etiquetas_nuevas == k)
            print(f'  caracter {j+1} recuperado con Top-Hat horizontal')

    return caracteres, reparaciones, (sin_lineas if reparaciones else None)


# --- Proceso las 12 imagenes ---------------------------------------------------
fig_resumen = plt.figure(figsize=(15, 12))
contador = 0
patentes_detectadas = []   # comparto recortes de A con B en memoria.

for numero in range(1, 13):
    nombre = f'img{numero:02d}.png'
    img = cv2.imread(os.path.join(carpeta_imagenes, nombre))
    if img is None:
        print('No se pudo cargar:', nombre)
        continue

    caja, cantidad_candidatos, etapas = detectar_patente(img)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resultado = img_RGB.copy()

    if caja is None:
        print(nombre, ': no se detecto patente')
    else:
        contador += 1
        x, y, w, h = caja
        # primero 10% de margen vertical. Si una letra toca el borde,
        # intento 25% para incluirla completa (por ejemplo img02).
        recorte_bgr = recortar_patente(img, caja, proporcion=0.10)
        rec_gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
        prelim, _, _, _, _ = segmentar_caracteres(rec_gris)
        if len(prelim) == 6 and any((c[1] <= 1 or c[1]+c[3] >= rec_gris.shape[0]-1) for c in prelim):
            recorte_amplio = recortar_patente(img, caja, proporcion=0.25)
            ampliado, _, _, _, _ = segmentar_caracteres(cv2.cvtColor(recorte_amplio, cv2.COLOR_BGR2GRAY))
            if len(ampliado) == 6:
                recorte_bgr = recorte_amplio
                print(f'{nombre}: amplio margen vertical para no cortar una letra')
        recorte = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
        patentes_detectadas.append((numero, recorte_bgr))  # No releer PNG en B.
        cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(f'{nombre}: patente en ({x}, {y}), {w}x{h}, candidatos={cantidad_candidatos}')
        plt.imsave(os.path.join(carpeta_resultados,
                     f'patente_{numero:02d}.png'), recorte)

        # --- Muestro todas las etapas de esta imagen -------------------
        gris, suavizada, morfologica, binaria, clausura = etapas
        imagenes = [img_RGB, gris, suavizada, morfologica, binaria,
                    clausura, img_resultado, recorte]
        titulos = ['Original', 'Grises', 'Suavizado', 'Realce morfologico',
                   'Umbralado', 'Clausura', 'Deteccion', 'Patente recortada']
        plt.figure(figsize=(12, 7))
        for k in range(8):
            plt.subplot(2, 4, k+1)
            if imagenes[k].ndim == 2:
                plt.imshow(imagenes[k], cmap='gray')
            else:
                plt.imshow(imagenes[k])
            plt.title(titulos[k])
            plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_resultados,
                    f'etapas_{numero:02d}.png'), dpi=130)
        if mostrar_todas_etapas or numero == 5:
            plt.show(block=False)    # Ejemplo de las etapas en pantalla.
        else:
            plt.close()             # El resto queda guardado en archivos PNG.

    plt.figure(fig_resumen.number)
    plt.subplot(4, 3, numero)
    plt.imshow(img_resultado)
    plt.title(nombre)
    plt.xticks([]), plt.yticks([])

print(f'Patentes detectadas: {contador}/12')
print('Las etapas de cada imagen se guardaron en:', carpeta_resultados)
plt.figure(fig_resumen.number)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados,
            'resumen_12_patentes.png'), dpi=150)
plt.show()

# =============================================================================
# --- PROBLEMA 2.B: SEGMENTACION DE CARACTERES EN LAS 12 PATENTES --------------
# =============================================================================

carpeta_resultados_caracteres = 'resultados/problema2/caracteres'
os.makedirs(carpeta_resultados_caracteres, exist_ok=True)

# --- utilizo en memoria los recortes de A, no cv2.imread de PNG --------
fig_resumen = plt.figure(figsize=(15, 12))
correctas = 0

# --- Resumen comparativo: guardo cada recorte y su mascara en memoria --------
resumen_comparativo = []

for numero, img in patentes_detectadas:
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    caracteres, binaria, etiquetas, umbral_usado, apertura_usada = segmentar_caracteres(gris)
    caracteres, reparaciones, imagen_tophat = recuperar_caracteres_angostos(
        gris, caracteres, umbral_usado)

    img_mascara = np.zeros_like(gris)
    img_marcada = img_RGB.copy()

    for x, y, w, h, etiqueta in caracteres:
        if etiqueta in reparaciones:
            img_mascara[reparaciones[etiqueta]] = 255
        else:
            img_mascara[etiquetas == etiqueta] = 255
        cv2.rectangle(img_marcada, (x, y), (x+w, y+h), (255, 0, 0), 1)

    # --- Guardo cada caracter por separado ------------------------------------
    for j, (x, y, w, h, etiqueta) in enumerate(caracteres, 1):
        caracter = img_mascara[y:y+h, x:x+w]
        plt.imsave(os.path.join(carpeta_resultados_caracteres,
                   f'caracter_{numero:02d}_{j}.png'), caracter, cmap='gray')

    cantidad = len(caracteres)
    # Guardo la placa original y los caracteres que ya fueron segmentados.
    # No vuelvo a leer imagenes ni aplico un procesamiento nuevo.
    resumen_comparativo.append((numero, img_RGB.copy(), img_mascara.copy(), cantidad))

    if cantidad == 6:
        correctas += 1
        print(f'Patente {numero:02d}: 6 caracteres alineados | '
              f'umbral={umbral_usado if umbral_usado!=0 else "Otsu"} | apertura={apertura_usada}')
    else:
        print(f'Patente {numero:02d}: requiere revision manual')

    # --- muestro la etapa extra cuando se separa una letra del marco.
    # Esto documenta la operacion MORPH_TOPHAT de U6 en img03.
    if imagen_tophat is not None:
        plt.figure(figsize=(9, 4))
        plt.subplot(121), plt.imshow(binaria, cmap='gray'), plt.title('Umbralado inicial')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(imagen_tophat, cmap='gray'), plt.title('Top-Hat horizontal')
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_resultados_caracteres,
                    f'tophat_marco_{numero:02d}.png'), dpi=150)
        plt.close()

    # --- Guardo las etapas de procesamiento ----------------------------------
    plt.figure(figsize=(12, 7))
    imagenes = [img_RGB, gris, binaria, img_mascara, img_marcada]
    titulos = ['Patente original', 'Escala de grises', 'Umbralado / Apertura',
               'Caracteres segmentados', f'Caracteres separados: {cantidad}']
    for k in range(5):
        plt.subplot(2, 3, k+1)
        if imagenes[k].ndim == 2:
            plt.imshow(imagenes[k], cmap='gray')
        else:
            plt.imshow(imagenes[k])
        plt.title(titulos[k])
        plt.xticks([]), plt.yticks([])

    # --- Muestro los seis caracteres recortados de forma individual -----------
    # Es el resultado pedido en el punto B: cada componente por separado.
    # Reutilizo la mascara y las cajas calculadas, sin aplicar otro procesamiento.
    if caracteres:
        ancho_celda = max(c[2] for c in caracteres) + 8
        alto_celda = max(c[3] for c in caracteres) + 8
        img_individuales = np.zeros((alto_celda, cantidad*ancho_celda), np.uint8)

        for j, (x, y, w, h, etiqueta) in enumerate(caracteres):
            caracter = img_mascara[y:y+h, x:x+w]
            x0 = j*ancho_celda + (ancho_celda-w)//2
            y0 = (alto_celda-h)//2
            img_individuales[y0:y0+h, x0:x0+w] = caracter
    else:
        img_individuales = np.zeros_like(gris)

    plt.subplot(2, 3, 6)
    plt.imshow(img_individuales, cmap='gray', vmin=0, vmax=255)
    plt.title('Caracteres individuales')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_resultados_caracteres,
                f'etapas_caracteres_{numero:02d}.png'), dpi=150)
    if mostrar_todas_etapas or numero == 5:
        plt.show(block=False)    # Ejemplo de segmentacion de caracteres.
    else:
        plt.close()             # Las 12 figuras quedan guardadas.

    # --- Agrego al resumen general -------------------------------------------
    plt.figure(fig_resumen.number)
    plt.subplot(4, 3, numero)
    plt.imshow(img_marcada)
    plt.title(f'img{numero:02d}: {cantidad} caracteres')
    plt.xticks([]), plt.yticks([])

print(f'Patentes con seis caracteres validados: {correctas}/12')
print('Los recortes individuales y etapas se guardaron en:', carpeta_resultados_caracteres)
plt.figure(fig_resumen.number)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados_caracteres, 'resumen_caracteres.png'), dpi=150)
plt.show(block=False)

# =============================================================================
# --- Resumen comparativo: patente original y caracteres segmentados ----------
# =============================================================================
# Presento dos patentes por fila: recorte original a la izquierda y mascara
# de sus caracteres a la derecha, usando los resultados que ya calcule en B.
plt.figure(figsize=(16, 18))
for fila, (numero, patente, mascara, cantidad) in enumerate(resumen_comparativo):
    # Son doce patentes: seis filas por dos pares (4 columnas).
    renglon = fila // 2
    par = fila % 2
    posicion = renglon * 4 + par * 2

    plt.subplot(6, 4, posicion + 1)
    plt.imshow(patente)
    plt.title(f'img{numero:02d}.png', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplot(6, 4, posicion + 2)
    plt.imshow(mascara, cmap='gray', vmin=0, vmax=255)
    plt.title(f'{cantidad} caracteres', fontsize=10)
    plt.xticks([]), plt.yticks([])

plt.suptitle('Resumen: patentes detectadas y caracteres segmentados', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.975])
plt.savefig(os.path.join(carpeta_resultados_caracteres,
            'resumen_patentes_y_caracteres.png'), dpi=150)
plt.show()
