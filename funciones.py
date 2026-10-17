import cv2
import numpy as np
import matplotlib.pyplot as plt


def mostrar_segmentacion(ruta):
    img = cv2.imread(ruta)
    if img is None:
        print("No se pudo leer la imagen. Revisa la ruta.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Escala de grises y suavizado para reducir ruido
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # 2. Umbral automático (Otsu) para separar objeto y fondo
    _, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Limpiar la máscara (cerrar huecos y quitar puntos sueltos)
    kernel = np.ones((7, 7), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # 4. Quedarse con el contorno más grande (el objeto principal)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado = img_rgb.copy()
    if contornos:
        mayor = max(contornos, key=cv2.contourArea)
        mascara = np.zeros_like(mascara)
        cv2.drawContours(mascara, [mayor], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(resultado, [mayor], -1, (0, 255, 0), 4)

    objeto = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara)

    # 5. Mostrar los pasos
    titulos = ["Original", "Máscara", "Objeto segmentado", "Contorno"]
    imagenes = [img_rgb, mascara, objeto, resultado]

    plt.figure(figsize=(16, 4))
    for i, (titulo, im) in enumerate(zip(titulos, imagenes), start=1):
        plt.subplot(1, 4, i)
        plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
        plt.title(titulo)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
