import cv2
import numpy as np
import matplotlib.pyplot as plt

def agregar_ruido_gaussiano(imagen, desviacion):
    ruido = np.random.normal(0, desviacion, imagen.shape).astype(np.float32)
    imagen_ruidosa = cv2.add(imagen.astype(np.float32), ruido)
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)

def agregar_ruido_sal_pimienta(imagen, cantidad):
    imagen_ruidosa = imagen.copy()
    num_sal = int(np.ceil(cantidad * imagen.size * 0.5))
    num_pimienta = int(np.ceil(cantidad * imagen.size * 0.5))
    
    coordenadas_sal = [np.random.randint(0, i - 1, num_sal) for i in imagen.shape[:2]]
    coordenadas_pimienta = [np.random.randint(0, i - 1, num_pimienta) for i in imagen.shape[:2]]
    
    imagen_ruidosa[coordenadas_sal[0], coordenadas_sal[1]] = 255
    imagen_ruidosa[coordenadas_pimienta[0], coordenadas_pimienta[1]] = 0
    
    return imagen_ruidosa

def agregar_ruido_multiplicativo(imagen, desviacion):
    ruido = np.random.normal(1, desviacion, imagen.shape).astype(np.float32)
    imagen_ruidosa = cv2.multiply(imagen.astype(np.float32), ruido)
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)

# Cargar imágenes
nombres_imagenes = ["imagen1.jpg", "imagen2.jpg", "imagen3.jpg", "imagen4.jpg"]
imagenes = [cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) for nombre in nombres_imagenes]

# Aplicar ruido y mostrar resultados
for i, img in enumerate(imagenes):
    if img is None:
        print(f"Error: No se pudo cargar {nombres_imagenes[i]}")
        continue
    
    img_gaussiano = agregar_ruido_gaussiano(img, desviacion=25)
    img_sal_pimienta = agregar_ruido_sal_pimienta(img, cantidad=0.02)
    img_multiplicativo = agregar_ruido_multiplicativo(img, desviacion=0.1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"{nombres_imagenes[i]} - Original")
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_gaussiano, cmap='gray')
    plt.axis('off')
    plt.title("Ruido Gaussiano")
    
    plt.subplot(1, 4, 3)
    plt.imshow(img_sal_pimienta, cmap='gray')
    plt.axis('off')
    plt.title("Ruido Sal y Pimienta")
    
    plt.subplot(1, 4, 4)
    plt.imshow(img_multiplicativo, cmap='gray')
    plt.axis('off')
    plt.title("Ruido Multiplicativo")
    
    plt.show()

