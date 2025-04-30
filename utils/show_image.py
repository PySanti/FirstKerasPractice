import matplotlib.pyplot as plt
import numpy as np
def show_image(image):
    """
    Grafica una imagen en escala de grises utilizando matplotlib.
    
    Parámetros:
    - image: Array de la imagen (2D) en escala de grises.
    - title: Título de la gráfica (opcional).
    """
    X_new = image.copy()
    plt.imshow(np.reshape(X_new, (28,28)), cmap=plt.cm.gray)
    plt.show()


