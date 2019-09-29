import matplotlib.pyplot as plt # crea una figura, o lineas en un area y pone labes
import matplotlib.image as mpimg # carga la info de la data (solo funciona en formato .png)
from scipy import ndimage # paquete que contiene procesamientos de imagenes multi-dimencional
from scipy import optimize # paquete para reducir y optimizar ecuaciones o formulas
import numpy as np # libreria para computacion cientifica para arrays multi-dimensionales
import math

# FORMULA: fl(xDk)=s×rotateθ(flipd(reduce(xDk)))+b
# REDUCE es para ir de 8x8 a 4x4
# s es el contraste y b el brillo
# flip y rotate son transformaciones junto con reduce

# Manipulación de canales de color

def escala_grises(img): # tonos de grises (entre blanco y negro 256) del RGB
    return np.mean(img[:,:,:2], 2) # .mean devuelve el promedio de un array

def extraer_rgb(img): # extraemos los colores RGB verde, rojo y azul
    return img[:,:,0], img[:,:,1], img[:,:,2] #[:, :, 0] = Verde [:, :, 1] = Azul [:, :, 2] = Rojo

def construir_rgb(img_r, img_g, img_b):
    forma = (img_r.shape[0], img_r.shape[1], 1)
    return np.concatenate((np.reshape(img_r, forma), np.reshape(img_g, forma), #une una secuencia de arrays con un axis existente
        np.reshape(img_b, forma)), axis=2) # .reshape cambia la forma del array

# Transformaciones

def reducir(img, factor): # reduce la imagen por el promedio de cada elemento
    resultado = np.zeros((img.shape[0] // factor, img.shape[1] // factor)) #retorna nuevo array del tamaño y tipo indicado, llenado de 0s
    for i in range(resultado.shape[0]):
        for j in range(resultado.shape[1]):
            resultado[i,j] = np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor]) #promedio del array (reduce el tamaño del la imagen)
    return resultado

def rotar(img, angulo): # rotas la imagen
    return ndimage.rotate(img, angulo, reshape=False) # se rota por el angulo dado como parametro (para preservar la forma de la imagen el angulo tiene que estar entre {0, 90, 180, 270})

def invertir(img, direccion): # voltea imagen
    return img[::direccion,:] # rota si es -1 y no hace nada si es 1

def aplicar_transformacion(img, direccion, angulo, contraste=1.0, brillo=0.0): # aplica la transformacion
    return contraste*rotar(invertir(img, direccion), angulo) + brillo #1 bit del flip, 2 bits del angulo, 8 para el contraste y la brillo

# Contraste y brillo

def encontrar_brillo_contraste(D, S):
    # Arreglar el contraste y brillo
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    return x[1], x[0]

# Compression for greyscale images

def generar_todos_bloques_transformados(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            S = reducir(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidatos:
                transformed_blocks.append((k, l, direction, angle, aplicar_transformacion(S, direction, angle)))
    return transformed_blocks

def comprimir(img, tamano_origen, tamano_destino, pasos):
    transformacion = []
    transformacion_bloques = generar_todos_bloques_transformados(img, tamano_origen, tamano_destino, pasos)
    i_cont = img.shape[0] // tamano_destino
    j_cont = img.shape[1] // tamano_destino
    for i in range(i_cont):
        transformacion.append([])
        for j in range(j_cont):
            print("{}/{} ; {}/{}".format(i, i_cont, j, j_cont))
            transformacion[i].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i*tamano_destino:(i+1)*tamano_destino,j*tamano_destino:(j+1)*tamano_destino]
            # Test all possible transformations and take the best one
            for k, l, direccion, angulo, S in transformacion_bloques:
                contraste, brillo = encontrar_brillo_contraste(D, S)
                S = contraste*S + brillo
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transformacion[i][j] = (k, l, direccion, angulo, contraste, brillo)
    return transformacion

def descomprimir(transformaciones, tamano_origen, tamano_destino, pasos, nb_iter=8):
    factor = tamano_origen // tamano_destino
    altura = len(transformaciones) * tamano_destino
    ancho = len(transformaciones[0]) * tamano_destino
    iteraciones = [np.random.randint(0, 256, (altura, ancho))]
    cur_img = np.zeros((altura, ancho))
    for i_iter in range(nb_iter):
        print(i_iter)
        for i in range(len(transformaciones)):
            for j in range(len(transformaciones[i])):
                # Apply transform
                k, l, invertir, angulo, contraste, brillo = transformaciones[i][j]
                S = reducir(iteraciones[-1][k*pasos:k*pasos+tamano_origen,l*pasos:l*pasos+tamano_origen], factor)
                D = aplicar_transformacion(S, invertir, angulo, contraste, brillo)
                cur_img[i*tamano_destino:(i+1)*tamano_destino,j*tamano_destino:(j+1)*tamano_destino] = D
        iteraciones.append(cur_img)
        cur_img = np.zeros((altura, ancho))
    return iteraciones

# Compression for color images

def reducir_rgb(img, factor):
    img_r, img_g, img_b = extraer_rgb(img)
    img_r = reducir(img_r, factor)
    img_g = reducir(img_g, factor)
    img_b = reducir(img_b, factor)
    return construir_rgb(img_r, img_g, img_b)

def comprimir_rgb(img, source_size, destination_size, step):
    img_r, img_g, img_b = extraer_rgb(img)
    return [comprimir(img_r, source_size, destination_size, step), \
        comprimir(img_g, source_size, destination_size, step), \
        comprimir(img_b, source_size, destination_size, step)]

def descomprimir_rgb(transformations, source_size, destination_size, step, nb_iter=8):
    img_r = descomprimir(transformations[0], source_size, destination_size, step, nb_iter)[-1]
    img_g = descomprimir(transformations[1], source_size, destination_size, step, nb_iter)[-1]
    img_b = descomprimir(transformations[2], source_size, destination_size, step, nb_iter)[-1]
    return construir_rgb(img_r, img_g, img_b)

# Plot

def plot_iteraciones(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

# Parameters

direcciones = [1, -1]
angulos = [0, 90, 180, 270]
candidatos = [[direction, angle] for direction in direcciones for angle in angulos]

# Tests

def test_escalagrises():
    img = mpimg.imread('monkey.gif')
    img = escala_grises(img)
    img = reducir(img, 4)
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    transformaciones = comprimir(img, 8, 4, 8)
    iteraciones = descomprimir(transformaciones, 8, 4, 8)
    plot_iteraciones(iteraciones, img)
    plt.show()

def test_rgb():
    img = mpimg.imread('lena.gif')
    img = reducir_rgb(img, 8)
    transformations = comprimir_rgb(img, 8, 4, 8)
    retrieved_img = descomprimir_rgb(transformations, 8, 4, 8)
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
    plt.subplot(122)
    plt.imshow(retrieved_img.astype(np.uint8), interpolation='none')
    plt.show()

if __name__ == '__main__':
    test_escalagrises()
    #test_rgb()
