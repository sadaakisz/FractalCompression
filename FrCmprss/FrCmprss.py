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

def generate_all_transformed_blocks(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            S = reducir(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append((k, l, direction, angle, aplicar_transformacion(S, direction, angle)))
    return transformed_blocks

def compress(img, source_size, destination_size, step):
    transformations = []
    transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
            print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            transformations[i].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
            # Test all possible transformations and take the best one
            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = encontrar_brillo_contraste(D, S)
                S = contrast*S + brightness
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transformations[i][j] = (k, l, direction, angle, contrast, brightness)
    return transformations

def decompress(transformations, source_size, destination_size, step, nb_iter=8):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                # Apply transform
                k, l, flip, angle, contrast, brightness = transformations[i][j]
                S = reducir(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
                D = aplicar_transformacion(S, flip, angle, contrast, brightness)
                cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))
    return iterations

# Compression for color images

def reduce_rgb(img, factor):
    img_r, img_g, img_b = extraer_rgb(img)
    img_r = reducir(img_r, factor)
    img_g = reducir(img_g, factor)
    img_b = reducir(img_b, factor)
    return construir_rgb(img_r, img_g, img_b)

def compress_rgb(img, source_size, destination_size, step):
    img_r, img_g, img_b = extraer_rgb(img)
    return [compress(img_r, source_size, destination_size, step), \
        compress(img_g, source_size, destination_size, step), \
        compress(img_b, source_size, destination_size, step)]

def decompress_rgb(transformations, source_size, destination_size, step, nb_iter=8):
    img_r = decompress(transformations[0], source_size, destination_size, step, nb_iter)[-1]
    img_g = decompress(transformations[1], source_size, destination_size, step, nb_iter)[-1]
    img_b = decompress(transformations[2], source_size, destination_size, step, nb_iter)[-1]
    return construir_rgb(img_r, img_g, img_b)

# Plot

def plot_iterations(iterations, target=None):
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

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]

# Tests

def test_greyscale():
    img = mpimg.imread('monkey.gif')
    img = escala_grises(img)
    img = reducir(img, 4)
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    transformations = compress(img, 8, 4, 8)
    iterations = decompress(transformations, 8, 4, 8)
    plot_iterations(iterations, img)
    plt.show()

def test_rgb():
    img = mpimg.imread('lena.gif')
    img = reduce_rgb(img, 8)
    transformations = compress_rgb(img, 8, 4, 8)
    retrieved_img = decompress_rgb(transformations, 8, 4, 8)
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
    plt.subplot(122)
    plt.imshow(retrieved_img.astype(np.uint8), interpolation='none')
    plt.show()

if __name__ == '__main__':
    test_greyscale()
    #test_rgb()
