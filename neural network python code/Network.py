import numpy as np
import tensorflow as tf

# Modelo de la red neuronal:

# se definen las capas que tendra el modelo
def LoadModel():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(96, (11,11),activation='relu', input_shape=(128,128,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(256, (7,7),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(384, (5,5),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(384, (5,5),activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(9, activation="softmax") 
        ])
        
        return model


# se define la funcion de normalizacion
def Normalize(image):
    stdR = np.std(image[:,:,0]) #Calculamos la varianza para cada espacio de color
    stdG = np.std(image[:,:,1])
    stdB = np.std(image[:,:,2])
    
    R = np.mean(image[:,:,0]) #La mediana de cada espacio de color
    G = np.mean(image[:,:,1])
    B = np.mean(image[:,:,2])
    
    image[:,:,0] = image[:,:,0] - R
    image[:,:,1] = image[:,:,1] - G
    image[:,:,2] = image[:,:,2] - B
    
    image[:,:,0] = image[:,:,0]/stdR
    image[:,:,1] = image[:,:,1]/stdG
    image[:,:,2] = image[:,:,2]/stdB
    
    minimoR = np.min(image[:,:,0])
    minimoG = np.min(image[:,:,1])
    minimoB = np.min(image[:,:,2])
    
    image[:,:,0] = image[:,:,0]-minimoR # Restamos el valor minimo de cada espacio para que el valor mas bajo sea 0
    image[:,:,1] = image[:,:,1]-minimoG
    image[:,:,2] = image[:,:,2]-minimoB
    
    maximoR = np.max(image[:,:,0]) 
    maximoG = np.max(image[:,:,1])
    maximoB = np.max(image[:,:,2])
    
    factorR = 1/maximoR #Calculamos el factor escala por el que hay que multiplicar cada espacio para que su valor maximo sea 1
    factorG = 1/maximoG
    factorB = 1/maximoB
    
    image[:,:,0] = image[:,:,0] * factorR #Operamos de modo que cada color tiene sus valores comprendidos entre [0,1]
    image[:,:,1] = image[:,:,1] * factorG
    image[:,:,2] = image[:,:,2] * factorB
    
    return image