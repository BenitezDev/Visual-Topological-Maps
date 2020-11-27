import os
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from Network import LoadModel
import loadFramesFromVideos as lf
import numpy as np
import pickle


# Train y Test de la red

# se cargan los datos
output_dimension = (128,128) # dimension to resize the images, do not touch or u will have to change the net model

x,y = lf.developmentTest(100) # we get the training set from the test video

x = np.asarray(x) # change them from list to np array
y = np.asarray(y) 

x_dev, y_dev = lf.developmentTest(100) # we get the development set from the test video

x_dev = np.asarray(x_dev)
y_dev = np.asarray(y_dev) 


# se carga y se compila el modelo
model = LoadModel() # we get the model

model.compile(optimizer="Adam",  # compile it
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()],
              )


# se entrena la red
result = 0  #Best score in the training state
for i in range(100):
    right=0 #Auxiliar to calculate the accuracy
    model.fit(x, y, epochs=1, batch_size=32) #The training state
    prediction = model.predict(x_dev) #We predict on the dev sey and calculate the amount of matches out of it
    prediction = np.argmax(prediction, axis=1)
    for j in range(len(prediction)): 
        right += (prediction[j]==y_dev[j])
    score = right/len(prediction)
    print(score)
    if(score > result):
        best_model = model #We save the model that got best score on the dev set
        result = score
print("Accuracy sobre el test set {:}".format(result))



# se calcula la prediccion de la red sobre el conjunto de test
num = lf.getFrameNumber() #Cargamos el numero de frames del video

test = [] 
n = int(num//50) #Calculamos de 50 en 50 para evitar colapso en la ram o la memoria de video
for i in range(n):
    print("{:} de {:}".format(i, n))
    frames = lf.getFrames(50)
    frames = np.asarray(frames)
    predictions = best_model.predict(frames) #Testeamos el modelo sobre el mejor resultado
    for j in range(len(predictions)):
        test.append(np.argmax(predictions[j]))
        
gt = pickle.load(open("clasificacion_frames.sav", 'rb')) #Cargamos el ground truth 

#Calculamos la tasa de acierto
right = 0
for i in range(len(test)):
    right += test[i]==gt[i]
print(right/len(test))

        
