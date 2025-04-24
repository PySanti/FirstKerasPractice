# FirstKerasPractice

El objetivo de este proyecto sera por primera vez realizar una practica de DNN (Deep Neural Networks) utilizando Keras.

El dataset a utilizar sera el provisto por `tensorflow.keras.datasets.mnist`.

El objetivo sera crear una red neuronal capaz de predecir cual numero contiene una imagen del `0-9`.

El conjunto esta compuesto por imagenes de `28x28` pixeles.

## Preprocesamiento

Shape del dataset:

Se aplanaran las imagenes para lograr representar cada imagen en forma de vector.

Se normalizaran los valores de las imagenes.

No se llevara a cabo ninguna de las 9 tecnicas de preprocesamiento clasicas (duplicados, nans, etc.) dado que no es necesario.


## Entrenamiento

En el proceso de entrenamiento se utilizara una MLP, utilizando `softmax` en la output layer y `ReLu` en el resto de capas.

## Evaluacion

