# FirstKerasPractice

El objetivo de este proyecto sera por primera vez realizar una practica de DNN (Deep Neural Networks) utilizando Keras.

El dataset a utilizar sera el provisto por `tensorflow.keras.datasets.mnist`.

El objetivo sera crear una red neuronal capaz de predecir cual numero contiene una imagen del `0-9`.

El conjunto esta compuesto por imagenes de `28x28` pixeles.

## Preprocesamiento

Shape del dataset: `(70000, 28, 28)` + `(70000, )`.

70000 registros de imagenes con su etiqueta, estando las imagenes representadas en forma de matriz de `28x28`.

Los registros vienen del modulo ya separados en train y test, nosotros separamos el testset en test-val a traves del siguiente codigo:

```
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split


(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=0.5, random_state=42, stratify=Y_test)
```





Se aplanaran las imagenes para lograr representar cada imagen en forma de vector.

Se normalizaran los valores de las imagenes.

No se llevara a cabo ninguna de las 9 tecnicas de preprocesamiento clasicas (duplicados, nans, etc.) dado que no es necesario.

Despues de dividir los conjuntos, aplanar las imagenes, normalizar y usar one-hot-encoding sobre los targets usando el siguiente codigo:

```
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def preprare_dataset(X_data, Y_data):
    X_data = X_data.reshape(X_data.shape[0], 28*28)
    X_data = X_data.astype('float32') / 255
    Y_data = to_categorical(Y_data, num_classes=10)
    return [X_data, Y_data]


def show_shape(X_data, Y_data, label):
    print(f"Mostrando info de {label}")

    print(f"Shape del x_data : {X_data.shape}")
    print(f"Shape del y_data : {Y_data.shape}")



(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=0.5, random_state=42, stratify=Y_test)

X_train, Y_train = preprare_dataset(X_train, Y_train)
X_val, Y_val = preprare_dataset(X_val, Y_val)
X_test, Y_test = preprare_dataset(X_test, Y_test)

show_shape(X_train, Y_train, "train_set")
show_shape(X_test, Y_test, "test_set")
show_shape(X_val, Y_val, "val_set")

```

Recibimos el siguiente output:

```
Mostrando info de train_set
Shape del x_data : (60000, 784)
Shape del y_data : (60000, 10)

Mostrando info de test_set
Shape del x_data : (5000, 784)
Shape del y_data : (5000, 10)

Mostrando info de val_set
Shape del x_data : (5000, 784)
Shape del y_data : (5000, 10)
```


## Entrenamiento

En el proceso de entrenamiento se utilizara una MLP, utilizando `softmax` en la output layer y `ReLu` en el resto de capas.

## Evaluacion

