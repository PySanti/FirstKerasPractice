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

Primero buscaremos entrenar la red de manera estandar.

Buscaremos probar el rendimiento y posibles errores ante los siguientes escenarios:

1- No normalizar las imagenes
2- No aplicar `one-hot` sobre los targets
3- Utilizar `one-hot` sobre los targets pero no utilizar `categorical_cross_entropy`
4- Rendimiento utilizando `sigmoid` como funcion de activacion.
5- Rendimiento utilizando +/- neuronas.
6- Rendimiento utilizando +/- capas.

## Evaluacion


En su forma estandar, a traves del siguiente codigo:

```
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from utils.preprare_dataset import preprare_dataset
from utils.show_shape import show_shape
from tensorflow.keras import models
from tensorflow.keras import layers

(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=0.5, random_state=42, stratify=Y_test)

X_train, Y_train = preprare_dataset(X_train, Y_train)
X_val, Y_val = preprare_dataset(X_val, Y_val)
X_test, Y_test = preprare_dataset(X_test, Y_test)

show_shape(X_train, Y_train, "train_set")
show_shape(X_test, Y_test, "test_set")
show_shape(X_val, Y_val, "val_set")

net = models.Sequential()

net.add(layers.Dense(300, activation="relu", input_shape=(28*28,)))
net.add(layers.Dense(100, activation="relu"))
net.add(layers.Dense(50, activation="relu"))
net.add(layers.Dense(10, activation="softmax"))

net.compile(
    loss="categorical_crossentropy",
    optimizer="sgd",
    metrics=['accuracy', 'precision', 'recall']
)


history = net.fit(      X_train, 
                        Y_train, 
                        epochs=10, 
                        validation_data=(X_val, Y_val))

```

Obtuvimos los siguientes resultados:

```
Epoch 1/10
   1/1875 ━━━━━━━━━━━━━━━━━━━━ 18:41 599ms/step - accuracy: 0.1875 - loss: 2.3466 - precision: 0.0000e+00 - recall: 0.0000  21/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - accuracy: 0.1808 - loss: 2.3148 - precision: 0.0000e+00 - recall: 0.0000e+00 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.7178 - loss: 1.0550 - precision: 0.8909 - recall: 0.4703 - val_accuracy: 0.9190 - val_loss: 0.2781 - val_precision: 0.9423 - val_recall: 0.9016
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9242 - loss: 0.2594 - precision: 0.9460 - recall: 0.9060 - val_accuracy: 0.9396 - val_loss: 0.2019 - val_precision: 0.9564 - val_recall: 0.9266
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9424 - loss: 0.2018 - precision: 0.9558 - recall: 0.9293 - val_accuracy: 0.9500 - val_loss: 0.1689 - val_precision: 0.9606 - val_recall: 0.9412
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9531 - loss: 0.1631 - precision: 0.9641 - recall: 0.9437 - val_accuracy: 0.9572 - val_loss: 0.1401 - val_precision: 0.9672 - val_recall: 0.9508
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9601 - loss: 0.1367 - precision: 0.9681 - recall: 0.9526 - val_accuracy: 0.9584 - val_loss: 0.1320 - val_precision: 0.9663 - val_recall: 0.9532
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9653 - loss: 0.1145 - precision: 0.9732 - recall: 0.9591 - val_accuracy: 0.9668 - val_loss: 0.1127 - val_precision: 0.9738 - val_recall: 0.9608
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9706 - loss: 0.1003 - precision: 0.9762 - recall: 0.9657 - val_accuracy: 0.9684 - val_loss: 0.0979 - val_precision: 0.9743 - val_recall: 0.9636
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9741 - loss: 0.0894 - precision: 0.9790 - recall: 0.9694 - val_accuracy: 0.9708 - val_loss: 0.0931 - val_precision: 0.9756 - val_recall: 0.9670
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9776 - loss: 0.0773 - precision: 0.9820 - recall: 0.9738 - val_accuracy: 0.9718 - val_loss: 0.0928 - val_precision: 0.9758 - val_recall: 0.9668
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9809 - loss: 0.0662 - precision: 0.9844 - recall: 0.9779 - val_accuracy: 0.9706 - val_loss: 0.0917 - val_precision: 0.9758 - val_recall: 0.9674

```

### Sin normalizar 

```
Epoch 1/10
  22/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.1137 - loss: 135021685249350683946316201984.0000 - precision: 0.1  46/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.1142 - loss: nan - precision: 0.1033 - recall: 0.0484            1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.1012 - loss: nan - precision: 0.1008 - recall: 0.0030 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.1007 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0961 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 4/10
   1/1875 ━━━━━━━━━━━━━━━━━━━━ 43s 23ms/step - accuracy: 0.0000e+00 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+  25/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.0744 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0991 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0988 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 6/10
   1/1875 ━━━━━━━━━━━━━━━━━━━━ 41s 22ms/step - accuracy: 0.0000e+00 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+  24/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0808 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0978 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0973 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0975 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0965 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.0984 - loss: nan - precision: 0.0000e+00 - recall: 0.0000e+00 - val_accuracy: 0.0980 - val_loss: nan - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
```

Como vemos, los resultados sin normalizar son infinitamente peores. Cosa a tener muy en cuenta para proyectos futuros.

### Sin aplicar one-hot sobre el target

Obtenemos el siguiente error:

```
  File "/home/santiago/Escritorio/Aprendizaje ML/practicas/session9/FirstKerasPractice/main.py", line 33, in <module>
    history = net.fit(      X_train, 
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/santiago/Escritorio/Aprendizaje ML/practicas/session9/FirstKerasPractice/dep/lib/python3.12/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/santiago/Escritorio/Aprendizaje ML/practicas/session9/FirstKerasPractice/dep/lib/python3.12/site-packages/keras/src/backend/tensorflow/nn.py", line 653, in categorical_crossentropy
    raise ValueError(
ValueError: Arguments `target` and `output` must have the same rank (ndim). Received: target.shape=(32,), output.shape=(32, 10)
```


