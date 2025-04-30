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
    loss="crossentropy",
    optimizer="sgd",
    metrics=['accuracy', 'precision', 'recall']
)


history = net.fit(      X_train, 
                        Y_train, 
                        epochs=10, 
                        validation_data=(X_val, Y_val))

