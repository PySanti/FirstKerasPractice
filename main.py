from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split


(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
X_test, Y_test, X_val, Y_val = train_test_split(X_test, Y_test, train_size=0.5, random_state=42, stratify=Y_test)

