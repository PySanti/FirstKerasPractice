from keras.utils import to_categorical

def preprare_dataset(X_data, Y_data):
    X_data = X_data.reshape(X_data.shape[0], 28*28)
    X_data = X_data.astype('float32') / 255
    #Y_data = to_categorical(Y_data, num_classes=10)
    return [X_data, Y_data]


