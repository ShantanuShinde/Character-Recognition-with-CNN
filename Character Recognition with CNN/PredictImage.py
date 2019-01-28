from keras.models import load_model


def Predict(X):
    model = load_model("model.h5")
    Y = model.predict(x=X)
    return Y


