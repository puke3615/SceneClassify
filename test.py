import keras

model = keras.applications.VGG16(weights=None)
# print model.summary()
# print model.get_config()
weights = model.get_weights()
print weights