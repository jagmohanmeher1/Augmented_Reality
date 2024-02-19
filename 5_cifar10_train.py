import tensorflow 
import matplotlib.pyplot as plt
from keras import *

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

image=x_train[2]
plt.imshow(image)
plt.show()

plt.figure(figsize=(10,10))
for i in range(5):
  plt.subplot(1,5,i+1)
  
  plt.imshow(x_train[i],cmap=plt.cm.binary)
  plt.xlabel(y_train[i])




# model = tf.keras.applications.VGG19(
# include_top=True,
# weights=None,
# input_tensor=None,
# input_shape=(32,32,3),
# pooling=None,
# classes=10,
# classifier_activation="softmax",
# )

# model.summary()

# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# history = model.fit(x_train, y_train, epochs = 30, batch_size = 100, validation_data=(x_test,y_test))

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()