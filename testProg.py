import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

## loading data set from keras
mnist=keras.datasets.mnist
(X_train_full, Y_train_full),(X_test, Y_test)=mnist.load_data()
X_train_full.shape
X_test.shape
X_train_full[0]

## showing images in data
fig, axes=plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
a=0
for i in range(3):
    for j in range(3):
        axes[i, j].imshow(X_train_full[a], cmap=plt.get_cmap('gray'))
        a=a+1
plt.show()

#dividing data to normalize(informal) for change by diving pixel number 255

X_valid, X_train=X_train_full[:5000]/255, X_train_full[5000:]/255
Y_valid, Y_train=Y_train_full[:5000], Y_train_full[5000:]
X_test= X_test/255

class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

print(class_names[Y_train[1]])
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.show()

model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation="softmax"))

print(model.summary())
print(model.layers)

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
print(X_train.shape)

history=model.fit(X_train, Y_train, epochs=30, validation_data=(X_valid, Y_valid), batch_size=32)

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

print(model.evaluate(X_test, Y_test))

y_prob=model.predict(X_test)
y_classes=y_prob.argmax(axis=-1)
print(y_classes)

confusion_matrix=tf.math.confusion_matrix(Y_test, y_classes)

fig2=sb.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Greens')
fig2.set_xlabel('Predict Labels')
fig2.set_ylabel('True Labels')
fig2.set_title('Confusion Matrix')
fig2.xaxis.set_ticklabels(class_names)
fig2.yaxis.set_ticklabels(class_names)
fig2.figure.set_size_inches(10, 10)

plt.show()



