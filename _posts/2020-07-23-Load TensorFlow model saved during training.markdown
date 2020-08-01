---
layout: post
title:  "Loading TensorFlow models saved during training"
date:   2020-07-23 10:52:17 +0200
---
# Loading TensorFlow models saved during training
When you are training large models on large datasets you normally do not want to wait and drink coffee while models are training. After starting a training you frequently move on to a next experiment, and check the results later. This tutorial shows you how to save models during training, and how you can later load and evaluate these models. 


```python
import tensorflow as tf
import numpy as np
import sklearn.metrics

print("The TensorFlow version used in this tutorial is", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Transform the input into floating point inputs between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

def get_model():
    # Define a very simple model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
model = get_model()

# Compile and train the model for one epoch... It's only to have something trained, not get the best score
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)
```

    The TensorFlow version used in this tutorial is 2.2.0


## Add the right callback
To save a model during training you have to add a callback to the ``model.fit`` function. You use a ``ModelCheckpoint`` callback. From the documentation, options this callback provides include:
* Whether to only keep the model that has achieved the "best performance" so far, or whether to save the model at the end of every epoch regardless of performance.
* Definition of 'best'; which quantity to monitor and whether it should be maximized or minimized.
* The frequency it should save at. Currently, the callback supports saving at the end of every epoch, or after a fixed number of training batches.
* Whether only weights are saved, or the whole model is saved.

Another thing I like to do is adjust the name of the model to save. It gives you many options which help you later selecting the right model to load. From the documentation, `filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.



```python
filepath = 'saved_models/saved_model_checkpoint'
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='train_loss', verbose=1, save_best_only=False)]

model.fit(x_train, y_train,epochs=5, callbacks=callbacks)
```

    Epoch 1/5
    1856/1875 [============================>.] - ETA: 0s - loss: 0.2617 - accuracy: 0.9249
    Epoch 00001: saving model to saved_models/saved_model_checkpoint
    INFO:tensorflow:Assets written to: saved_models/saved_model_checkpoint/assets
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2607 - accuracy: 0.9251
    Epoch 2/5
    1863/1875 [============================>.] - ETA: 0s - loss: 0.1172 - accuracy: 0.9653
    Epoch 00002: saving model to saved_models/saved_model_checkpoint
    INFO:tensorflow:Assets written to: saved_models/saved_model_checkpoint/assets
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.1171 - accuracy: 0.9653
    Epoch 3/5
    1854/1875 [============================>.] - ETA: 0s - loss: 0.0793 - accuracy: 0.9763
    Epoch 00003: saving model to saved_models/saved_model_checkpoint
    INFO:tensorflow:Assets written to: saved_models/saved_model_checkpoint/assets
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0794 - accuracy: 0.9762
    Epoch 4/5
    1872/1875 [============================>.] - ETA: 0s - loss: 0.0592 - accuracy: 0.9815
    Epoch 00004: saving model to saved_models/saved_model_checkpoint
    INFO:tensorflow:Assets written to: saved_models/saved_model_checkpoint/assets
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0593 - accuracy: 0.9815
    Epoch 5/5
    1875/1875 [==============================] - ETA: 0s - loss: 0.0449 - accuracy: 0.9862
    Epoch 00005: saving model to saved_models/saved_model_checkpoint
    INFO:tensorflow:Assets written to: saved_models/saved_model_checkpoint/assets
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0449 - accuracy: 0.9862





    <tensorflow.python.keras.callbacks.History at 0x7f34b808cac8>



## Loading the model
To verify that the model has been saved and loaded correctly we test the accuracy of the model on the MNIST test-set. 


```python
loaded_model = tf.keras.models.load_model(filepath)
```


```python
y_pred = np.argmax(loaded_model(x_test), axis=1)

# Calculate the accuracy and confusion matrics with sklearn
accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
print("The loaded model has a test accuracy of ", accuracy_score)
```

    The loaded model has a test accuracy of  0.9764



```python

```
