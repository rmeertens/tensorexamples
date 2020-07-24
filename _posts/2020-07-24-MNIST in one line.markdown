---
layout: post
title:  "Training and evaluating MNIST in one line"
date:   2020-07-24 10:52:17 +0200
categories: tensorflow train MNIST one python
---
# Training and evaluating MNIST in one line
Python is a fantastic language which both allows you to make self-documenting understandable code, and allows you to make short powerful statements. I wanted to try to train and evaluate a neural network on the MNIST dataset. In this post I will guide you through my progress on achieving this. 

## Cheating
First of all I wanted to totally admit that I'm cheating. It is possible to import things in the same place where you use it, but the end result would look even more terrible than it looks now. For the rest of this post, assume that Python would be extra-batteries-included, with tensorflow, numpy, and sklearn already included. 


```python
import tensorflow as tf
import numpy as np
import sklearn.metrics

print("The TensorFlow version used in this tutorial is", tf.__version__)
```

    The TensorFlow version used in this tutorial is 2.2.0


## Initial code
It is already quite amazing that it is so easy nowadays to train a computer to recognize written numbers. Only ten years ago I spent weeks in a neural network course implementing backpropagation and data loading. This program is already short, clear, and has all functionality I would like: 


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Transform the input into floating point inputs between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Compile and train the model for one epoch... It's only to have something trained, not get the best score
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train,epochs=1)


y_pred = np.argmax(model(x_test), axis=1)

# Calculate the accuracy and confusion matrics with sklearn
accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
print("The loaded model has a test accuracy of ", accuracy_score)
```

    1875/1875 [==============================] - 3s 2ms/step - loss: 0.4678 - accuracy: 0.8778
    The loaded model has a test accuracy of  0.9142


## Simplify the metrics
Simplifying the metrics is an easy step. Instead of three lines of code I can reduce it to one line


```python
print("The loaded model has a test accuracy of ", 
      sklearn.metrics.accuracy_score(y_test, 
                                     np.argmax(model(x_test), axis=1)))
```

    The loaded model has a test accuracy of  0.9142


## Compiling and fitting
This was actually the most challenging part of this challenge. The `model.compile` function acts on the object, and does not return itseld, making a `.compile().fit().predict()` chain impossible. I tried using Python's `map` functions, but in the end realized putting function calls in `print` also changes the object I would like to change. Here I already added the model accuracy evaluation to it. 


```python
print(model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']), 
      model.fit(x_train, y_train,epochs=1), 
      "\nThe loaded model has a test accuracy of ", 
      sklearn.metrics.accuracy_score(y_test, np.argmax(model(x_test), axis=1)))

```

    1875/1875 [==============================] - 3s 2ms/step - loss: 0.3025 - accuracy: 0.9150
    None <tensorflow.python.keras.callbacks.History object at 0x7f275c0474a8> 
    The loaded model has a test accuracy of  0.921


## Defining the model somewhere
So far I assumed model would be available to me. However, I need to define it on the same line. I decided to use a list-comprehension to define the model and immediately act on it. You can now even define multiple models to evaluate all of them in one line ;). 


```python
[print(model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']), 
      model.fit(x_train, y_train,epochs=1), 
      "\nThe loaded model has a test accuracy of ", 
      sklearn.metrics.accuracy_score(y_test, np.argmax(model(x_test), axis=1)))
    for model in [tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])]]
```

    1875/1875 [==============================] - 3s 2ms/step - loss: 0.4684 - accuracy: 0.8770
    None <tensorflow.python.keras.callbacks.History object at 0x7f275c037e48> 
    The loaded model has a test accuracy of  0.9151





    [None]



## Loading the data
The last step is loading all data. Here I got pretty stuck for a while, as I initially trained on both train and test set. In the end I ended up loaded the data 4 times, to be able to do everything in one line. 

This bring us to the completed one line training of MNIST: 


```python
[[print(model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']), 
      model.fit(x_train/255.0, y_train,epochs=1), 
      "\nThe loaded model has a test accuracy of ", 
      sklearn.metrics.accuracy_score(y_test, np.argmax(model(x_test/255.0), axis=1)))
    for model in [tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])]] for x_train, y_train, x_test, y_test in [(tf.keras.datasets.mnist.load_data()[0][0], 
                                                   tf.keras.datasets.mnist.load_data()[0][1],
                                                   tf.keras.datasets.mnist.load_data()[1][0], 
                                                   tf.keras.datasets.mnist.load_data()[1][1])]]
```

    1875/1875 [==============================] - 3s 2ms/step - loss: 0.4707 - accuracy: 0.8763
    None <tensorflow.python.keras.callbacks.History object at 0x7f274c6ad438> 
    The loaded model has a test accuracy of  0.9161





    [[None]]



## Conclusion
Here we go, training and evaluating a character recognition program in only one line! I showed this program to some other people, who had ideas on how to make it either more readable, or shorter than what I have. If you are such a person, please consider sharing the idea on Twitter, LinkedIn, or on your own blog :). 


```python

```
