---
layout: post
title:  "Quantization aware training for tensorflow-lite"
date:   2020-07-25 10:52:17 +0200
---

# Quantization aware training for tensorflow-lite
If you want to run your TensorFlow code on an embedded platform you want to quantize your neural network. Especially edge-tpu devices or raspberry pi devices are very suitable for running quantized code. However, when quantizing your neural network it's possible that the performance goes down. What helps against this is running quantization aware training, where your model already experiences a downgrade in performance and is able to adjust to it. 

In case you would like to know why and how this works, take a look at these articles: https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html, https://www.tensorflow.org/model_optimization/guide/quantization/training

## Option 1: using the tensorflow-model-optimization library
I think the easiest option is to use the tensorflow-model-optimization library. If you install this library you literally only have to add one line of code to quantize your Keras model during training. Below I created a very simple MNIST example. I personally think it's best to start training in a quantization aware way immediately. However, it's also possible to add the quantization layers at a later stage so you have both a normal model (maybe for a big server) and a quantization aware model (maybe for a raspberry pi or a robot). 


```
# Method 1: tensorflow-model-optimization library
!pip3 install --user --upgrade tensorflow-model-optimization

```

    Requirement already up-to-date: tensorflow-model-optimization in /root/.local/lib/python3.6/site-packages (0.3.0)
    Requirement already satisfied, skipping upgrade: dm-tree~=0.1.1 in /root/.local/lib/python3.6/site-packages (from tensorflow-model-optimization) (0.1.5)
    Requirement already satisfied, skipping upgrade: six~=1.10 in /usr/local/lib/python3.6/dist-packages (from tensorflow-model-optimization) (1.14.0)
    Requirement already satisfied, skipping upgrade: numpy~=1.14 in /usr/local/lib/python3.6/dist-packages (from tensorflow-model-optimization) (1.18.4)
    [33mWARNING: You are using pip version 20.1; however, version 20.1.1 is available.
    You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.[0m



```
import tensorflow as tf
import numpy as np
import sklearn.metrics
import tensorflow_model_optimization as tfmot

print("The TensorFlow version used in this tutorial is", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Transform the input into floating point inputs between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define a very simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print the summary of the object before making it quantization aware
model.summary()

# The one line to add quantization aware layers around the existing layers of your model. 
model = tfmot.quantization.keras.quantize_model(model)

# Print the summary of the object after making it quantization aware
model.summary()

# Compile the model. You always have to do that after quantizing your weights. 
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

# Train for one epoch to verify this works. 
model.fit(x_train, y_train,epochs=1, validation_data = (x_test, y_test))


```

    The TensorFlow version used in this tutorial is 2.2.0
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    quant_flatten_2 (QuantizeWra (None, 784)               1         
    _________________________________________________________________
    quant_dense_4 (QuantizeWrapp (None, 128)               100485    
    _________________________________________________________________
    quant_dense_5 (QuantizeWrapp (None, 10)                1295      
    =================================================================
    Total params: 101,781
    Trainable params: 101,770
    Non-trainable params: 11
    _________________________________________________________________
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.2556 - accuracy: 0.9277 - val_loss: 0.1357 - val_accuracy: 0.9600





    <tensorflow.python.keras.callbacks.History at 0x7f6fc1fa7048>



## Option 2: using tf.quantization.fake_quant_with_min_max_args

If you want a bit more control over your fake quantization you can use the `tf.quantization.fake_quant_with_min_max_args` function. This function immitates the quantization with parameters you can enter yourself. If you would like to have specific ranges, you can specify these already. What's also nice is that you can specify the number of bits with which you are going to perform inference. You might want to do 8-bit inference for things like the NVIDIA Jetson, or even 4-bit inference if you have access to Turing TensorCores. 

What's also nice is that you can choose what to quantize and what not. Perhaps you discovered that the backbone should be quantized, but that you want to keep your head unquantized. Then you don't have to use quantization aware training... 



```
# Create a simple quantization aware dense layer
def quantization_aware_dense_layer(num_output_neurons, activation, x):
    x = tf.quantization.fake_quant_with_min_max_args(x)
    x = tf.keras.layers.Dense(num_output_neurons,activation=activation)(x)
    return x

# Define a very simple model using our new layer
input_layer = tf.keras.layers.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(input_layer)
x = quantization_aware_dense_layer(128, 'relu',x)
output_layer = quantization_aware_dense_layer(10, 'softmax',x)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
model.summary()

```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         [(None, 28, 28)]          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    tf_op_layer_FakeQuantWithMin [(None, 784)]             0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    tf_op_layer_FakeQuantWithMin [(None, 128)]             0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________



```
# Compile the model. 
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

# Train for one epoch to verify this works. 
model.fit(x_train, y_train,epochs=1, validation_data = (x_test, y_test))

```

    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2606 - accuracy: 0.9256 - val_loss: 0.1423 - val_accuracy: 0.9577





    <tensorflow.python.keras.callbacks.History at 0x7f6fd763a518>



### Removing these fake quantization layers again

Some tools can't really handle fake quantization layers. Luckily, especially for sequential models, it's easy to remove these layers again. In this case you can simply define a new sequential model, and leave out the fake quantization layers with a simple string filter. 


```
new_model = tf.keras.Sequential([layer for layer in model.layers if 'FakeQuantWithMinMaxArgs' not in layer.name])

new_model.summary()

# You have to re-compile the model. Note that this does NOT reset the weights!
new_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

new_model.evaluate(x_test, y_test)
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_3 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________
    313/313 [==============================] - 1s 2ms/step - loss: 0.1423 - accuracy: 0.9580





    [0.14229463040828705, 0.9580000042915344]


