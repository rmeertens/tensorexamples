---
layout: post
title:  "Saving and loading models in tensorflow"
date:   2020-07-23 10:52:17 +0200
categories: tensorflow saving loading models
---

# Saving and loading models in tensorflow
In this post I will show you how to go from training a simple neural network to saving and loading it. 

## Step 1: train a model 
For the sake of having a model to quantize we are building a simple classifier for MNIST digits. What model you use exactly doesn't really matter, so I will take an easy one here. Feel free to experiment and make the model better. 


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
model.fit(x_train, y_train,epochs=1)


```

    The TensorFlow version used in this tutorial is 2.2.0
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2611 - accuracy: 0.9252





    <tensorflow.python.keras.callbacks.History at 0x7fc0e7907d30>



## Option 1: using Keras save function
By default your model comes with a save function. From the documentation: 

This function saves the model to Tensorflow SavedModel or a single HDF5 file.

The savefile includes:
* The model architecture, allowing to re-instantiate the model.
* The model weights.
* The state of the optimizer, allowing to resume training exactly where you left off.

This allows you to save the entirety of the state of a model
in a single file.

Saved models can be reinstantiated via `keras.models.load_model`.
The model returned by `load_model` is a compiled model ready to be used
(unless the saved model was never compiled in the first place).




```python
save_path = "saved_models/save_load_method1" 
model.save(save_path)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: saved_models/save_load_method1/assets



```python
loaded_model = tf.keras.models.load_model(save_path)
loaded_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________


This method is particularly nice as the model remembers it's architecture. If you were not sure what number of layers you used in a saved model, this is a good method to save and restore it. 

## Option 2: using Keras save weights
This way you save only the layer weights. From the documentation: 

Either saves in HDF5 or in TensorFlow format based on the `save_format`
argument.

When saving in HDF5 format, the weight file has:
* `layer_names` (attribute), a list of strings (ordered names of model layers).
* For every layer, a `group` named `layer.name`
      * For every such layer group, a group attribute `weight_names`, a list of strings (ordered names of weights tensor of the layer).
      * For every weight in the layer, a dataset storing the weight value, named after the weight tensor.

When saving in TensorFlow format, all objects referenced by the network are saved in the same format as `tf.train.Checkpoint`, including any `Layer` instances or `Optimizer` instances assigned to object attributes. For networks constructed from inputs and outputs using `tf.keras.Model(inputs, outputs)`, `Layer` instances used by the network are tracked/saved automatically. For user-defined classes which inherit from `tf.keras.Model`, `Layer` instances must be assigned to object attributes, typically in the constructor. See the documentation of `tf.train.Checkpoint` and `tf.keras.Model` for details.

While the formats are the same, do not mix `save_weights` and
`tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should be loaded using `Model.load_weights`. Checkpoints saved using
`tf.train.Checkpoint.save` should be restored using the corresponding `tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over `save_weights` for training checkpoints.


```python
save_path = 'saved_models/save_load_method2'
model.save_weights(save_path)
```


```python
# Create the second model, but don't train it yet, so we can compare the performance to a trained model
second_model = get_model()

# Evaluate the performance of both models
NUM_EVALUATE_SAMPLES = 128
y_true = y_test[:NUM_EVALUATE_SAMPLES]
y_pred_trained = np.argmax(model.predict(x_test[:NUM_EVALUATE_SAMPLES]), axis=1)
y_pred_untrained = np.argmax(second_model.predict(x_test[:NUM_EVALUATE_SAMPLES]), axis=1)

# Calculate the accuracy and confusion matrics with sklearn
accuracy_score_trained_model = sklearn.metrics.accuracy_score(y_true, y_pred_trained)
accuracy_score_untrained_model = sklearn.metrics.accuracy_score(y_true, y_pred_untrained)
print("Accuracy trained", accuracy_score_untrained_model, "Accuracy untrained", accuracy_score_untrained_model)
```

    Accuracy trained 0.1796875 Accuracy untrained 0.1796875



```python
second_model.load_weights(save_path)
y_pred_untrained = np.argmax(second_model.predict(x_test[:NUM_EVALUATE_SAMPLES]), axis=1)
accuracy_score_untrained_model = sklearn.metrics.accuracy_score(y_true, y_pred_untrained)
print("After loading weights", accuracy_score_untrained_model)
```

    After loading weights 0.984375


As you can see the weights succesfully saved and loaded! One benefit of loading raw weights comes in when you are using custom layers in TensorFlow. In the past I experienced that the keras save function could fail or produce weird results. However, as we will see next, it is absolutely possible to save and load custom layers.

## Saving and loading custom layers
It is even possible to define custom layers in Tensorflow, and it is also possible to save and load these custom layers. 


```python
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        multiplied = tf.matmul(input, self.kernel)
        return tf.nn.relu(multiplied)
        return multiplied

custom_layer = MyDenseLayer(128)


model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        custom_layer,
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Show that we have a custom layer in there
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

# Show that this custom layer also achieves reasonable results
model.fit(x_train, y_train,epochs=1)

```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    my_dense_layer (MyDenseLayer (None, 128)               100352    
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,642
    Trainable params: 101,642
    Non-trainable params: 0
    _________________________________________________________________
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2619 - accuracy: 0.9252





    <tensorflow.python.keras.callbacks.History at 0x7fc0d44f79e8>




```python
name = 'saved_models/custom_model_saved'
model.save(name)
```

    INFO:tensorflow:Assets written to: saved_models/custom_model_saved/assets



```python
loaded_model = tf.keras.models.load_model(name)
loaded_model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    my_dense_layer (MyDenseLayer (None, 128)               100352    
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,642
    Trainable params: 101,642
    Non-trainable params: 0
    _________________________________________________________________


## Saving and loading models by name
Sometimes, when working on transfer learning, or when you have a solid backbone trained for your model you would like to reuse, you want to only load specific weights. 

Pay special attention to the fact that only topological loading (`by_name=False`) is supported when loading weights
from the TensorFlow format. To load weights by name you have to make sure the save_name ends with 'h5', or you define the save_format argument when saving the weights.


```python
model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28), name='input'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128,activation='relu', name='first_layer'),
        tf.keras.layers.Dense(128,activation='relu', name='head_1'),
        tf.keras.layers.Dense(10, activation='softmax', name='output_a')
    ])
model.summary()

save_name = 'saved_models/save_load_by_name.h5'
model.save_weights(save_name)

second_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28),name='input'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128,activation='relu', name='first_layer'),
        tf.keras.layers.Dense(128,activation='relu', name='head_2_1'),
        tf.keras.layers.Dense(128,activation='relu', name='head_2_2'),
        tf.keras.layers.Dense(10, activation='softmax', name='output_a')
    ])
second_model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    first_layer (Dense)          (None, 128)               100480    
    _________________________________________________________________
    head_1 (Dense)               (None, 128)               16512     
    _________________________________________________________________
    output_a (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 118,282
    Trainable params: 118,282
    Non-trainable params: 0
    _________________________________________________________________
    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    first_layer (Dense)          (None, 128)               100480    
    _________________________________________________________________
    head_2_1 (Dense)             (None, 128)               16512     
    _________________________________________________________________
    head_2_2 (Dense)             (None, 128)               16512     
    _________________________________________________________________
    output_a (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 134,794
    Trainable params: 134,794
    Non-trainable params: 0
    _________________________________________________________________



```python
second_model.load_weights(save_name, by_name=True)
```
