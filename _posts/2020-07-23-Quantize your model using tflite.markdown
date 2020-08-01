---
layout: post
title:  "Quantization in tensorflow-lite"
date:   2020-07-23 10:52:17 +0200
---

# Quantization in tensorflow-lite
If you want to run your TensorFlow code on an embedded platform you want to quantize your neural network. Especially edge-tpu devices or raspberry pi devices are very suitable for running quantized code. 

In this post I will show you how to go from training a simple neural network to running a quantized version of this network. 

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

# Define a very simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model for one epoch... It's only to have something trained, not get the best score
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)
model.fit(x_train, y_train,epochs=1)


```

    The TensorFlow version used in this tutorial is 2.2.0
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.2602 - accuracy: 0.9253





    <tensorflow.python.keras.callbacks.History at 0x7f0f8a5cfac8>



## Step 2: save the model 

Imagine that you finally trained the perfect image classification algorithm! Naturally you save it to be able to load it later, or evaluate in different environments. 


```python
saved_model_dir = 'saved_models/saved_quantization_model'
model.save(saved_model_dir)

```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: saved_models/saved_quantization_model/assets


## Step 3: Convert the model to a tflite model

Now is the moment to take your model and turn it into a calibrated model. To calibrate your model you show the neural network multiple possible inputs. In the background the activations are calculated to get a feeling for the spread of the activations. This is why it's important to make the calibration dataset representative for your use case. 

If you during this step never show an image of a certain number/class it is possible that your model in production has issues recognizing this number/class. For now I just take a low amount of samples to speed up the process. 

In this tutorial we are making all inference run in int8 values. The range of int8 is very low, there are now only 255 options for each activation in each layer. This will let you lose a lot of precision in your neural network. Always make sure in an evaluation step that your model is still able to detect everything. 


```python
num_calibration_steps = 150

# Load the model we saved in step 2, and set the optimizations
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Make a way to load a representativce dataset. 
def representative_dataset_gen():
    sample_per_calibration_step = 8
    for calib_index in range(num_calibration_steps):
        # Get sample input data as a numpy array. You can either randomly select, or have a fixed calibration dataset. 
        yield [[x_train[sample_per_calibration_step,...]]]
    
# Choose the inference input and output, and set the supported ops.     
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

```

## Step 4: save the tflite model
So far your work has been on a big computer. Now it's possible to transport your model to a low power device, such as your raspberry pi. 


```python
tflite_quant_model_save_path = 'saved_models/model_quant.tflite'
NUM_EVALUATE_SAMPLES = 128

open(tflite_quant_model_save_path, "wb").write(tflite_quant_model)


```




    104848



## Step 5: load the tflite model and run inference. 
This is all code which is necessary to run inference on your embedded device :) 
Once you have this all looks really simple, right? 


```python
# Load quantized TFLite model
tflite_interpreter_quant = tf.lite.Interpreter(model_path=tflite_quant_model_save_path)

# Set the input and output tensors to handle a small batch of images to evaluate on
input_details = tflite_interpreter_quant.get_input_details()
output_details = tflite_interpreter_quant.get_output_details()
tflite_interpreter_quant.resize_tensor_input(input_details[0]['index'], (NUM_EVALUATE_SAMPLES, 28, 28))
tflite_interpreter_quant.resize_tensor_input(output_details[0]['index'], (NUM_EVALUATE_SAMPLES, 10))
tflite_interpreter_quant.allocate_tensors()

# Run inference on the first set of test images
tflite_interpreter_quant.set_tensor(input_details[0]['index'], x_test[:NUM_EVALUATE_SAMPLES, ...])
tflite_interpreter_quant.invoke()

tflite_q_model_predictions = tflite_interpreter_quant.get_tensor(output_details[0]['index'])
print("\nPrediction results shape:", tflite_q_model_predictions.shape)
```

    
    Prediction results shape: (128, 10)


## Step 6: evaluate your model performance
As I mentioned in a previous step your model performance might have gone down. It is important to evaluate both your general performance as well as your performance for specific classes. Things which are rare or very similar to other objects might not be represented very well anymore. 


```python
# Get the true values and the predictions for the first N samples
y_true = y_test[:NUM_EVALUATE_SAMPLES]
y_pred = np.argmax(tflite_q_model_predictions, axis=1)

# Calculate the accuracy and confusion matrics with sklearn
accuracy_score = sklearn.metrics.accuracy_score(y_true, y_pred)
confusion_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)

# Print the accuracy score and confusion matrix
print("Accuracy score:", accuracy_score)
print("Confusion matrix")
print(confusion_mat)

```

    Accuracy score: 0.9609375
    Confusion matrix
    [[10  0  0  0  0  0  0  0  0  0]
     [ 0 15  0  0  0  0  0  0  0  0]
     [ 0  0  9  1  0  0  0  0  0  0]
     [ 0  0  0 12  0  0  0  0  0  0]
     [ 0  0  0  0 20  0  0  0  0  0]
     [ 0  0  0  1  0  8  1  0  0  0]
     [ 0  0  0  0  0  0 12  0  0  0]
     [ 0  0  0  0  1  0  0 18  0  0]
     [ 0  0  0  0  0  0  0  0  3  0]
     [ 0  0  0  0  1  0  0  0  0 16]]


## Conclusion
Now you are able to run inference on low-power embedded devices! Enjoy!


```python

```
