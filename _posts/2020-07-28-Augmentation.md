---
layout: post
title: "Augmentation"
--- 
# Augmenting your input in TensorFlow

One good way to improve the performance of your neural network is by using
augmentations. There are several ways to augment images, so I will discuss some
of them in this article. First of all it would be nice to have some images we
can apply our augmentations on. For this article I will use the cifar10 dataset,
as it is included with Keras, is small to download, and has three image
channels. 

**In [10]:**

{% highlight python %}
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# For memory purposes I will subsample the train dataset. 
SAMPLE_TRAINING = 100
x_subset, y_subset = x_train[:SAMPLE_TRAINING] / 255.0, y_train[:SAMPLE_TRAINING] 
dataset = tf.data.Dataset.from_tensor_slices(x_subset)

{% endhighlight %}

**In [11]:**

{% highlight python %}
print(dataset)
{% endhighlight %}

    <TensorSliceDataset shapes: (32, 32, 3), types: tf.float64>


**In [12]:**

{% highlight python %}
import matplotlib.pyplot as plt
for index, inputs in enumerate(dataset.as_numpy_iterator()):
    if index >= 3: 
        break
    plt.imshow(inputs)
    plt.show()
    
{% endhighlight %}


 ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_3_0.png) 


  
 ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_3_1.png) 


  
 ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_3_2.png) 

  
 ## How to augment
 There are multiple ways which are normally beneficial to your training
 algorithm. Think about rotating and flipping the images, or changing the
 colours. Luckily TensorFlow already has many of these algorithms included: 

 **In [13]:**

 {% highlight python %}
 image_tensor = tf.convert_to_tensor(x_subset[0,...], dtype=np.float32)
 plt.imshow(image_tensor)
 {% endhighlight %}




     <matplotlib.image.AxesImage at 0x7fe5ba968a20>



  
 ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_5_1.png) 


**In [14]:**

{% highlight python %}
plt.imshow(tf.image.rot90(image_tensor, 3).numpy())
{% endhighlight %}




    <matplotlib.image.AxesImage at 0x7fe5ba8cab00>



 
![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_6_1.png) 

 
### Augmenting the color spaces
 

**In [18]:**

{% highlight python %}

{% endhighlight %}

**In [22]:**

{% highlight python %}
hue = tf.image.random_hue(image_tensor, 0.2)
saturation = tf.image.random_saturation(image_tensor, 0.5, 1.5)
brightness = tf.image.random_brightness(image_tensor, 0.2)
contrast = tf.image.random_contrast(image_tensor, 0.2, 0.5)

for image in [hue, saturation, brightness, contrast]:
    plt.imshow(image)
    plt.show()
{% endhighlight %}

 
![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_9_0.png) 


 
![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_9_1.png) 


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


 
![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_9_3.png) 


 
![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_9_4.png) 

 
Hopefully you can still recognize the image, although the colours are jumbled a
bit.
One thing you can already see is that it's possible that your image pixel values
are outside of the normal range. To fix this you can call the function
`tf.clip_by_value(tensor, 0, 1)`.
 

 ## Applying the augmentations on your `tf.data.Dataset`
 Now that you know what functions are available the next step is to map these
 functions to your dataset. The Dataset object itself supports this with it's map
 function. However, you need to wrap the function you map inside a new function
 which contains all your parameters.

 If you get a message like `InvalidArgumentError: Length for attr 'output_shapes' of 0 must be at least minimum 1` it means that you
 forgot to return the tensor after applying an operation.


  **In [8]:**

  {% highlight python %}
  def augment_hue(tensor):
      return tf.image.random_hue(tensor, 0.2)
  def augment_brightness(tensor):
      return tf.image.random_brightness(tensor, 0.2)

  augmented_dataset = dataset.map(augment_hue)
  augmented_dataset = dataset.map(augment_brightness)

  {% endhighlight %}
   
  You can even make the code a bit shorter and use the function directly, but then
  you have to map it into a lambda function. 

  **In [9]:**

  {% highlight python %}
  augmented_dataset = dataset.map(lambda x: tf.image.random_hue(x, 0.2))
  {% endhighlight %}
   
  Now you can also stack functions on top of each other to get super-
  augmentations: 

  **In [23]:**

  {% highlight python %}
  augmented_dataset = dataset
  augmented_dataset = augmented_dataset.map(augment_hue)
  augmented_dataset = augmented_dataset.map(augment_brightness)
  augmented_dataset = augmented_dataset.map(lambda x: tf.image.random_saturation(x, 0.5, 1.5))
  augmented_dataset = augmented_dataset.map(lambda x: tf.image.random_contrast(x, 0.2, 0.5))

  # Don't forget to make the pixel values sane again
  augmented_dataset = augmented_dataset.map(lambda x: tf.clip_by_value(x, 0.0, 1.0))
  {% endhighlight %}

  **In [24]:**

  {% highlight python %}
  for index, inputs in enumerate(augmented_dataset.as_numpy_iterator()):
      if index >= 3: 
          break
      plt.imshow(inputs)
      plt.show()
  {% endhighlight %}

   
  ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_17_0.png) 


   
  ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_17_1.png) 


   
  ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_17_2.png) 

   
  Images are looking great now, but the amount of augmentations is maybe a bit too
  much now.

  If you get the error `TypeError: 'Tensor' object is not callable` you did not
  wrap the functions in a lambda/callable function. 

  **In [37]:**

  {% highlight python %}
  def maybe_augment(tensor, function):
      APPLY_AUGMENTATIONS_PROBABILITY = 0.5
      return tf.cond(tf.random.uniform([], 0, 1) > APPLY_AUGMENTATIONS_PROBABILITY, 
              true_fn=lambda: function(tensor), # apply the function
              false_fn=lambda: tensor) # return the tensor

  augmented_dataset = dataset
  augmented_dataset = augmented_dataset.map(lambda x: maybe_augment(x, augment_hue))
  augmented_dataset = augmented_dataset.map(lambda x: maybe_augment(x, augment_brightness))
  {% endhighlight %}

  **In [41]:**

  {% highlight python %}
  for index, inputs in enumerate(augmented_dataset.as_numpy_iterator()):
      if index >= 3: 
          break
      plt.imshow(inputs)
      plt.show()
  {% endhighlight %}

   
  ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_20_0.png) 


   
  ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_20_1.png) 


   
  ![png](https://raw.githubusercontent.com/rmeertens/tensorexamples/master/images/augmentation_20_2.png) 

   
  As you can see images are looking a bit more recognisable now. 

  **In [None]:**

  {% highlight python %}

  {% endhighlight %}

