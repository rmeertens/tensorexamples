---
layout: post
title:  "Running TensorBoard in a Dockerfile"
date:   2020-08-14 15:52:17 +0200
---
# Running TensorBoard in a Dockerfile
Tenosrboard is one of the most useful tools you can use when training deep neural networks. If you run your TensorFlow experiments from a Dockerfile it's also possible to run Tensorboard in the same dockerfile. Things to keep in mind are: 
* Make sure your port 6006 is open. If you are running a dockerfile directly, either with a locally defined dockerfile or using an image from the DockerHub, you have to pass the argument `-p 6006:6006` to the command line. You will normally pass `-p 8888:888` already, so simply extend that command so you get `-p 8888:888 -p 6006:6006`. 
* If you are using docker-compose you should also add port 6006 to the ports in your `docker-compose.yml` file. 
* You don't have to make a separate Dockerfile for TensorBoard. I used to make this mistake for many years, where I defined a separate Dockerfile for Tensorboard. Turns out you can start tensorboard from a Jupyter notebook or command line later. To do this in a jupyter notebook you have to use the `load_ext tensorboard` command in a cell to start using it. Then visualise TensorBoard in a Jupyter notebook cell using the `%tensorboard --logdir logs --bind_all` command. You only have to execute this command once. Every next time you use this command you will get the `Reusing TensorBoard on port 6006` message, which will just show your current existing tensorboard session. For me killing tensorboard doesn't work, and it required me to restart the whole docker container. 
* If you get `127.0.0.1 didn’t send any data.` you likely forgot the `--bind_all` command at some point in time. As I mention at my previous point: I simply had to restart my docker container by killing it. Even using the python tools by importing tensorboard (using `from tensorboard import notebook`) I could not make it work. Normally you might be able to run it with a different port, but then you would have to open that one first in your docker container. 

## TensorBoard example
TensorFlow has a great example on how to use their tool, which is described here: https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks. I decided not to copy paste their example code so you won't have to deal with an outdated piece of code in a few years. 


```python

```
