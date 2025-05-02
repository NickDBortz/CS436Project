# CS436Project
# Final Project for CS436

* Author: Nick Bortz
* Class: CS436
* Semester: Spring 2025

## Overview

This is a program is a training script for training a language model to replicate some desired type of data.
For the example data, I used a series of movie scripts from the Marvel movie franchise.

## Setup and Using

To setup the environment, ensure you have a version of python 3 downloaded on your computer
Then, install numpy and tensorflow like so:
```
pip install numpy tensorflow
```

To change any configurations, open the config.py file and edit the parameters as needed.

Now open a terminal in this repository's folder.
Then, if your device has a usable graphics card, run the train_gpu file:
```
python ./train_gpu
```
If this doesn't work, run the train_cpu file:
```
python ./train_cpu
```

These will take some time to run, but will save the model to a file for use later

When you want to test the model, again run test_gpu if you have a gpu:
```
python ./test_gpu
```
Or test_cpu otherwise:
```
python ./test_cpu
```