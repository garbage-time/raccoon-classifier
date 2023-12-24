# Raccoon Image Classifier

#### Summary

This is an image classifier built with `Tensorflow`. The model can classify whether something is a raccoon, dog, cat, or something else. This repo is packaged into a `Streamlit` app can be on my blogpost here:

* https://garbage-time.github.io/posts/raccoon-classifier/

The model is still a work-in-progress as it tends to over-classify images as 'dogs', due to sample inadequacies.

#### Contents

* `model.py` fits the model. If the user wishes to train their own model, they will have to construct their own dataset. This script is a modified variant of `tensorflow`'s image classification tutorial found here: https://tensorflow.org/tutorials/images/classification
* `app.py` creates the `streamlit` applicaton.
* `raccoon.tflite` is the trained model. You may load it to try out the classifier outside of the application.
