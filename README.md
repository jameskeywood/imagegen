# geno

## Summary
- Machine learning algorithm that can create an image from a collection of photographs.
- Hoping to create a GAN (Generative Adversarial Network) machine learning algorithm.

##

![GANs](https://github.com/jameskeywood/geno/blob/main/research/GANs.png?raw=true)

## Forks
Instagram account to post the generated images.
- Will post daily images based on collections such as landscapes.
- Collect information such as likes and if above a threshold add to collection.
- Create more and more abstract imagery.

Search function to create a collection based on internet images.
- User will be able to type in a search term.
- Script will scrape internet for collection of related images.
- Algorithm will then produce an image based on search.

## Resources
- Image generation: https://towardsdatascience.com/image-generation-in-10-minutes-with-generative-adversarial-networks-c2afc56bfa3b
- GAN overview: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
- Building a simple GAN: https://blog.paperspace.com/implementing-gans-in-tensorflow/
- Deep convolutional GAN: https://www.tensorflow.org/tutorials/generative/dcgan
- Load images: https://www.tensorflow.org/tutorials/load_data/images
- Datasets similar to MNIST: https://analyticsindiamag.com/mnist/
- Flower dataset: https://medium.com/@nutanbhogendrasharma/tensorflow-image-classification-with-tf-flowers-dataset-e36205deb8fc
- Keras Image data preprocessing: https://keras.io/api/preprocessing/image/#imgtoarray-function
- PIL resize images: https://www.geeksforgeeks.org/python-pil-image-resize-method/
- Face GAN: https://betterprogramming.pub/making-a-face-gan-with-tensorflow-23b4b79b4de7
- Dog Dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/

## Datasets
Put the expected archive in the directory of the script, and it will be extracted and used.
- Face dataset: http://vis-www.cs.umass.edu/lfw/lfw.tgz
- Flower dataset: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- Dog dataset: vision.stanford.edu/aditya86/ImageNetDogs/images.tar

## Versions
- TensorFlow v2.6.0
- CUDA v11.0.2
- cuDDN v8.2.1
