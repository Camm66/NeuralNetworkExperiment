# A Simple Neural Network
This program creates and trains a simple neural network that can recognize handwritten numbers by interpreting some simple input data.

The code and techniques utilized in this program were adapted from the book, _'Make Your Own Neural Network'_ by Tariq Rashid.

## Instructions to run this program:
From the command line:
```
> python NeuralNetwork.py
```

## The Training and Testing Data
The data used on this network consists of csv files which can be found [here](https://pjreddie.com/projects/mnist-in-csv/), or directly from the links below.
##### [Training Set](https://pjreddie.com/media/files/mnist_train.csv)
##### [Testing Set](https://pjreddie.com/media/files/mnist_test.csv)

## Files needed to run this program:
eg:
1. NeuralNetwork.py
2. mnist_train.csv
3. mnist_test.csv
4. [UserFile].png/jpg -- (must be 28px by 28px)

## Testing user provided images
The network can query user provided images of handwritten numbers via the `testImage()` method.

* The method should be provided with a 28px by 28px .png file of the handwritten number.

The requesite code changes are provided below in square brackets:

1.  Provide the correct answer:
```
image_num = [ImageNumberGoesHere]
```

2. Provide the .png file to query:
```
img_array = imageio.imread("[ImageFileGoesHere]", as_gray=True)
```

## License
License information is available in the LICENSE.txt file.
