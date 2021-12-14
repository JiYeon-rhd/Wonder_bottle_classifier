import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader


def showSample(data):

    plt.figure(figsize=(10,10))
    for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)
        plt.xlabel(data.index_to_label[label.numpy()])
    plt.show()

def get_label_color(val1, val2):
    if val1 == val2:
        return 'black'
    else:
        return 'red'


def showPredicted(data):

    plt.figure(figsize=(20, 20))
    predicts = model.predict_top_k(data)
    for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(100)):
        ax = plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)

        predict_label = predicts[i][0][0]
        color = get_label_color(
            predict_label,
            data.index_to_label[label.numpy()]
            )
        ax.xaxis.label.set_color(color)
        plt.xlabel('Predicted: %s' % predict_label)
    plt.show()


image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)

print("======================================")
print("image_path=",image_path)
print("======================================")

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

showSample(data)

print("======================================")
print("The number of Data =", len(data))
print("The number of Train data =", len(train_data))
print("The number of Test Data =", len(test_data))
print("======================================")

model = image_classifier.create(train_data)

loss, accuracy = model.evaluate(test_data)
showPredicted(test_data)

model.export(export_dir='.', with_metadata=False)