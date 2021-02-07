import tensorflow as tf


def segmentation_to_image(pred):
    img = tf.argmax(pred, axis=-1)
    img = img[..., tf.newaxis]
    return tf.keras.preprocessing.image.array_to_img(img)
