import tensorflow as tf
import matplotlib.pyplot as plt


def segmentation_to_image(pred):
    img = tf.argmax(pred, axis=-1)
    img = img[..., tf.newaxis]
    return tf.keras.preprocessing.image.array_to_img(img)

        
def predict_tf(model):
    def predict_func(sample):
        pred = model.predict(tf.expand_dims(sample[0], axis=0))
        return sample[0], pred[0]
    
    return predict_func


def display_dataset(ds, pred_func):
    for sample in ds:
        imgs = pred_func(sample)
        fig, axes = plt.subplots(1, len(imgs))
        
        for ax, img in zip(axes, imgs):
            if img.shape[-1] != 3:
                img = segmentation_to_image(img)
                
            ax.imshow(img)
