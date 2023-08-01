from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import json

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    image = np.asarray(img)
    processed_image = np.expand_dims(process_image(image), axis=0)
    preds = model.predict(processed_image)
    classes = np.argsort(preds[0])[-top_k:]+1
    probs = np.sort(preds[0])[-top_k:]
    return probs, classes

def process_image(image):
    img_tf = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img_tf, (224,224))
    img /= 255
    return img.numpy()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Predict the flower type from an image of it.')
    parser.add_argument('image_path', type=str,
                        help='path to file that contains the flower image to be classified.')
    parser.add_argument('--category_names', type=str, default='label_map.json',
                        help='path to file that maps class numbers to flower names.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='the k flowers with the highest probability will be printed.')
    args = parser.parse_args()

    filepath='./FlowerNet.h5'
    model=tf.keras.models.load_model(filepath, custom_objects={'KerasLayer':hub.KerasLayer})
    probs, classes = predict(args.image_path, model, args.top_k)
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        cls_names = []
    for k in range(args.top_k):
        cls_names.append(class_names[str(int(classes[k]))])
    print(cls_names)
    print(probs)
