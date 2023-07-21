
import io
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify , abort
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import skimage.io

import time
import os
import sys

ROOT_DIR = os.path.abspath("../")
import warnings
warnings.filterwarnings("ignore")

# Importing Mask RCNN 
sys.path.append(ROOT_DIR) 
sys.path.append(os.path.join(ROOT_DIR, "ImageSegmentation/samples/coco/")) 
import coco

# Define the Flask app
app = Flask(__name__)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "ImageSegmentation/logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

class InferenceConfig(coco.CocoConfig):
    # Setting batch size equal to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()


# Create a Mask R-CNN model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir="mask_rcnn_coco.hy", config=config)

# Load pre-trained weights
model.load_weights("mask_rcnn_coco.h5", by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

@app.route("/image", methods=["POST"])
def detect():
    # Get the image file from the request
    image = request.files["images"]
    # image_name = image.filename
    # image.save(os.path.join(os.getcwd(), image_name))

    # Load the image using Pillow
    image = Image.open(io.BytesIO(image.read()))
    image = np.array(image)

    # Preprocess the image using Mask R-CNN's preprocess_input function
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)
    
    t1 = time.time()
    # Perform object detection on the input image
    results = model.detect([image], verbose=0)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    r = results[0]
    # output = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], show_bbox=True) #['BG'] + class_names , 
    
    # np.squeeze(output)

    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']

    objects = []
    for i in range(len(class_ids)):
        class_id = class_ids[i]
        class_name = class_names[class_id]
        score = scores[i]
        object = {
            'class_name': class_name,
            'score': float(score*100)
        }
        objects.append(object)
    response = {
        'objects': objects
    }

    # Convert the dictionary to JSON and return it
    return jsonify(response, ('time: taken ', t2 - t1))

if __name__ == '__main__':
    app.run(debug=True)