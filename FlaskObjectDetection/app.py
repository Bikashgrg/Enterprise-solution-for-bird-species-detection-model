# importing required libraries 
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import time

# importing OpenCV for image processing
import cv2

# importing gRPC to build scalable and fast APIs
import grpc
from grpc.beta import implementations

# import prediction service functions from TF-Serving API
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields

# Parent of the current working directory
sys.path.append("..")
tf.get_logger().setLevel('ERROR')

# Path to label map(.pbtxt) file and number of classes the model will classify
PATH_TO_LABELS = "./data/label_map.pbtxt"
NUM_CLASSES = 3

# To load in our label map file
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# To specify flask instance
app = Flask(__name__)

# For storing our output image
app.config['UPLOAD_FOLDER'] = 'uploads/'

# To allow only three image extensions
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# Function for allowed extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# Function to setup gRPC channel to communicate with tensorflow serving
def get_stub(host='127.0.0.1', port='8500'):
    channel = grpc.insecure_channel('127.0.0.1:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub

# Function for loading an image into numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Function for converting numpy array to tensor proto
def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis = 0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor

# Function for doing the inference of the model
def inference(frame, stub, model_name='od'):
    # To call tensorflow server with the help of gRPC
    channel = grpc.insecure_channel('localhost:8500', options=(('grpc.enable_http_proxy',0),))
    print("Channel: ", channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print("Stub: ", stub)
    request = predict_pb2.PredictRequest()
    print("Request: ", request)
    request.model_spec.name = 'od'

    # To convert color using openCV
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    input_tensor = load_input_tensor(image)
    request.inputs['input_tensor'].CopyFrom(input_tensor)

    result = stub.Predict(request, 60.0)
    image_np = frame.copy()

    # To create detection boxes
    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8) # detection classes should be integer
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)

    # To put the  detection boxes. class name  and  scores  on  the image
    frame = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np, output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh = .50,
        agnostic_mode=False
    )
    return (frame)

# To display the index page
@app.route('/')
def index():
    return render_template('index.html')

# Function to upload the valid image
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))
    if file.filename == '':
        print('No image is selected for uploading.')
        return redirect(request.url)
    else:
        print("This image extension is not allowed. Only .jpg, .png, and .jpeg extensions are allowed.")
        return redirect(request.url)

# Function to do inferencing and then save the uploaded image (with detection boxes and accuracy)
@app.route('/results/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)
    stub = get_stub()
    
    # To save the detected image or output image to the 'uploads' folder
    for image_path in TEST_IMAGE_PATHS:
        image_np = np.array(Image.open(image_path))
        image_np_inferenced = inference(image_np, stub)
        im = Image.fromarray(image_np_inferenced)
        im.save('uploads/' + filename)
    
    # To show the output image in the results.html page
    image_url = url_for('extract_file', filename=filename)
    
    return render_template('results.html',image_url=image_url)

# Function to show the detected or output image after inferencing
@app.route('/extract/<filename>')
def extract_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# To start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
