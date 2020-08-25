import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# for solving 'cudnn failed to initialize error'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

SAVED_MODEL_PATH = "../saved_model"
IMAGE_SIZE = [180, 180]

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	print("* Loading model...")
	model = keras.models.load_model(SAVED_MODEL_PATH)
	print("* Model loaded")


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)


def prep_image(image_bytes):
    img = np.array(image_bytes)
    img = tf.convert_to_tensor(img)
    img = decode_img(img)
    img = np.array(img)
    img = img.reshape(1, 180, 180, 3)
    return img


def get_pred_label(preds):
	labelled_preds = []
	for pred in preds:
		if pred[0] > 0.8:
			labelled_preds.append(('PNEUMONIA', pred[0]))
		else:
			labelled_preds.append(('NORMAL', pred[0]))
		print(labelled_preds)
	return labelled_preds


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()

			# preprocess the image and prepare it for classification
			image = prep_image(image)

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = get_pred_label(preds)
			data["predictions"] = []

			for result in results:
				r = {"label": result[0], "probability": float(result[1])}
				data["predictions"].append(r)
			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print("* Loading Keras model and Flask starting server...")
	load_model()
	print("* Model Loaded.")
	app.run(port=5001)