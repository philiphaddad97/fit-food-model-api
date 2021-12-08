import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

model = keras.models.load_model("model.h5")


def calories_estimation(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_arr = 1. / 255 * image.img_to_array(img)
    feature_vec = np.expand_dims(img_arr, axis=0)
    return model.predict(feature_vec)[0][0]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            f = request.files['file']
            image_path = 'images/' + f.filename
            f.save(image_path)
            prediction = calories_estimation(image_path)
            data = {"prediction": float(prediction)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)