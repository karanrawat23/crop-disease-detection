import json
import tempfile
from pathlib import Path

import h5py
import keras
import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import model_from_json
from keras.preprocessing import image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model.hdf5"
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def load_legacy_model(model_path):
    with h5py.File(model_path, "r") as model_file:
        model_config = model_file.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        config = json.loads(model_config)

    first_layer_config = config["config"]["layers"][0]["config"]
    if "batch_input_shape" in first_layer_config:
        first_layer_config["input_shape"] = first_layer_config.pop(
            "batch_input_shape"
        )[1:]

    custom_objects = {
        "Sequential": keras.Sequential,
        "Conv2D": keras.layers.Conv2D,
        "MaxPooling2D": keras.layers.MaxPooling2D,
        "BatchNormalization": keras.layers.BatchNormalization,
        "Flatten": keras.layers.Flatten,
        "Dense": keras.layers.Dense,
        "Dropout": keras.layers.Dropout,
    }
    loaded_model = model_from_json(
        json.dumps(config),
        custom_objects=custom_objects,
    )
    loaded_model.load_weights(str(model_path))
    return loaded_model


print(" ** Model Loading **")
model = load_legacy_model(MODEL_PATH)
print(" ** Model Loaded **")


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    pixels = image.img_to_array(img)
    pixels = np.expand_dims(pixels, axis=0) / 255

    predictions = np.asarray(model.predict(pixels, verbose=0)).reshape(-1)
    if predictions.size != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {predictions.size} classes; "
            f"expected {len(CLASS_NAMES)}"
        )

    crop, disease = CLASS_NAMES[int(np.argmax(predictions))].split("___", 1)
    return crop, disease


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify(error="Image must be 10 MB or smaller."), 413


@app.route("/predict", methods=["POST"])
def upload():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Please choose an image file."), 400

    extension = Path(uploaded_file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Only JPG and PNG images are supported."), 400

    UPLOAD_DIR.mkdir(exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=extension, delete=False
    )
    file_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        uploaded_file.save(file_path)
        crop, disease = model_predict(file_path, model)
    except OSError as error:
        return jsonify(error=f"Unable to process this image: {error}"), 400
    finally:
        file_path.unlink(missing_ok=True)

    return (
        f"Predicted Crop:{crop}  "
        f"Predicted Disease:{disease.title().replace('_', ' ')}"
    )


if __name__ == "__main__":
    app.run()
