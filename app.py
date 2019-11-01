import pickle
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras import backend as K
from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/analyse', methods=["GET","POST"])
def predict_plant_disease():
    try:
        if request.method == "GET" :
            return_data = {
                "error" : "0",
                "message" : "Plant Disease Recognition Api."
            }
        else:
            if request.files:
                image_file = request.files["plant_image"]
                image_data = base64.b64encode(image_file.read())
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    if image is not None :
                        image = image.resize(tuple((256, 256)), Image.ANTIALIAS)
                        image_array = np.expand_dims(img_to_array(image), axis=0)
                    else :
                        raise Exception('Error loading image file')
                except Exception as e:
                    return None, str(e)
                K.clear_session()
                model_file = f"./cnn_model.pkl"
                saved_classifier_model = pickle.load(open(model_file,'rb'))
                prediction = saved_classifier_model.predict(image_array) 
                label_binarizer = pickle.load(open(f"./label_transform.pkl",'rb'))
                return_data = {
                    "error" : "0",
                    "data" : f"{label_binarizer.inverse_transform(prediction)[0]}"
                }
            else :
                return_data = {
                    "error" : "1",
                    "message" : "Request Body is empty",
                }
    except Exception as e:
        return_data = {
            "error" : "3",
            "message" : f"Error : {str(e)}",
        }
    return return_data

if __name__ == '__main__':
    app.run()