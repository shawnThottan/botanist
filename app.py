import pickle
import io
import base64
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from flask import Flask
from flask import request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/analyse', methods=["GET","POST"])
def predict_plant_disease():
    try:
        if request.method == "GET" :
            return_data = {
                "error" : "0",
                "message" : "Plant Disease Recognition Api."
            }
        else:
            if request.form:
                print(0)
                request_data = request.form["plant_image"]
                header, image_data = request_data.split(';base64,')
                image_array = 0
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    if image is not None :
                        image = image.resize(tuple((256, 256)), Image.ANTIALIAS)
                        image_array = np.expand_dims(img_to_array(image), axis=0)
                    else :
                        raise Exception('Error loading image file')
                except Exception as e:
                    return None, str(e)
                model_file = f"cnn_model.pkl"
                saved_classifier_model = pickle.load(open(model_file,'rb'))
                prediction = saved_classifier_model.predict(image_array) 
                label_binarizer = pickle.load(open(f"label_transform.pkl",'rb'))
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