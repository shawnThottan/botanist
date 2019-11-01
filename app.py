import pickle
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from flask import Flask
from flask import request
app = Flask(__name__)

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
                        image = image.resize(tuple((224, 224)), Image.ANTIALIAS)
                        image_array = np.expand_dims(img_to_array(image), axis=0)
                    else :
                        raise Exception('Error loading image file')
                except Exception as e:
                    return None, str(e)
                model_file = f"cnn_model.pkl"
                
                interpreter = tf.lite.Interpreter(model_path="optimized_graph.tflite")
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], image_array)

                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                best_match_index = np.argmax(output_data)

                with open('retrained_labels.txt', 'r') as f:
                    matrix = [line.strip() for line in f]
                return_data = {
                    "error" : "0",
                    "data" : matrix[best_match_index]
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