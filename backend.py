from flask import Flask, request, render_template, redirect, url_for, jsonify
# import tensorflow as tf
# import onnxruntime as ort
from PIL import Image
# import numpy as np
# import io
import matlab.engine
import os
from meta_ai_api import MetaAI



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory to save uploaded files

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

'''
# Load TensorFlow model
# model = tf.keras.models.load_model('Rice_model.h5')  # Update with your model path
session = ort.InferenceSession("Rice_model.onnx")
input_name = session.get_inputs()[0].name



# Define the target image size (matching model input size)
IMAGE_SIZE = (224, 224)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(IMAGE_SIZE)
    image = np.array(image).astype(np.float32)  # Convert image data to float32
    image = image / 255.0  # Normalize the image
    image = np.transpose(image, (2, 0, 1))  # Reorder dimensions to (channels, height, width)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# def predict_image(image):
#     prediction = input_name.predict(image)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     return f"Predicted Class: {predicted_class}"

def predict_image(image):
    # Run inference using the ONNX session
    prediction = session.run(None, {input_name: image})
    predicted_class = np.argmax(prediction[0], axis=1)[0]
    return f"Predicted Class: {predicted_class}"
'''

@app.route('/')
def home():
    return render_template('temp.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return redirect(url_for('home'))

#     file = request.files['file']
#     if file.filename == '':
#         return redirect(url_for('home'))

#     if file:
#         image_bytes = file.read()
#         image = preprocess_image(image_bytes)
#         result = predict_image(image)
#         return render_template('web.html', result=result)

def getResult():
    eng = matlab.engine.start_matlab()
    predicted_label_str = eng.predict_rice(nargout=1)

    # Close the MATLAB engine
    eng.quit()

    return str(predicted_label_str)



@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpeg")
        file.save(file_path)
        result=getResult()
        return jsonify({'message': result})

    return jsonify({'error': 'File upload failed'}), 500

@app.route('/result')
def result():
    return render_template('result.html')  # Render the result.html page



@app.route('/chatapi', methods=['POST'])
def submit_data():
    plant_health = request.form.get('plantHealth')
    weather_data = request.form.get('weatherData')

    response = {
        'receivedPlantHealth': plant_health,
        'receivedWeatherData': weather_data
    }
    print(plant_health+"\n")

    if plant_health.strip()=='healthy_rice_plant':
        query="Analyze this Environmental condition, "+weather_data+" and according to this tell what are the possible diseases can occur in the Rice/paddy crop and suggest preventive measures and treatments based on real-time data.Provide the data in 5 bullet points"
    else:
        query="For this, "+plant_health+" disease and Environmental condition, "+weather_data+" suggest preventive measures and treatments based on real-time data.Provide the data in 5 bullet points"


    print(query)

    ai = MetaAI()
    response = ai.prompt(message=query)["message"].replace('\n', ' ')
    # response=ai(query)
    response2=ai.prompt(message="what is "+plant_health)["message"].replace('\n', ' ')

    return jsonify({"suggestion":response,"disease":response2})

if __name__ == '__main__':
    app.run(debug=True)
