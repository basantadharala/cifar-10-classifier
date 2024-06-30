from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)
model = load_model('cifar-10.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file.save(file_path)
            predicted_class, confidence = predict_image(file_path)
            return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)