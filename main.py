import tensorflow as tf
from keras.models import load_model;
from PIL import Image;
from flask import Flask, render_template, request;
import numpy as np;

# Inisialisasi Flask app
app = Flask(__name__)

# Memuat model dari file .h5
model = load_model('modelflower85nice.h5')
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']


# Fungsi untuk memproses gambar
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Mengubah ukuran gambar menjadi 150 x 150 piksel
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Route utama untuk upload gambar
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Mendapatkan file gambar dari form
        file = request.files['file']

        # Menyimpan file sementara
        file_path = 'static/temp.jpg'
        file.save(file_path)

        # Memproses gambar
        img = process_image(file_path)

        # Melakukan klasifikasi gambar
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]
        accuracy = predictions[0][predicted_class] * 100
        accuracy_formatted = "{:.2f}".format(accuracy)

        # Mengirimkan hasil klasifikasi ke halaman web
        return render_template('results.html', label=predicted_label, accuracy=accuracy_formatted, image_file=file_path)
        
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
