import numpy as np
import cv2
from flask import Flask, render_template, request, redirect
from scipy.fft import fft2, ifft2, fftshift
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def high_pass_filter(img, cutoff=30):
    # Transformasi Fourier
    f_transform = fft2(img)
    f_transform_shift = fftshift(f_transform)

    # Membuat mask untuk filter frekuensi tinggi
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # Aplikasikan mask
    f_transform_shift = f_transform_shift * mask

    # Transformasi balik ke domain spasial
    f_ishift = np.fft.ifftshift(f_transform_shift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Membaca gambar grayscale
        img = cv2.imread(filepath, 0)

        # Proses perbaikan citra
        processed_img = high_pass_filter(img)

        # Simpan hasil gambar
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
        cv2.imwrite(processed_filepath, processed_img)

        return render_template('index.html', original=filepath, processed=processed_filepath)

if __name__ == '__main__':
    app.run(debug=True)
