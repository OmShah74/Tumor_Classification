from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your best trained model
MODEL_PATH = '../models/cnn-parameters-improvement-04-0.82.keras'
model = load_model(MODEL_PATH)

IMG_WIDTH, IMG_HEIGHT = 240, 240

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    from data_utils import crop_brain_contour
    image_cropped = crop_brain_contour(image.copy())
    image_resized = cv2.resize(image_cropped, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded, image

def segment_and_draw_bbox(image, save_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)  # Tune this value as needed

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imwrite(save_path, image)
        return

    tumor_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(tumor_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(save_path, image)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img, original = preprocess_image(filepath)
            if img is None:
                return render_template('index.html', error="Invalid image format")

            prediction = model.predict(img)[0][0]
            result = "Tumor" if prediction >= 0.5 else "No Tumor"
            accuracy = round(prediction * 100 if result == "Tumor" else (100 - prediction * 100), 2)

            output_filename = 'bbox_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            if result == "Tumor":
                segment_and_draw_bbox(original, output_path)
            else:
                cv2.imwrite(output_path, original)

            return render_template('index.html', result=result, accuracy=accuracy, filename=output_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='upload/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
