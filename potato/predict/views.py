import os
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from .forms import ImageUploadForm

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")
model = load_model(MODEL_PATH)

# Class Labels
CLASS_LABELS = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight"
]

def predict_disease(request):
    prediction = None
    confidence = None
    img_url = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = request.FILES["image"]

            # Ensure media directory exists
            MEDIA_DIR = "media"
            os.makedirs(MEDIA_DIR, exist_ok=True)

            # Save image
            img_path = os.path.join(MEDIA_DIR, img_file.name)
            with open(img_path, "wb+") as destination:
                for chunk in img_file.chunks():
                    destination.write(chunk)

            # Load image for prediction
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Predict
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            prediction = CLASS_LABELS[predicted_class_index]
            confidence = float(np.max(predictions)) * 100  # Convert to percentage

            # Store image URL for display
            img_url = f"/media/{img_file.name}"

    else:
        form = ImageUploadForm()

    return render(request, "index.html", {"form": form, "prediction": prediction, "confidence": confidence, "img_url": img_url})
