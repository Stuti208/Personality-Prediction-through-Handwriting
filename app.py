from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model("handwriting_traits_model.h5")

# Define the classes (Make sure these match your model's output classes)
class_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

trait_descriptions = {
    "Openness": (
        "You have a vivid imagination and an active curiosity for new experiences. "
        "You are highly creative, enjoy exploring new ideas, and appreciate art, beauty, and innovation. "
        "Your open-minded nature allows you to embrace different perspectives and think outside the box. "
        "You are often adventurous, intellectually curious, and eager to engage in activities that challenge your thinking. "
        "You thrive in environments that encourage originality and self-expression."
    ),
    "Conscientiousness": (
        "You are organized, dependable, and show a strong sense of duty and responsibility. "
        "You approach tasks with precision, diligence, and a clear plan to achieve your goals. "
        "You have excellent self-control and are often driven by a desire to excel in your commitments. "
        "Your structured and disciplined approach helps you manage time effectively and meet deadlines consistently. "
        "You value hard work, reliability, and integrity, and often act as a role model for others."
    ),
    "Extraversion": (
        "You are outgoing, energetic, and thrive in social situations. "
        "You draw energy from interacting with people and often bring enthusiasm to group activities. "
        "You are confident, assertive, and enjoy leading conversations or taking charge in social settings. "
        "Your optimistic and upbeat nature makes you approachable, and you’re often the life of the party. "
        "You seek opportunities to connect with others and gain fulfillment from collaborative experiences."
    ),
    "Agreeableness": (
        "You are compassionate, cooperative, and deeply value positive relationships with others. "
        "Your empathetic and caring nature makes you a trusted confidant and a supportive friend. "
        "You are driven by a desire to help others, often prioritizing harmony and mutual understanding. "
        "Your ability to forgive and willingness to compromise help you maintain strong, lasting bonds. "
        "You are a peace-seeker who values fairness and enjoys fostering goodwill in your community."
    ),
    "Neuroticism": (
        "You are emotionally sensitive and deeply in tune with your feelings. "
        "While you may experience stress or worry more acutely than others, this makes you self-aware and thoughtful. "
        "You are often introspective, which can lead to personal growth and a deeper understanding of yourself. "
        "Your heightened emotional awareness enables you to empathize with others and connect on a meaningful level. "
        "While challenges may feel overwhelming at times, they often serve as catalysts for your resilience and creativity."
    ),
}

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((128, 128))  # Resize image to match model input size
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route to home page
@app.route('/')
def index():
    return render_template('index1.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400 
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Attempting to save file at: {file_path}")

    try:
        file.save(file_path)
        print(f"File successfully saved at: {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return 'Error saving file', 500
   
    try:
        # Open the uploaded image
        img = Image.open(file).convert('RGB')  # Ensure RGB format
        img = img.resize((128, 128))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        class_names = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
        trait = class_names[predicted_class[0]]
        description = trait_descriptions.get(trait, "No description available.")
        return render_template('index1.html', prediction=trait,description=description, image_file=f"images/{filename}")
    
    except Exception as e:
        print("Error during prediction:", e)
        return 'Error processing file', 500

if __name__ == '__main__':
    app.run(debug=True)
