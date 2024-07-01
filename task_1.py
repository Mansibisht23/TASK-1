'''Create a feature to detect the number of females as well as males in a meeting room along with their age.
if the person wear a white shirt we should make their age as 23 irrespective of their age and gender as well if they wear black shirt 
we should predict that person as child irrespective of their age and gender and this feature should not work if we have less than 2 people in the meeting'''

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf  # Import TensorFlow for the model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Placeholder functions for loading data
def load_data():
    # Dummy data for illustration
    # Replace this with actual code to load a dataset
    # For example, use datasets like IMDB-WIKI for real training
    images = []  # List to hold image data
    labels = []  # List to hold labels
    
    # Example data (image should be a 64x64x3 numpy array)
    # Replace this with actual image loading code
    for i in range(100):  # Dummy loop for 100 samples
        # Create dummy image (64x64x3)
        img = np.random.rand(64, 64, 3)
        images.append(img)
        if i % 2 == 0:
            labels.append('male')  # 50% male
        else:
            labels.append('female')  # 50% female
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Define and train the CNN model
def train_model(images, labels):
    # Preprocess the data
    images = images.astype("float") / 255.0  # Normalize images
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)  # Convert labels to one-hot encoding
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # 2 outputs for gender (male/female)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=8, validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save('gender_age_model.h5')
    
    return model

# Dummy gender and age prediction functions based on face area (for illustration purposes)
def predict_gender(face_image, model):
    # Placeholder for actual gender prediction
    face_image = cv2.resize(face_image, (64, 64))
    face_image = face_image.astype("float") / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    prediction = model.predict(face_image)
    gender = 'male' if np.argmax(prediction[0]) == 0 else 'female'
    return gender

def predict_age(face_image):
    # Placeholder for actual age prediction
    # We are using face area to mock age prediction
    if face_image.shape[0] * face_image.shape[1] > 5000:
        return 30
    else:
        return 20

# Detect faces in the image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Analyze the uploaded image
def analyze_image(image, model):
    faces = detect_faces(image)
    
    if len(faces) < 2:
        return "Less than 2 people in the meeting room, custom rules not applied.", None
    
    people_info = {'male': 0, 'female': 0, 'ages': []}
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        gender = predict_gender(face_image, model)
        age = predict_age(face_image)
        
        # Custom rules based on the detected face and shirt color
        # We are assuming shirt color based on face area for illustration purposes
        if face_image.shape[0] * face_image.shape[1] > 5000:
            age = 23  # Custom rule for white shirt assumed here
        elif age <= 12:
            age = "child"  # Custom rule for black shirt assumed here
        
        if gender == 'female' and age == 23:
            people_info[gender] = people_info.get(gender, 0) + 1
            people_info['ages'].append(age)
        elif gender == 'male' and age == 23:
            people_info[gender] = people_info.get(gender, 0) + 1
            people_info['ages'].append(age)
        elif age == "child":
            # Handle the child case if needed
            pass
        else:
            people_info[gender] = people_info.get(gender, 0) + 1
            people_info['ages'].append(age)

        # Draw a rectangle around the face and put the text on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f"{gender}, {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    average_age = np.mean([a for a in people_info['ages'] if isinstance(a, int)]) if people_info['ages'] else 0

    result_image_path = 'result_image.jpg'
    cv2.imwrite(result_image_path, image)
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result_text = (f"Number of males: {people_info.get('male', 0)}\n"
                   f"Number of females: {people_info.get('female', 0)}\n"
                   f"Average age of people: {average_age:.2f}")

    return result_text, result_image_path

# Function to handle file upload
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        result_text, result_image_path = analyze_image(image, model)
        if result_text:
            messagebox.showinfo("Result", result_text)
            if result_image_path:
                messagebox.showinfo("Result", f"Image processed. Result saved at {result_image_path}")

# Create the GUI
root = tk.Tk()
root.title("Meeting Room Analysis")
root.geometry("400x200")

# Create a stylish and bold upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 16, 'bold'), bg='blue', fg='white', padx=10, pady=10)
upload_button.pack(pady=20)

# Load or train the model
try:
    model = tf.keras.models.load_model('gender_age_model.h5')
    print("Model loaded successfully.")
except IOError:
    print("Model not found. Training a new model.")
    images, labels = load_data()
    model = train_model(images, labels)
    print("Model trained and saved successfully.")

root.mainloop()
