import cv2
import face_recognition
import os
import numpy as np
import configs

# Step 1: Load images and detect faces
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Step 2: Extract embeddings from each image
def extract_embeddings(images):
    embeddings = []
    face_locations = []
    for image in images:
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        face_locations_in_image = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations_in_image)
        
        # Store the embeddings and corresponding face locations
        for encoding, location in zip(face_encodings, face_locations_in_image):
            embeddings.append(encoding)
            face_locations.append(location)

    return embeddings, face_locations

# Step 3: Compare the embeddings to identify unique faces
def identify_unique_faces(embeddings, threshold=0.6):
    unique_faces = []
    unique_embeddings = []

    # Compare each embedding with the already stored unique embeddings
    for encoding in embeddings:
        matches = face_recognition.compare_faces(unique_embeddings, encoding, tolerance=threshold)
        
        if True not in matches:
            unique_faces.append(encoding)
            unique_embeddings.append(encoding)

    return unique_faces

# Step 4: Save unique faces to the specified folder
def save_unique_faces(images, face_locations, unique_faces, unique_embeddings, filenames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, encoding in enumerate(unique_faces):
        # Find the face location for the unique face
        location = face_locations[i]
        # Extract the face image based on the location
        top, right, bottom, left = location
        face_image = images[i][top:bottom, left:right]
        
        # Save the unique face to the output folder
        unique_face_filename = f"unique_face_{i + 1}.jpg"
        unique_face_path = os.path.join(output_folder, unique_face_filename)
        
        cv2.imwrite(unique_face_path, face_image)
        print(f"Saved unique face {i + 1} to {unique_face_path}")
        

# Step 5: Main function
def main():
    folder_path = configs.DETECTED_FACE_FOLDER
    output_folder = configs.UNIQUE_FACE_FOLDER  # Folder to save unique faces

    # Load all images from the folder
    images, filenames = load_images_from_folder(folder_path)
    
    # Extract embeddings and face locations
    embeddings, face_locations = extract_embeddings(images)
    
    # Identify unique faces based on embeddings
    unique_faces = identify_unique_faces(embeddings, 0.3)

    print(f"Number of unique faces identified: {len(unique_faces)}")
    # Save the unique faces to the specified output folder
    save_unique_faces(images, face_locations, unique_faces, embeddings, filenames, output_folder)

    print(f"Total unique faces saved: {len(unique_faces)}")




if __name__ == "__main__":
    main()