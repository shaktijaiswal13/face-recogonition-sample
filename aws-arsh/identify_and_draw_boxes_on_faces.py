import face_recognition
from PIL import Image, ImageDraw
import numpy as np

bill_image = face_recognition.load_image_file("test/image3.jpg")
bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

elon_image = face_recognition.load_image_file("test/image1.jpg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

known_face_encodings = [
    bill_face_encoding,
    elon_face_encoding
]
known_face_names = [
    "Bill Gates",
    "Elon Musk"
]

unknown_image = face_recognition.load_image_file("two_people.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding)

    name = "Unknown"
    face_distances = face_recognition.face_distance(
        known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))


del draw

pil_image.show()
