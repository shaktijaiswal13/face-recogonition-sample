import face_recognition
elon_image = face_recognition.load_image_file("test/image1.jpg")
bill_image = face_recognition.load_image_file("test/image3.jpg")
unknown_image = face_recognition.load_image_file("test/image4.jpg")
try:
    elon_face_encoding = face_recognition.face_encodings(elon_image)[0]
    bill_face_encoding = face_recognition.face_encodings(bill_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    elon_face_encoding,
    bill_face_encoding
]

results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of elon? {}".format(results[0]))
print("Is the unknown face a picture of bill? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(
    not True in results))
