import face_recognition

# Load the jpg files into numpy arrays
kishan_image = face_recognition.load_image_file("kishan.jpg")
akshay_image = face_recognition.load_image_file("akshay.jpg")
stephen_image = face_recognition.load_image_file("stephen.jpg")
david_image = face_recognition.load_image_file("david.jpg")

unknown_image = face_recognition.load_image_file("???.jpg")
unknown_image2 = face_recognition.load_image_file("???-s.jpg")
unknown_image3 = face_recognition.load_image_file("???-d.jpg")
unknown_image4 = face_recognition.load_image_file("???-k.jpg")


# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    kishan_face_encoding = face_recognition.face_encodings(kishan_image)[0]
    akshay_face_encoding = face_recognition.face_encodings(akshay_image)[0]
    stephen_face_encoding = face_recognition.face_encodings(stephen_image)[0]
    david_face_encoding = face_recognition.face_encodings(david_image)[0]

    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    unknown_face_encoding2 = face_recognition.face_encodings(unknown_image2)[0]
    unknown_face_encoding3 = face_recognition.face_encodings(unknown_image3)[0]
    unknown_face_encoding4 = face_recognition.face_encodings(unknown_image4)[0]

except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    kishan_face_encoding,
    akshay_face_encoding,
    stephen_face_encoding,
    david_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
print(results)
print("Is the unknown face a picture of Kishan? {}".format(results[0]))
print("Is the unknown face a picture of Akshay? {}".format(results[1]))
print("Is the unknown face a picture of Stephen? {}".format(results[2]))
print("Is the unknown face a picture of David? {}".format(results[3]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

print("---unk 2--- should be Stephen")

results = face_recognition.compare_faces(known_faces, unknown_face_encoding2)
print(results)
print("Is the unknown2 face a picture of Kishan? {}".format(results[0]))
print("Is the unknown2 face a picture of Akshay? {}".format(results[1]))
print("Is the unknown2 face a picture of Stephen? {}".format(results[2]))
print("Is the unknown2 face a picture of David? {}".format(results[3]))
print("Is the unknown2 face a new person that we've never seen before? {}".format(not True in results))

print("---unk 3--- should be David")

results = face_recognition.compare_faces(known_faces, unknown_face_encoding3)
print(results)
print("Is the unknown3 face a picture of Kishan? {}".format(results[0]))
print("Is the unknown3 face a picture of Akshay? {}".format(results[1]))
print("Is the unknown3 face a picture of Stephen? {}".format(results[2]))
print("Is the unknown3 face a picture of David? {}".format(results[3]))
print("Is the unknown3 face a new person that we've never seen before? {}".format(not True in results))

print("---unk 4--- should be Kishan") 

results = face_recognition.compare_faces(known_faces, unknown_face_encoding4)
print(results)
print("Is the unknown4 face a picture of Kishan? {}".format(results[0]))
print("Is the unknown4 face a picture of Akshay? {}".format(results[1]))
print("Is the unknown4 face a picture of Stephen? {}".format(results[2]))
print("Is the unknown4 face a picture of David? {}".format(results[3]))
print("Is the unknown4 face a new person that we've never seen before? {}".format(not True in results))



