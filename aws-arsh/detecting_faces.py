import boto3
import json
import configs


def detect_faces(photo, bucket):

    rekognition = boto3.client('rekognition',  aws_access_key_id=configs.AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY,
                               aws_session_token=configs.AWS_SESSION_TOKEN
                               region_name=configs.REGION_NAME)

    response = rekognition.detect_faces(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
                                        Attributes=['ALL'])

    print('Detected faces for ' + photo)
    for faceDetail in response['FaceDetails']:
        print('The detected face is between ' + str(faceDetail['AgeRange']['Low'])
              + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')

        print('Here are the other attributes:')
        print(json.dumps(faceDetail, indent=4, sort_keys=True))

        print("Gender: " + str(faceDetail['Gender']))
        print("Smile: " + str(faceDetail['Smile']))

    return len(response['FaceDetails'])


def main():
    photo = 'test.jpeg'
    bucket = 'amazon-rekognition123'
    face_count = detect_faces(photo, bucket)
    print("Faces detected: " + str(face_count))


if __name__ == "__main__":
    main()
