import boto3
import configs

BUCKET = "amazon-rekognition123"
KEY = "people.jpeg"
FEATURES_BLACKLIST = ("Landmarks", "Emotions", "Pose",
                      "Quality", "BoundingBox", "Confidence")


def detect_faces(bucket, key):
    rekognition = boto3.client('rekognition',  aws_access_key_id=configs.AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY,
                               aws_session_token=configs.AWS_SESSION_TOKEN
                               region_name=configs.REGION_NAME)
    response = rekognition.detect_faces(
        Image={
            "S3Object": {
                "Bucket": bucket,
                "Name": key,
            }
        },
        Attributes=['ALL']
    )
    return response.get('FaceDetails', [])


for face in detect_faces(BUCKET, KEY):
    print("Face ({Confidence}%)".format(**face))

    # Emotions
    for emotion in face.get('Emotions', []):
        print("  {Type} : {Confidence}%".format(**emotion))

    # Quality
    for quality, value in face['Quality'].items():
        print("  {quality} : {value}".format(quality=quality, value=value))

    # Facial features
    for feature, data in face.items():
        if feature not in FEATURES_BLACKLIST:
            value = data.get('Value', 'N/A')
        confidence = data.get('Confidence', 'N/A')
        print("  {feature}({value}) : {confidence}%".format(
            feature=feature, value=value, confidence=confidence))
