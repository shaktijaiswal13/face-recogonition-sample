import boto3
import configs

BUCKET = "amazon-rekognition123"
KEY = "people.jpeg"


def detect_labels(bucket, key, max_labels=10, min_confidence=90):
    rekognition = boto3.client('rekognition',  aws_access_key_id=configs.AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY,
                               aws_session_token=configs.AWS_SESSION_TOKEN
                               region_name=configs.REGION_NAME)
    response = rekognition.detect_labels(
        Image={
            "S3Object": {
                "Bucket": bucket,
                "Name": key,
            }
        },
        MaxLabels=max_labels,
        MinConfidence=min_confidence,
    )
    return response['Labels']


for label in detect_labels(BUCKET, KEY):
    print("{Name} - {Confidence}%".format(**label))
