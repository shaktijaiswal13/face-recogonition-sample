import boto3
import configs

BUCKET = "amazon-rekognition123"
KEY_SOURCE = "test.jpeg"
KEY_TARGET = "target.jpeg"


def compare_faces(bucket, key, bucket_target, key_target, threshold=80):
    rekognition = boto3.client('rekognition',  aws_access_key_id=configs.AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY,
                               aws_session_token=configs.AWS_SESSION_TOKEN
                               region_name=configs.REGION_NAME)
    response = rekognition.compare_faces(
        SourceImage={
            "S3Object": {
                "Bucket": bucket,
                "Name": key,
            }
        },
        TargetImage={
            "S3Object": {
                "Bucket": bucket_target,
                "Name": key_target,
            }
        },
        SimilarityThreshold=threshold,
    )
    return response['SourceImageFace'], response['FaceMatches']


source_face, matches = compare_faces(BUCKET, KEY_SOURCE, BUCKET, KEY_TARGET)

# the main source face
print("Source Face {Confidence}%".format(**source_face))

# one match for each target face
for match in matches:
    print("Target Face {Confidence}%".format(**match['Face']))
    print("  Similarity : {}%".format(match['Similarity']))
