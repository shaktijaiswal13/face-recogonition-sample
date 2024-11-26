import os
import boto3
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# AWS credentials and region
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.getenv('AWS_SESSION_TOKEN')
aws_region = os.getenv('AWS_DEFAULT_REGION')

# Initialize AWS DynamoDB client
dynamodb = boto3.resource('dynamodb',
                          region_name=aws_region,
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          aws_session_token=aws_session_token)
table_name = 'facerecognition'
table = dynamodb.Table(table_name)

# Scan the table to find items where fullname is missing or null
response = table.scan(
    FilterExpression='FullName = :full_name',
    ExpressionAttributeValues={':full_name': 'Unknown'}
)

# Check if there are items to delete
if 'Items' in response:
    for item in response['Items']:
        rekognition_id = item['RekognitionId']  # Assuming RekognitionId is the primary key
        
        # Delete each item by RekognitionId
        delete_response = table.delete_item(
            Key={
                'RekognitionId': rekognition_id
            }
        )
        print(f"Deleted item with RekognitionId: {rekognition_id}")
else:
    print("No items with FullName 'Unknown' found.")
