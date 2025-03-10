import requests
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'ap-southeast-2')
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'similar-dainam')

# Verify that required environment variables are set
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY]):
    raise ValueError("AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY and AWS_SECRET_KEY.")

def create_presigned_url(bucket_name, object_key, expiration=900):
    """Create pre-signed URL for S3 object"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        return None

# @app.route('/presigned-url/<subject_id>/<file_name>', methods=['GET'])
def read_presigned_url(subject_id, file_name):
    object_key = f"{subject_id}/{file_name}"
    url = create_presigned_url(BUCKET_NAME, object_key)
    # File name to save as
    output_file = f"./temp/{file_name}.pdf"
    print(url)
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully as {output_file}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")
    if not url:
        print("Failed to create presigned URL")
        return None
    
    return output_file

# if __name__ == '__main__':
#     app.run(debug=True, port=5555)