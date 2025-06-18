import requests
import boto3
from config.conf import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, BUCKET_NAME
import os
from botocore.exceptions import ClientError

# Verify that required environment variables are set
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY]):
    raise ValueError("AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY and AWS_SECRET_KEY.")

def create_presigned_url(bucket_name, object_key, expiration=900):
    """Create pre-signed URL for S3 object"""
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


def read_presigned_url(subject_id, file_name):

    object_key = f"checker/{subject_id}/{file_name}"
    url = create_presigned_url(BUCKET_NAME, object_key)

    # File name to save as
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    output_file = f"{parent_dir}/file_db/{file_name}"
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


def check_file_in_s3(subject_id: str, file_name: str) -> tuple[bool, Exception | None]:
   
    if not BUCKET_NAME:
        return False, ValueError("Biến môi trường AWS_BUCKET không được thiết lập")
    
   
    if not AWS_REGION:
        return False, ValueError("Biến môi trường AWS_REGION không được thiết lập")
    
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    print(f"Kiểm tra sự tồn tại của file: {file_name}")
    s3_client.head_object(
        Bucket=BUCKET_NAME,
        Key=f"checker/{subject_id}/{file_name}"
    )
    return True, None
        
    # except ClientError as e:
    #     # Nếu file không tồn tại, trả về False với error là None
    #     if e.response['Error']['Code'] == "404":
    #         return False, None
    #     # Các lỗi khác thì trả về False với error
    #     return False, e
    # except Exception as e:
    #     # Xử lý các lỗi khác
    #     return False, e