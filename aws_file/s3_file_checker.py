import os
import boto3
from botocore.exceptions import ClientError

def check_file_in_s3(subject_id: str, file_name: str) -> tuple[bool, Exception | None]:
    """
    Kiểm tra sự tồn tại của file trong S3 bucket.
    
    Args:
        subject_id: ID của subject
        file_name: Tên file cần kiểm tra
        
    Returns:
        Tuple chứa (kết quả kiểm tra, lỗi nếu có)
    """
    # Lấy cấu hình từ biến môi trường
    bucket = os.getenv("AWS_BUCKET")
    if not bucket:
        return False, ValueError("Biến môi trường AWS_BUCKET không được thiết lập")
    
    region = os.getenv("AWS_REGION")
    if not region:
        return False, ValueError("Biến môi trường AWS_REGION không được thiết lập")
    
    try:
        # Khởi tạo client S3
        s3_client = boto3.client('s3', region_name=region)
        
        # Kiểm tra sự tồn tại của file
        s3_client.head_object(
            Bucket=bucket,
            Key=f"checker/{subject_id}/{file_name}"
        )
        return True, None
        
    except ClientError as e:
        # Nếu file không tồn tại, trả về False với error là None
        if e.response['Error']['Code'] == "404":
            return False, None
        # Các lỗi khác thì trả về False với error
        return False, e
    except Exception as e:
        # Xử lý các lỗi khác
        return False, e