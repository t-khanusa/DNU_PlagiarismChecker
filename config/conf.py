from dotenv import load_dotenv
import os
from os.path import join, dirname

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'ap-southeast-2')
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'similar-dainam')

DATABASE_USERNAME = os.getenv('DATABASE_USERNAME')
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD')
DATABASE_HOSTNAME = os.getenv('DATABASE_HOSTNAME')
DATABASE_PORT = os.getenv('DATABASE_PORT')
DATABASE_NAME = os.getenv('DATABASE_NAME')

MILVUS_HOST=os.getenv('MILVUS_HOST')
MILVUS_PORT=os.getenv('MILVUS_PORT')
MILVUS_DB_NAME=os.getenv('MILVUS_DB_NAME')
BASEURL=os.getenv('baseURL')
COOKIE=os.getenv('COOKIE')