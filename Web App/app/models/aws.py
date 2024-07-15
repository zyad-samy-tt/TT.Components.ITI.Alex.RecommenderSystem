import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from config import Config

# accessKeys = pd.read_csv(r'E:\ITI\TweentyToo\web app_v2\iti-user_accessKeys.csv')
aws_access_key_id = Config.AWS_ACCESS_KEY_ID
aws_secret_access_key = Config.AWS_SECRET_ACCESS_KEY
region_name = Config.REGION_NAME
bucket_name = Config.BUCKET_NAME
folder_path = Config.FOLDER_PATH

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

s3 = session.client('s3')

def get_images(product_ids):
    image_urls = {}
    for product_id in product_ids:
        try:
            image_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': f"{folder_path}{product_id}_0.jpg"},
                ExpiresIn=3600
            )
            # Check if the image actually exists
            try:
                s3.head_object(Bucket=bucket_name, Key=f"{folder_path}{product_id}_0.jpg")
                image_urls[product_id] = image_url
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    image_urls[product_id] = None  # Image does not exist
                else:
                    raise  # Handle other errors if needed
        except NoCredentialsError:
            image_urls[product_id] = None  # No credentials to access S3
    return image_urls