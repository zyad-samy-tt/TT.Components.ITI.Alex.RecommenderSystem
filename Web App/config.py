from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Config:
    # Example configuration variables
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    REGION_NAME = os.getenv('REGION_NAME')
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    FOLDER_PATH = os.getenv('FOLDER_PATH')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_API_ANNOY = os.getenv('PINECONE_API_ANNOY')
    PINECONE_INDEX_NAME1 = os.getenv('PINECONE_INDEX_NAME1')
    PINECONE_INDEX_NAME2 = os.getenv('PINECONE_INDEX_NAME2')
    PINECONE_INDEX_ANNOY = os.getenv('PINECONE_INDEX_ANNOY')
