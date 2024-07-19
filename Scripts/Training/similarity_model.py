import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm  # PyTorch Image Models
from PIL import Image
from transformers import RobertaTokenizer, RobertaModel
from io import BytesIO
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('REGION_NAME')
BUCKET_NAME = os.getenv('BUCKET_NAME')
FOLDER_PATH = os.getenv('FOLDER_PATH')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

# Set up argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path', type=str, required=True, help='Path to the input data CSV file.')

# Load pre-trained Roberta for text
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
text_model = RobertaModel.from_pretrained('roberta-large', add_pooling_layer=False)

# Load pre-trained EfficientNet-B3 for images
image_model = timm.create_model('efficientnet_b5', pretrained=True,
                                num_classes=0)  # num_classes=0 removes the classification layer

# Set models to evaluation mode
text_model.eval()
image_model.eval()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap models with DataParallel
text_model = nn.DataParallel(text_model).to(device)
image_model = nn.DataParallel(image_model).to(device)

# Define image preprocessing for EfficientNet
preprocess = transforms.Compose([
    transforms.Resize(224),  # Resize the shortest side to 224 pixels
    transforms.CenterCrop(224),  # Crop the center to get 224x224 image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
])


# Concatenation Fusion Model with Dimensionality Reduction
class ConcatFusion(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim):
        super(ConcatFusion, self).__init__()
        self.fc = nn.Linear(text_dim + image_dim, output_dim)

    def forward(self, text_features, image_features):
        concatenated_features = torch.cat((text_features, image_features), dim=1)
        combined_features = self.fc(concatenated_features)
        normalized_features = F.normalize(combined_features, p=2, dim=1)  # L2 normalization
        return normalized_features


# Define feature dimensions
text_dim = 1024  # Dimension of the DistilBert embeddings
image_dim = image_model.module.num_features  # Dimension of the EfficientNet-B0 embeddings
hidden_dim = 1024  # Hidden dimension for attention

# Instantiate Concat Fusion Model
concat_fusion_model = ConcatFusion(text_dim, image_dim, hidden_dim)
concat_fusion_model = nn.DataParallel(concat_fusion_model).to(device)

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME
)

s3 = session.client('s3')

pc = Pinecone(api_key=PINECONE_API_KEY)  # Replace with your api key

# Create a Pinecone index
try:
    #     print("hello")
    pc.create_index(PINECONE_INDEX_NAME, dimension=hidden_dim, metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    ))
except:
    pass

index = pc.Index(PINECONE_INDEX_NAME)


# Function to extract text features
def get_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    text_features = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return text_features


# Function to extract image features from S3
def get_image_features(product_id):
    try:
        object_key = f'{FOLDER_PATH}{product_id}_0.jpg'
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = image_model(image_tensor).squeeze().cpu()
        return image_features
    except Exception as e:
        print(f"Error loading image {product_id}: {e}")
        return None


# Function to process a single row
def process_row(row):
    # description = row['Original_Description']
    product_id = row['product_id']

    description = row.get('Original_Description', "")
    title = row.get('title', "")
    department = row.get('department', "")
    target_audience = row.get('target_audience', "")
    # print(type(description),)

    # Ensure all text fields are not None
    description_edited = description if description != "" and isinstance(description, str) else ""
    title_edited = f'Title: {title};' if title != "" and isinstance(title, str) else ""
    department_edited = f'Department: {department};' if department != "" and isinstance(department, str) else ""
    target_audience_edited = f'Target Audience: {target_audience};' if target_audience != "" and isinstance(
        target_audience, str) else ""

    combined_description = f"{title_edited} {department_edited} {target_audience_edited} {description_edited}"

    # print(combined_description)

    text_features = get_text_features(combined_description).to(device).view(1, -1)  # Ensure correct shape
    image_features = get_image_features(product_id)

    if image_features is not None:
        image_features = image_features.to(device).view(1, -1)  # Ensure correct shape
        fused_features = concat_fusion_model(text_features, image_features)
        fused_features = fused_features.cpu().detach().numpy().flatten()

        # Prepare vector in Pinecone format
        vector = {
            "id": str(product_id),
            "values": fused_features.tolist(),
            "metadata": {
                "title": title,
                "department": department,
                "target_audience": target_audience
            }
        }

        return vector
    else:
        return None


df = pd.read_csv("/kaggle/working/Cleaning_Product_FullData.csv")
rows = df.to_dict(orient='records')

import os

processed_ids_file = './processed_ids.txt'
if os.path.exists(processed_ids_file):
    with open(processed_ids_file, 'r') as f:
        processed_ids = set(line.strip() for line in f)
else:
    processed_ids = set()


# Function to process a chunk of rows
def process_chunk(device_id, chunk):
    torch.cuda.set_device(device_id)

    batch_size = 500
    batch = []
    batch_ids = []

    # print(type(chunk),chunk[0][1][0],processed_ids,device_id)
    for row in tqdm(chunk[device_id][1], total=len(chunk[device_id][1]), desc=f"Processing rows on GPU {device_id}"):
        if str(row['product_id']) in processed_ids:
            continue  # Skip already processed rows

        vector = process_row(row)
        if vector is not None:
            batch.append(vector)
            batch_ids.append(str(row['product_id']))
            processed_ids.add(str(row['product_id']))

            # Upload batch to Pinecone
            if len(batch) >= batch_size:
                with open(processed_ids_file, 'a') as f:
                    f.write("\n".join(batch_ids) + "\n")
                index.upsert(vectors=batch)
                batch = []
                batch_ids = []

    # Upload any remaining vectors to Pinecone
    if batch:
        with open(processed_ids_file, 'a') as f:
            f.write("\n".join(batch_ids) + "\n")
        index.upsert(vectors=batch)


if __name__ == '__main__':
    # Split the rows into chunks
    num_chunks = 2  # Number of GPUs
    chunk_size = len(rows) // num_chunks
    chunks = [rows[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    if len(rows) % num_chunks != 0:
        chunks[-1].extend(rows[chunk_size * num_chunks:])  # Add remaining rows to the last chunk

    # Create a list of arguments for spawn
    spawn_args = [(i, chunk) for i, chunk in enumerate(chunks)]

    # Use torch.multiprocessing.spawn to run each chunk on a separate GPU
    torch.multiprocessing.spawn(process_chunk,
                                args=(spawn_args,),
                                nprocs=num_chunks,
                                join=True)

    print("Processing complete and vectors uploaded to Pinecone!")
