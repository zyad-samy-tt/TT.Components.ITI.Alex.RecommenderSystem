import pandas as pd
import numpy as np
import boto3
import pandas as pd
from io import StringIO
import os
from transformers import CLIPProcessor, CLIPModel
import torch
df2=pd.read_csv("/content/drive/MyDrive/Cleaning_Product_FullData.csv")
df3=pd.read_csv("/content/drive/MyDrive/all_filtered_df1&df3.csv")



# Load the model and processor
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Function to get embeddings with proper handling
def get_embedding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)  # Set max_length as needed
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.numpy().flatten()
final=df3[["User_ID","product_id_cleaned","event_name"]]
final.dropna(subset=['product_id_cleaned'],inplace=True)
final.dropna(subset=['User_ID'],inplace=True)
final.drop_duplicates(subset=['User_ID','product_id_cleaned',"event_name"],inplace=True)

interaction_strength = {
    'product_similar_items_clicking': 1,
    'product_details_page_2_view': 2,
    'product_details_page_View': 2,
    'generic_clicking': 3,
    'ButtonClick': 3,
    'shopping_bag_clicking': 4
}

final['interaction_strength'] = final['event_name'].map(interaction_strength)
final.drop(["event_name"],axis=1,inplace=True)

aws_access_key_id = 'aws_access_key_id'
aws_secret_access_key = 'aws_secret_access_key'
region_name = 'region_name'

# Initialize a session
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Initialize S3 resource
s3 = session.resource('s3')
bucket_name = 'tt-iti'
bucket = s3.Bucket(bucket_name)

# List to hold image paths
image_paths = []

# List all images
for obj in bucket.objects.filter(Prefix="ITI Alexandria Branch/images/"):
    if obj.key.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        # Append the S3 path to the list
        image_paths.append(obj.key)

# Save paths to a text file
with open('image_paths.txt', 'w') as f:
    for path in image_paths:
        f.write(f"{path}\n")

desired_parts = [os.path.splitext(os.path.basename(path))[0].split('_')[0] for path in image_paths]

unique_parts = list(set(desired_parts))
final_filtered = final[final['product_id_cleaned'].isin(unique_parts)]
final_filtered=final_filtered.merge(df2[['product_id', 'title',"Original_Description","department"]],
              left_on='product_id_cleaned',
              right_on='product_id',
              how='left')
final_filtered['title'] = final_filtered['title'].fillna(final_filtered['Original_Description'])
final_filtered['title'] = final_filtered['title'].fillna(final_filtered['department'])
final_filtered.dropna(subset=['title'],inplace=True)
final_filtered.to_csv("/content/drive/MyDrive/final_filtered.csv",index=False)
products=final_filtered.drop_duplicates(subset=['product_id_cleaned'])[['product_id_cleaned',"title"]]
products["embedding"]=products["title"].apply(get_embedding)
products['embedding_str'] = products['embedding'].apply(lambda x: ','.join(map(str, x)))
products.to_csv("/content/drive/MyDrive/product_embeddings.csv",index=False)

final_filtered=final_filtered.merge(products[["product_id_cleaned","embedding"]],on="product_id_cleaned",how="left")
final_filtered['embedding'] = final_filtered['embedding'].apply(np.array)

avg_embeddings = final_filtered.groupby('User_ID')['embedding'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()

avg_embeddings.rename(columns={'embedding': 'avg_embedding'}, inplace=True)
avg_embeddings['embedding_str'] = avg_embeddings['avg_embedding'].apply(lambda x: ','.join(map(str, x)))
avg_embeddings.drop("avg_embedding",axis=1).to_csv("/content/drive/MyDrive/avg_embeddings.csv",index=False)

