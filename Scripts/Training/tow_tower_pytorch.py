import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/final_filtered.csv')

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode user and product IDs
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['User_ID_encoded'] = user_encoder.fit_transform(df['User_ID'])
df['product_id_encoded'] = item_encoder.fit_transform(df['product_id_cleaned'])

# Dataset class
class RecommendationDataset(Dataset):
    def __init__(self, user_ids, item_ids, targets):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.targets[idx]

# Prepare dataset
user_ids = df['User_ID_encoded'].values
item_ids = df['product_id_encoded'].values
targets = df['interaction_strength'].values/4

# Split into training and validation sets
train_user_ids, val_user_ids, train_item_ids, val_item_ids, train_targets, val_targets = train_test_split(
    user_ids, item_ids, targets, test_size=0.2, random_state=42
)

# Create DataLoader
batch_size = 64  # Increase batch size for better GPU utilization
num_workers = 4  # Number of workers for DataLoader

train_dataset = RecommendationDataset(train_user_ids, train_item_ids, train_targets)
val_dataset = RecommendationDataset(val_user_ids, val_item_ids, val_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(TwoTowerModel, self).__init__()

        # User tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_dense1 = nn.Linear(embedding_dim, 64)
        self.user_bn1 = nn.BatchNorm1d(64)
        self.user_dense2 = nn.Linear(64, 32)

        # Item tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_dense1 = nn.Linear(embedding_dim, 64)
        self.item_bn1 = nn.BatchNorm1d(64)
        self.item_dense2 = nn.Linear(64, 32)

        # Final dense layers after concatenation
        self.dense3 = nn.Linear(32 + 32, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, 1)

        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_input, item_input):
        # User tower forward pass
        user_embedded = self.user_embedding(user_input)
        user_flat = user_embedded.view(user_embedded.size(0), -1)
        user_out = self.relu(self.user_bn1(self.user_dense1(user_flat)))
        user_out = self.dropout(user_out)
        user_out = self.user_dense2(user_out)

        # Item tower forward pass
        item_embedded = self.item_embedding(item_input)
        item_flat = item_embedded.view(item_embedded.size(0), -1)
        item_out = self.relu(self.item_bn1(self.item_dense1(item_flat)))
        item_out = self.dropout(item_out)
        item_out = self.item_dense2(item_out)

        # Concatenate towers
        combined = torch.cat((user_out, item_out), dim=1)

        # Final dense layers
        out = self.relu(self.bn3(self.dense3(combined)))
        out = self.dropout(out)

        # Output
        output = self.output_layer(out)
        return output
# Check for GPU


# Model initialization
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
embedding_dim = 256  # Set embedding dimension

model = TwoTowerModel(num_users, num_items, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Training loop
num_epochs = 20
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for user_batch, item_batch, target_batch in train_loader:
        user_batch, item_batch, target_batch = user_batch.to(device), item_batch.to(device), target_batch.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(user_batch, item_batch)
            loss = criterion(outputs.view(-1), target_batch.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')


# Validation loop
model.eval()
val_loss = 0

with torch.no_grad():
    for user_batch, item_batch, target_batch in val_loader:
        user_batch, item_batch, target_batch = user_batch.to(device), item_batch.to(device), target_batch.to(device)

        outputs = model(user_batch, item_batch)
        loss = criterion(outputs.view(-1), target_batch.float())
        val_loss += loss.item()

average_val_loss = val_loss / len(val_loader)
print(f'Validation Loss: {average_val_loss:.4f}')


# Recommendation function
def recommend_items(user_id, num_recommendations=10):
    if user_id in user_encoder.classes_:
        user_encoded = user_encoder.transform([user_id])[0]

        # Get interacted items
        interacted_items = df[df['User_ID'] == user_id]['product_id_encoded'].unique()

        # Prepare batches for prediction
        item_ids = df['product_id_encoded'].unique()
        item_ids = [item_id for item_id in item_ids if item_id not in interacted_items]  # Exclude interacted items

        # Prepare the tensors
        user_ids_batch = torch.tensor([user_encoded] * len(item_ids), dtype=torch.long).to(device)
        item_ids_batch = torch.tensor(item_ids, dtype=torch.long).to(device)

        # Batch prediction
        with torch.no_grad():
            predictions = model(user_ids_batch, item_ids_batch)

        # Collect predictions with item IDs
        predictions = list(zip(item_ids, predictions.cpu().numpy().flatten()))

        # Sort predictions by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Top recommendations
        top_recommendations = predictions[:num_recommendations]
        print("Top Recommendations:")
        for item_id, rating in top_recommendations:
          original_item_id = item_encoder.inverse_transform([int(item_id)])[0]
          print(f"Product ID: {original_item_id}, Predicted Rating: {rating*4}")

    else:
        # Recommend top items by average interaction strength for new users
        avg_interaction = df.groupby('product_id_encoded')['interaction_strength'].mean().reset_index()
        freq_count = df['product_id_encoded'].value_counts().reset_index()
        freq_count.columns = ['product_id_encoded', 'frequency']

        # Merge average interaction with frequency count
        recommendations = avg_interaction.merge(freq_count, on='product_id_encoded')

        # Sort first by average interaction strength, then by frequency
        recommendations = recommendations.sort_values(by=['interaction_strength', 'frequency'], ascending=False)

        top_recommendations = [(row['product_id_encoded'], row['interaction_strength']) for _, row in recommendations.iterrows()][:num_recommendations]

    # Print top recommendations
        print("Top Recommendations:")
        for item_id, rating in top_recommendations:
          original_item_id = item_encoder.inverse_transform([int(item_id)])[0]
          print(f"Product ID: {original_item_id}, Predicted Rating: {rating}")



# Save the model
torch.save(model.state_dict(), '/content/drive/MyDrive/recommendation_model.pth')

import joblib

# Save the user and item encoders
joblib.dump(user_encoder, '/content/drive/MyDrive/user_encoder.pkl')
joblib.dump(item_encoder, '/content/drive/MyDrive/item_encoder.pkl')

joblib.dump(df, '/content/drive/MyDrive/dataframe.pkl')
unique_item_ids = df['product_id_encoded'].unique()
joblib.dump(unique_item_ids, '/content/drive/MyDrive/unique_item_ids.pkl')


