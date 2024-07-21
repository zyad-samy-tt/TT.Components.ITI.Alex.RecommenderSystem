

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


df=pd.read_csv("/content/drive/MyDrive/item-user.csv")
avg_embeddings=pd.read_csv("/content/drive/MyDrive/avg_embeddings.csv")
avg_embeddings['embedding'] = avg_embeddings['embedding_str'].apply(lambda x: np.fromstring(x, sep=','))
embeddings_matrix = np.stack(avg_embeddings['embedding'].values)
similarity_matrix = cosine_similarity(embeddings_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=avg_embeddings['User_ID'], columns=avg_embeddings['User_ID'])


# Function to get top N similar users
def get_top_n_similar_users(similarity_df, user_id, n=5):
    if user_id not in similarity_df.index:
        raise ValueError(f"User {user_id} not found in similarity matrix.")
    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:n+1]
    return similar_users

def recommend_items(df, similarity_df, user_id, top_n_users=5, top_n_items=5):
    if user_id in df['User_ID'].values:

      # Get similar users
      similar_users = get_top_n_similar_users(similarity_df, user_id, top_n_users)
      similar_users_data = df[df['User_ID'].isin(similar_users)]

      # Aggregate interaction strengths (using max to avoid high sums)
      item_scores = similar_users_data.groupby('product_id_cleaned')['interaction_strength'].max().reset_index()

      # Get items already interacted with by the target user
      user_items = df[df['User_ID'] == user_id]['product_id_cleaned'].tolist()

      # Exclude already interacted items
      item_scores = item_scores[~item_scores['product_id_cleaned'].isin(user_items)]

      # Sort by interaction strength and get the top N items
      recommended_items = item_scores.sort_values(by='interaction_strength', ascending=False).head(top_n_items)
      return recommended_items, similar_users
    else:
            avg_interaction = df.groupby('product_id_cleaned')['interaction_strength'].mean().reset_index()
            freq_count = df['product_id_cleaned'].value_counts().reset_index()
            freq_count.columns = ['product_id_cleaned', 'frequency']

            # Merge average interaction with frequency count
            recommendations = avg_interaction.merge(freq_count, on='product_id_cleaned')

            # Sort first by average interaction strength, then by frequency
            recommendations = recommendations.sort_values(by=['interaction_strength', 'frequency'], ascending=False)

            top_recommendations = [(row['product_id_cleaned'], row['interaction_strength']) for _, row in recommendations.iterrows()][:10]
            return top_recommendations


# Example user in data
user_id = 15.0
recommended_items ,similar_users= recommend_items(df, similarity_df, user_id)
print(f"This user {user_id} is simiar for {similar_users.values}")

print(recommended_items)

# Example user in not in data
user_id = 99999.0
recommended_items = recommend_items(df, similarity_df, user_id)
print("Top Recommendations:")
for item_id, rating in recommended_items:
        print(f"Product ID: {item_id}, Predicted Rating: {rating}")

