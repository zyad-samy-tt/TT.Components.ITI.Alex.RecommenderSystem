import tensorflow as tf

def get_user_based_recommendations(user_encoder, item_encoder, model, df, user_id, age=None, top_n=100):
    if user_id in df['User_ID'].values:
        user_encoded = user_encoder.transform([user_id])[0]

        # Get interacted items
        interacted_items = df[df['User_ID'] == user_id]['product_id_encoded'].unique()

        # Prepare batches for prediction
        item_ids = df['product_id_encoded'].unique()
        item_ids = [item_id for item_id in item_ids if item_id not in interacted_items]  # Exclude interacted items

        # Extract the region and age for all items
        region_encoded_series = df.drop_duplicates('product_id_encoded').set_index('product_id_encoded')['region_encoded']
        age_normalized_series = df.drop_duplicates('product_id_encoded').set_index('product_id_encoded')['age_normalized']

        # Prepare the tensors
        user_ids_batch = tf.convert_to_tensor([user_encoded] * len(item_ids), dtype=tf.int32)
        region_ids_batch = tf.convert_to_tensor(region_encoded_series.loc[item_ids].values, dtype=tf.int32)
        age_normalized_batch = tf.convert_to_tensor(age_normalized_series.loc[item_ids].values, dtype=tf.float32)
        item_ids_batch = tf.convert_to_tensor(item_ids, dtype=tf.int32)

        # Batch prediction
        predictions = model.predict([user_ids_batch, item_ids_batch, region_ids_batch, age_normalized_batch])

        # Collect predictions with item IDs
        predictions = list(zip(item_ids, predictions.flatten()))

        # Sort predictions by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Top recommendations
        top_recommendations = predictions[:top_n]  # Top 10 recommendations
    else:
        if age is not None:
            # Determine the age group
            age_group = None
            if age <= 12:
                age_group = 'Child'
            elif age <= 18:
                age_group = 'Teen'
            elif age <= 35:
                age_group = 'Adult'
            elif age <= 60:
                age_group = 'Middle Age'
            else:
                age_group = 'Senior'

            # Recommend items based on age group
            age_filtered_items = df[df['age_group'] == age_group]
            avg_interaction = age_filtered_items.groupby('product_id_encoded')['interaction_strength'].mean().reset_index()
            freq_count = age_filtered_items['product_id_encoded'].value_counts().reset_index()
            freq_count.columns = ['product_id_encoded', 'frequency']

            # Merge average interaction with frequency count
            recommendations = avg_interaction.merge(freq_count, on='product_id_encoded')

            # Sort first by average interaction strength, then by frequency
            recommendations = recommendations.sort_values(by=['interaction_strength', 'frequency'], ascending=False)

            top_recommendations = [(row['product_id_encoded'], row['interaction_strength']) for _, row in recommendations.iterrows()][:top_n]
        else:
            # Recommend top items by average interaction strength for new users without age
            avg_interaction = df.groupby('product_id_encoded')['interaction_strength'].mean().reset_index()
            freq_count = df['product_id_encoded'].value_counts().reset_index()
            freq_count.columns = ['product_id_encoded', 'frequency']

            # Merge average interaction with frequency count
            recommendations = avg_interaction.merge(freq_count, on='product_id_encoded')

            # Sort first by average interaction strength, then by frequency
            recommendations = recommendations.sort_values(by=['interaction_strength', 'frequency'], ascending=False)

            top_recommendations = [(row['product_id_encoded'], row['interaction_strength']) for _, row in recommendations.iterrows()][:top_n]

    original_item_ids = item_encoder.inverse_transform([int(item_id) for item_id, _ in top_recommendations])
    return original_item_ids.tolist()