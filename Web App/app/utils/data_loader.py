import pandas as pd
import os
import pickle
import joblib
import tensorflow as tf


def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(current_dir, 'products_ids_title.csv'), low_memory=False)
    user_data = pd.read_csv(os.path.join(current_dir, 'all_filtered_df1&df3.csv'), low_memory=False)
    product_ids_sample = data['product_id'].sample(1000).dropna().unique().tolist()
    product_ids = data['product_id'].dropna().unique().tolist()

    return user_data, data, product_ids_sample, product_ids


def load_fpgrowth_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(current_dir, 'fpgrowth_model.pkl')
    with open(pickle_path, 'rb') as f:
        frequent_itemsets, rules = pickle.load(f)
    return frequent_itemsets, rules


def load_user_based_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model(os.path.join(current_dir, 'user_based_model/my_model.h5'))
    user_encoder = joblib.load(os.path.join(current_dir, 'user_based_model/user_encoder.pkl'))
    item_encoder = joblib.load(os.path.join(current_dir, 'user_based_model/item_encoder.pkl'))
    region_encoder = joblib.load(os.path.join(current_dir, 'user_based_model/region_encoder.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'user_based_model/age_scaler.pkl'))
    df = pd.read_pickle(os.path.join(current_dir, 'user_based_model/dataframe.pkl'))
    return model, user_encoder, item_encoder, region_encoder, scaler, df
