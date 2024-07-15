import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path', type=str, required=True, help='Path to the input data CSV file.')
parser.add_argument('--model_path', type=str, required=True, help='Folder Path to save the output model pickle file.')

args = parser.parse_args()

data = pd.read_csv(args.data_path, low_memory=False)

data['time'] = pd.to_datetime(data['time'])

# Define a session threshold (60 minutes)
session_threshold = pd.Timedelta(minutes=60)

data['session_id'] = (data['time'] - data.groupby('User_ID')['time'].shift(1) <= session_threshold).cumsum()

# Group by session and list products
sessions = data.groupby(['User_ID', 'session_id'])['product_id_cleaned'].apply(list).reset_index()

sessions['product_id_cleaned'] = sessions['product_id_cleaned'].apply(lambda x: list(dict.fromkeys(x)))

sessions = sessions[sessions['product_id_cleaned'].map(len) > 1]

transactions = sessions['product_id_cleaned'].to_list()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.00005, use_colnames=True, verbose=1)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

# Save the frequent itemsets and rules
with open(os.path.join(args.model_path, 'fpgrowth_model.pkl'), 'wb') as f:
    pickle.dump((frequent_itemsets, rules), f)

print("Model saved successfully.")