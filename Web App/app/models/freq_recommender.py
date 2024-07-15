def get_recommendations(product_id, rules):
    relevant_rules = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    recommendations = set()
    for _, rule in relevant_rules.iterrows():
        recommendations.update(rule['consequents'])
    return list(recommendations)
