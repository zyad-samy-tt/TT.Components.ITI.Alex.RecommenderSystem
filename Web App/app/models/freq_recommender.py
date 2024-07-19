import json
from flask import current_app


def get_recommendations(product_id, rules):
    relevant_rules = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    recommendations = set()
    for _, rule in relevant_rules.iterrows():
        recommendations.update(rule['consequents'])
    return list(recommendations)


def get_user_support_for_item(item, data):
    """Returns the list of users who bought the given item."""
    return data[data['product_id_cleaned'] == item]['User_ID'].unique()


def get_common_users_for_items(item1, item2, data):
    """Returns the list of users who bought both item1 and item2."""
    users_item1 = set(get_user_support_for_item(item1, data))
    users_item2 = set(get_user_support_for_item(item2, data))
    return users_item1.intersection(users_item2)


def explain_recommendations(item, item_rules, data):
    explanations = {}
    for _, rule in item_rules.iterrows():
        consequents = list(rule['consequents'])
        for consequent in consequents:
            common_users = get_common_users_for_items(item, consequent, data)
            if len(common_users) > 0:
                explanation = f"Product {consequent} is recommended with {item} because {len(list(common_users))} users  bought both together."
            else:
                similar_products = [other_consequent for other_consequent in consequents if
                                    other_consequent != consequent]
                if similar_products:
                    explanation = f"Product {consequent} is recommended with {item} because it is similar to products {similar_products} bought with {item}."
                else:
                    explanation = f"Product {consequent} is recommended with {item} based on the association rules."
            explanations[consequent] = explanation
    # Debug output to ensure explanations are correct
    current_app.logger.debug(f"Explanations: {json.dumps(explanations, indent=2)}")
    return explanations
