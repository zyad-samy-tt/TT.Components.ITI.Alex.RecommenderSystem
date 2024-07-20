from flask import render_template, request, jsonify
from . import main
from ..models.freq_recommender import get_recommendations, explain_recommendations
from ..models.similarity_recommender import get_similar_products, get_index_for_model
from ..models.user_based_recommender import get_user_based_recommendations
from ..models.aws import get_images
from ..utils.data_loader import load_data, load_fpgrowth_model, load_user_based_model
import json
from flask import current_app

user_data, data, product_ids_sample, product_ids = load_data()
frequent_itemsets, rules = load_fpgrowth_model()
model, user_encoder, item_encoder, region_encoder, scaler, df = load_user_based_model()



@main.route('/')
def index():
    return render_template('index.html')


@main.route('/freq-bought-recommender')
def freq_bought_recommender():
    product_titles = data[data['product_id'].isin(product_ids_sample)]['title'].unique().tolist()
    return render_template('freq-bought-recommender.html', product_titles=product_titles)


@main.route('/recommend', methods=['POST'])
def recommend():
    product_title = request.form['product_title']
    product_id = data[data['title'] == product_title]['product_id'].values[0]
    recommendations = get_recommendations(product_id, rules)
    item_rules = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    explanations = explain_recommendations(product_id, item_rules, user_data)
    images = get_images(recommendations)
    selected_product_image = get_images([product_id]).get(product_id)
    # Debug output to ensure JSON response is correct
    current_app.logger.debug(
        f"Response: {json.dumps({'product_id': product_id, 'product_title': product_title, 'recommendations': recommendations, 'images': images, 'selected_product_image': selected_product_image, 'explanations': explanations}, indent=2)}")
    return jsonify(
        {'product_id': product_id, 'product_title': product_title, 'recommendations': recommendations, 'images': images,
         'selected_product_image': selected_product_image, 'explanations': explanations})


@main.route('/similarity-recommender')
def image_recommender():
    product_titles = data[data['product_id'].isin(product_ids_sample)]['title'].unique().tolist()
    return render_template('similarity_recommender.html', product_titles=product_titles)


# @main.route('/image-recommend', methods=['POST'])
# def image_recommend():
#     product_title = request.form['product_title']
#     product_id = data[data['title'] == product_title]['product_id'].values[0]
#     results = get_similar_products(product_id, 100)[1:]
#     recommendations = [result.id for result in results]
#     scores = [f"{result.score:.4f}" for result in results]
#     images = get_images(recommendations)
#     selected_product_image = get_images([product_id]).get(product_id)
#     return jsonify(
#         {'product_id': product_id, 'product_title': product_title, 'recommendations': recommendations, 'scores': scores,
#          'images': images, 'selected_product_image': selected_product_image})


@main.route('/image-recommend', methods=['POST'])
def image_recommend():
    model_name = request.form['model']  # Get selected model
    index_name = get_index_for_model(model_name)
    product_title = request.form['product_title']
    product_id = data[data['title'] == product_title]['product_id'].values[0]
    results = get_similar_products(index_name, product_id, 100)[1:]
    recommendations = [result.id for result in results]
    scores = [f"{result.score:.4f}" for result in results]
    images = get_images(recommendations)
    selected_product_image = get_images([product_id]).get(product_id)
    return jsonify(
        {'product_id': product_id, 'product_title': product_title, 'recommendations': recommendations, 'scores': scores,
         'images': images, 'selected_product_image': selected_product_image})


@main.route('/user_recommender')
def user_recommender():
    user_ids = df['User_ID'].sample(1000).values.tolist()
    return render_template('user-recommender.html', user_ids=user_ids)


@main.route('/user_recommend', methods=['POST'])
def user_recommend():
    user_id = request.form.get('user_id')

    user_exists = df['User_ID'].isin([float(user_id)]).any()
    if user_exists:
        return jsonify({"user_id": user_id, "new_user": False})
    else:
        return jsonify({"new_user": True})


@main.route('/new_user_recommend', methods=['POST'])
def new_user_recommend():
    user_id = request.form.get('user_id')
    # age = request.form.get('age')
    return jsonify({"user_id": user_id})


@main.route('/user_recommendations')
def user_recommendations():
    user_id = request.args.get('user_id')
    age = request.args.get('age')
    return render_template('user-recommendations.html', user_id=user_id, age=age)


@main.route('/get_user_recommendations', methods=['GET'])
def get_user_recommendations():
    user_id = request.args.get('user_id')
    age = request.args.get('age')
    if age:
        recommendations = get_user_based_recommendations(user_encoder, item_encoder, model, df, float(user_id),
                                                         int(age), 100)
    else:
        recommendations = get_user_based_recommendations(user_encoder, item_encoder, model, df, float(user_id),
                                                         age=None, top_n=100)
    images = get_images(recommendations)
    return jsonify({'user_id': user_id, 'recommendations': recommendations, 'images': images})
