from flask import render_template, request, jsonify
from . import main
from ..models.freq_recommender import get_recommendations
from ..models.similarity_recommender import get_similar_products
from ..models.user_based_recommender import get_user_based_recommendations
from ..models.aws import get_images
from ..utils.data_loader import load_data, load_fpgrowth_model, load_user_based_model

data, product_ids_sample, product_ids = load_data()
frequent_itemsets, rules = load_fpgrowth_model()
model, user_encoder, item_encoder, region_encoder, scaler, df = load_user_based_model()

@main.route('/')
def index():
    product_titles = data[data['product_id'].isin(product_ids_sample)]['title'].unique().tolist()
    return render_template('index.html', product_titles=product_titles)

@main.route('/recommend', methods=['POST'])
def recommend():
    product_title = request.form['product_title']
    product_id = data[data['title'] == product_title]['product_id'].values[0]
    recommendations = get_recommendations(product_id, rules)
    images = get_images(recommendations)
    selected_product_image = get_images([product_id]).get(product_id)
    return jsonify({'product_id': product_id, 'product_title': product_title, 'recommendations': recommendations, 'images': images, 'selected_product_image': selected_product_image})

@main.route('/image-recommender')
def image_recommender():
    product_titles = data[data['product_id'].isin(product_ids_sample)]['title'].unique().tolist()
    return render_template('image_recommender.html', product_titles=product_titles)

@main.route('/image-recommend', methods=['POST'])
def image_recommend():
    product_title = request.form['product_title']
    product_id = data[data['title'] == product_title]['product_id'].values[0]
    results = get_similar_products(product_id, 10)[1:]
    recommendations = [result.id for result in results]
    images = get_images(recommendations)
    selected_product_image = get_images([product_id]).get(product_id)
    return jsonify({'product_id': product_id, 'product_title': product_title, 'recommendations': recommendations, 'images': images, 'selected_product_image': selected_product_image})

@main.route('/user_recommender')
def user_recommender():
    user_ids = df['User_ID'].sample(10).values.tolist()
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
    age = request.form.get('age')
    return jsonify({"user_id": user_id, "age": age})

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
        recommendations = get_user_based_recommendations(user_encoder, item_encoder, model, df, float(user_id), int(age), 100)
    else:
        recommendations = get_user_based_recommendations(user_encoder, item_encoder, model, df, float(user_id), age=None, top_n=100)
    images = get_images(recommendations)
    return jsonify({'user_id': user_id, 'recommendations': recommendations, 'images': images})