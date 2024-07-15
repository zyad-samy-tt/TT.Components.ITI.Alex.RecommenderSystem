from flask import Flask
import os

def create_app():
    app = Flask(__name__, template_folder=os.path.abspath('app/main/templates'),static_folder=os.path.abspath('static'))

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app