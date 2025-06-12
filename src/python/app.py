from flask import Flask
from routes import create_blueprints


def create_app():
    """Initialize Flask app and register blueprints."""
    app = Flask(__name__)

    # ✅ Register all blueprints
    for blueprint in create_blueprints():
        app.register_blueprint(blueprint)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=6000)
