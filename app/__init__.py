"""RAG Document QA Service — Flask application factory."""

import os

from flask import Flask

from app.config import Config


def create_app(config=None):
    """Create and configure the Flask application.

    Args:
        config: Configuration object or dict. Defaults to Config.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    if config is None:
        app.config.from_object(Config)
    elif isinstance(config, dict):
        app.config.from_mapping(config)
    else:
        app.config.from_object(config)

    # Ensure required directories exist
    os.makedirs(os.path.dirname(app.config["DATABASE_PATH"]), exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize database
    from app.models.database import init_db

    init_db(app)

    # Register API
    from app.api import register_api

    register_api(app)

    return app
