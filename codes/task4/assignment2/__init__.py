import logging.config

from flask import Flask

from assignment2 import views
from assignment2.logging import init_logging


def create_app(config_overrides=None):
    init_logging()  # should be configured before any access to app.logger

    app = Flask(__name__)
    global base_dir
    base_dir = app.root_path
    app.config.from_object("assignment2.defaults")
    app.config.from_prefixed_env()

    if config_overrides is not None:
        app.config.from_mapping(config_overrides)

    app.register_blueprint(views.bp)

    return app
