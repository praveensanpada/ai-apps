# app/__init__.py

from flask import Flask

# Initialize Flask app
app = Flask(__name__)

# Import the routes after the app is initialized to avoid circular imports
from app import routes
