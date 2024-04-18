from flask import Flask

# Initialize Flask app
app = Flask(__name__)

# Import routes from routes.py
from app import routes