# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
DEBUG = os.getenv('DEBUG', 'True') == 'True'
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')

# Remote Model settings
MODEL_API_URL = os.getenv('MODEL_API_URL', 'https://your-ngrok-url.ngrok-free.app/generate')
MODEL_API_TIMEOUT = int(os.getenv('MODEL_API_TIMEOUT', 30))  # Timeout in seconds

# BigQuery settings
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
PROJECT_ID = os.getenv('PROJECT_ID', '')
DEFAULT_DATASET = os.getenv('DEFAULT_DATASET', '')

# CrewAI settings (if using CrewAI)
CREW_VERBOSE = os.getenv('CREW_VERBOSE', 'True') == 'True'
CREW_PROCESS = os.getenv('CREW_PROCESS', 'sequential')