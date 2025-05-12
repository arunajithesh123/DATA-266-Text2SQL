#!/bin/bash
# deployment.sh - Script to deploy the NL to SQL application with remote model

# Set error handling
set -e

# Default configuration
APP_NAME="nl-to-sql-app"
MODEL_API_URL="https://your-ngrok-url.ngrok-free.app/generate"
PROJECT_DIR="$(pwd)"
ENV_FILE=".env"
DEPLOY_TARGET="local"  # Options: local, gcloud
PORT=8080
HOST="0.0.0.0"

# Function to display usage information
show_usage() {
    echo "Usage: ./deployment.sh [OPTIONS]"
    echo "Deploy the Natural Language to SQL application"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message and exit"
    echo "  -u, --model-url URL     URL to the remote model API"
    echo "  -p, --port PORT         Port to run the application on (default: 8080)"
    echo "  -t, --target TARGET     Deployment target (local, gcloud)"
    echo "  -e, --env-file FILE     Path to environment file"
    echo ""
    echo "Example:"
    echo "  ./deployment.sh --model-url https://your-ngrok-url.ngrok-free.app/generate --target gcloud"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -u|--model-url)
            MODEL_API_URL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -t|--target)
            DEPLOY_TARGET="$2"
            shift 2
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating $ENV_FILE file..."
    cat > "$ENV_FILE" << EOF
# Application settings
DEBUG=True
PORT=$PORT
HOST=$HOST

# Remote Model settings
MODEL_API_URL=$MODEL_API_URL
MODEL_API_TIMEOUT=30

# BigQuery settings
GOOGLE_APPLICATION_CREDENTIALS=
PROJECT_ID=
DEFAULT_DATASET=

# CrewAI settings
CREW_VERBOSE=True
CREW_PROCESS=sequential
EOF
    echo "Please update $ENV_FILE with your Google Cloud credentials and project details."
fi

# Deploy based on target
case $DEPLOY_TARGET in
    local)
        echo "Starting application locally..."
        export FLASK_APP=app.py
        export FLASK_ENV=development
        flask run --host=$HOST --port=$PORT
        ;;
    gcloud)
        echo "Deploying to Google Cloud Run..."
        
        # Check if gcloud is installed
        if ! command -v gcloud &> /dev/null; then
            echo "Error: gcloud command not found. Please install Google Cloud SDK."
            exit 1
        fi
        
        # Create Dockerfile if it doesn't exist
        if [ ! -f "Dockerfile" ]; then
            echo "Creating Dockerfile..."
            cat > "Dockerfile" << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Load environment variables from .env file
ENV $(cat $ENV_FILE | grep -v '^#' | xargs)

# Expose port
EXPOSE $PORT

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
EOF
        fi
        
        # Build and deploy to Cloud Run
        echo "Building and deploying to Google Cloud Run..."
        gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/$APP_NAME
        gcloud run deploy $APP_NAME \
            --image gcr.io/$(gcloud config get-value project)/$APP_NAME \
            --platform managed \
            --allow-unauthenticated
        ;;
    *)
        echo "Error: Unknown deployment target: $DEPLOY_TARGET"
        show_usage
        exit 1
        ;;
esac

echo "Deployment completed successfully!"