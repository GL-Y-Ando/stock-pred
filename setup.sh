#!/bin/bash

# Setup script for Stock Prediction System in Docker

echo "🚀 Setting up Stock Prediction System in Docker..."

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p src data/price_data models output logs processed_data cache notebooks

# Move Python files to src directory if they're in root
echo "📦 Organizing Python files..."
if [ -f "main.py" ]; then
    mv *.py src/ 2>/dev/null || true
fi

# Create default .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating default .env file..."
    cat > .env << EOF
# Stock Prediction System Environment Variables

# API Configuration (if using external APIs)
API_URL=https://api.jquants.com
REFRESH_TOKEN=your_refresh_token_here

# System Configuration
LOG_LEVEL=INFO
TENSORFLOW_CPP_MIN_LOG_LEVEL=2

# Optional: Database Configuration
# DATABASE_URL=postgresql://stockuser:stockpass@postgres:5432/stockdb
EOF
fi

# Create default config.json if it doesn't exist
if [ ! -f "config.json" ]; then
    echo "⚙️ Creating default configuration..."
    cat > config.json << EOF
{
  "data": {
    "stock_list_path": "data/stock_list.csv",
    "price_data_path": "data/price_data/",
    "train_ratio": 0.8,
    "validation_ratio": 0.1,
    "test_ratio": 0.1,
    "min_data_points": 100
  },
  "model": {
    "short_term_window": 5,
    "long_term_window": 25,
    "lookback_period": 60,
    "hidden_layers": [64, 32, 16],
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
  },
  "prediction": {
    "confidence_threshold": 0.7,
    "ensemble_size": 5
  },
  "system": {
    "log_level": "INFO",
    "model_save_path": "models/",
    "output_path": "output/",
    "n_jobs": -1
  }
}
EOF
fi

# Build and run the Docker container
echo "🐳 Building Docker container..."
docker-compose build

echo "🎉 Setup complete! You can now use the following commands:"
echo ""
echo "Start the container:"
echo "  docker-compose up -d"
echo ""
echo "Access the container:"
echo "  docker-compose exec stock-prediction bash"
echo ""
echo "Run stock predictions:"
echo "  docker-compose exec stock-prediction python src/main.py --mode predict"
echo ""
echo "View logs:"
echo "  docker-compose logs -f stock-prediction"
echo ""
echo "Stop the container:"
echo "  docker-compose down"