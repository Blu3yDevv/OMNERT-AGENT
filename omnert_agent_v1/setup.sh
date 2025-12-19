#!/bin/bash

# OMNERT-AGENT Setup Script

echo "🚀 Setting up OMNERT-AGENT..."

# Create virtual environment
python3 -m venv venv
source venm/*ctivate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install langchain-groq==0.1.3
pip install langchain-community==0.0.25
pip install langchain-core==0.1.41
pip install langgraph==0.0.34
pip install langchain-experimental==0.0.52
pip install tavily-python==0.3.4
pip install python-dotenv==1.0.0
pip install httpx==0.26.0
pip install pydantic==2.5.3

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# API Keys (REQUIRED)" > .env
    echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
    echo "TAVILY_API_KEY=your_tavily_api_key_here" >> .env
    echo "" >> .env
    echo "# Optional Configuration" >> .env
    echo "LOG_LEVEL=INFO" >> .env
    echo "MAX_SEARCH_RESULTS=5" >> .env
    echo "" >> .env
    echo "⚠️  Please edit .env file and add your API keys!"
fi

echo "✅ Setup complete!"
echo ""
echo "To run the agent:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python main.py"
echo "3. For test: python main.py --test"