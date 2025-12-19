#!/bin/bash
echo "Setting up OmniAgent v2..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install langchain-groq tavily-python python-dotenv httpx
echo "✅ Setup complete!"
echo "Add API keys to .env file:"
echo "GROQ_API_KEY=your_key_here"
echo "TAVILY_API_KEY=your_key_here"