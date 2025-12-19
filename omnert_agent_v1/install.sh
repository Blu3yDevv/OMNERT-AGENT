#!/bin/bash

echo "🚀 Installing OMNERT-AGENT..."

# Install Python dependencies
pip install langchain-groq==0.1.3
pip install langchain-community==0.0.25
pip install langchain-core==0.1.41
pip install tavily-python==0.3.4
pip install python-dotenv==1.0.0
pip install httpx==0.26.0
pip install pydantic==2.5.3

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your API keys"
echo "2. Run: python main.py"
echo "3. For testing: python main.py --test"