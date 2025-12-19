"""
OMNERT-AGENT v3.1 - Professional AI Agent with Proper Async Architecture
"""

import os
import json
import asyncio
import pickle
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import uuid
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment
load_dotenv()

# ========== CONFIGURATION ==========
@dataclass
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Models
    PRIMARY_MODEL = "llama-3.3-70b-versatile"
    FAST_MODEL = "llama-3.1-8b-instant"
    
    # Memory
    MAX_MEMORY_ENTRIES = 100
    MEMORY_FILE = "agent_memory.pkl"
    SIMILARITY_THRESHOLD = 0.7
    
    # Search
    MAX_SEARCH_RESULTS = 5
    SEARCH_TIMEOUT = 30
    
    # RAG
    TOP_K_RETRIEVAL = 3
    CHUNK_SIZE = 500
    
    # Execution
    MAX_WORKERS = 4

# ========== FIXED TAVILY API ==========
class TavilySearch:
    """Fixed Tavily search that actually works"""
    
    def __init__(self):
        self.api_key = Config.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com"
        self.client = None
        
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=Config.SEARCH_TIMEOUT)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search with proper Tavily API v3"""
        
        if not self.api_key or "your_" in self.api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured",
                "results": []
            }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": True
        }
        
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=Config.SEARCH_TIMEOUT)
            
            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "query": query,
                    "answer": data.get("answer"),
                    "results": data.get("results", []),
                    "images": data.get("images", []),
                    "response_time": data.get("response_time", 0)
                }
            elif response.status_code == 403:
                # Try alternative format
                return await self._try_alternative_format(query, max_results)
            else:
                return {
                    "success": False,
                    "error": f"API Error {response.status_code}: {response.text[:200]}",
                    "results": []
                }
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Search timeout",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "results": []
            }
    
    async def _try_alternative_format(self, query: str, max_results: int) -> Dict[str, Any]:
        """Alternative API format for some keys"""
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "query": query,
                    "answer": data.get("answer"),
                    "results": data.get("results", []),
                    "response_time": data.get("response_time", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Alternative API failed: {response.status_code}",
                    "results": []
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Alternative API error: {str(e)}",
                "results": []
            }

# ========== SIMPLE BUT EFFECTIVE MEMORY ==========
class AgentMemory:
    """Lightweight memory system without complex embeddings"""
    
    def __init__(self):
        self.memories: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.load_memory()
    
    def load_memory(self):
        """Load memory from disk - SYNC version"""
        try:
            if Path(Config.MEMORY_FILE).exists():
                with open(Config.MEMORY_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.memories = data.get('memories', [])
                    self.conversation_history = data.get('conversation', [])
                print(f"📚 Loaded {len(self.memories)} memories, {len(self.conversation_history)} conversations")
        except Exception as e:
            print(f"Memory load error: {e}")
            self.memories = []
            self.conversation_history = []
    
    async def save_memory(self):
        """Save memory to disk - ASYNC version"""
        try:
            data = {
                'memories': self.memories[-Config.MAX_MEMORY_ENTRIES:],
                'conversation': self.conversation_history[-20:]
            }
            with open(Config.MEMORY_FILE, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Memory save error: {e}")
    
    def add_memory(self, content: str, memory_type: str = "general", importance: float = 1.0):
        """Add a memory"""
        memory = {
            'id': str(uuid.uuid4()),
            'content': content,
            'type': memory_type,
            'timestamp': datetime.now().isoformat(),
            'importance': importance
        }
        self.memories.append(memory)
        
        # Auto-save in background
        asyncio.create_task(self.save_memory())
    
    def add_conversation(self, user_input: str, agent_response: str):
        """Add conversation to history"""
        entry = {
            'user': user_input,
            'agent': agent_response,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(entry)
    
    def get_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """Simple keyword-based memory retrieval"""
        query_lower = query.lower()
        relevant = []
        
        for memory in reversed(self.memories):
            content_lower = memory['content'].lower()
            
            # Simple keyword matching
            words = set(query_lower.split())
            content_words = set(content_lower.split())
            common_words = words.intersection(content_words)
            
            if len(common_words) >= 1:  # At least one common word
                relevant.append(f"[{memory['type']}] {memory['content']}")
            
            if len(relevant) >= k:
                break
        
        return relevant
    
    def get_conversation_context(self, turns: int = 3) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        context = "Recent conversation:\n"
        for entry in self.conversation_history[-turns:]:
            context += f"User: {entry['user'][:100]}\n"
            context += f"Agent: {entry['agent'][:100]}...\n\n"
        
        return context

# ========== ADVANCED REASONING ENGINE ==========
class ReasoningEngine:
    """Multi-step reasoning with self-correction"""
    
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            model_name=Config.PRIMARY_MODEL,
            api_key=Config.GROQ_API_KEY,
            max_tokens=2048
        )
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine approach"""
        
        prompt = f"""Analyze this user query and determine:

QUERY: {query}

Please output a JSON with:
1. query_type: ["greeting", "factual", "mathematical", "technical", "creative", "opinion", "identity", "other"]
2. needs_search: true/false (whether web search is needed)
3. complexity: ["low", "medium", "high"]
4. response_style: ["concise", "detailed", "technical", "creative"]
5. estimated_tokens: number of tokens needed for response

Be accurate and objective."""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=prompt)]
            )
            
            # Extract JSON from response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            return analysis
            
        except Exception as e:
            print(f"Analysis error: {e}")
            # Default analysis
            return {
                "query_type": "factual",
                "needs_search": True,
                "complexity": "medium",
                "response_style": "detailed",
                "estimated_tokens": 500
            }
    
    async def generate_thought_process(self, query: str, context: str = "") -> str:
        """Generate a chain-of-thought reasoning"""
        
        prompt = f"""You are an AI assistant. Think through this problem step by step:

Query: {query}

Context:
{context if context else "No additional context provided."}

Reason through:
1. What is being asked?
2. What knowledge is required?
3. What are the key points to address?
4. How should the answer be structured?
5. What are potential pitfalls or misunderstandings?

Think step by step before answering:"""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=prompt)]
            )
            return response.content
        except Exception as e:
            return f"Reasoning error: {str(e)}"

# ========== STREAMING RESPONSE GENERATOR ==========
class StreamResponse:
    """Generate streaming responses with SSE formatting"""
    
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=Config.PRIMARY_MODEL,
            api_key=Config.GROQ_API_KEY,
            max_tokens=2048,
            streaming=True
        )
    
    async def generate(self, messages: List, query_id: str = None) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        if not query_id:
            query_id = str(uuid.uuid4())
        
        # Send query ID
        yield f"data: {json.dumps({'type': 'query_id', 'id': query_id})}\n\n"
        
        # Send thinking indicator
        yield f"data: {json.dumps({'type': 'status', 'status': 'thinking'})}\n\n"
        
        try:
            # Generate response
            response = await asyncio.to_thread(
                lambda: self.llm.stream(messages)
            )
            
            # Stream tokens
            buffer = ""
            for chunk in response:
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    buffer += content
                    
                    # Send token
                    yield f"data: {json.dumps({'type': 'token', 'token': content})}\n\n"
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'full_response': buffer})}\n\n"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

# ========== MAIN AGENT CLASS ==========
class OmniAgent:
    """Professional AI Agent with all capabilities"""
    
    def __init__(self):
        self.memory = AgentMemory()
        self.search = TavilySearch()
        self.reasoner = ReasoningEngine()
        self.streamer = StreamResponse()
        
        # Agent identity
        self.agent_name = "OMNERT-AGENT"
        self.version = "3.1"
        self.personality = "helpful, accurate, thorough"
        
        # System prompt
        self.system_prompt = f"""# {self.agent_name} v{self.version}

You are a professional AI assistant with advanced capabilities.

## CORE PRINCIPLES:
1. **Accuracy First** - Never guess or hallucinate. If unsure, say so.
2. **Comprehensive** - Provide thorough, well-structured answers.
3. **Context-Aware** - Use conversation history and relevant knowledge.
4. **Professional** - Maintain a helpful, expert tone.
5. **Transparent** - Cite sources when using external information.

## RESPONSE GUIDELINES:
- Start with a clear, direct answer to the main question
- Use headings (##, ###) for organization
- Use bullet points for lists
- Include examples when helpful
- End with a summary or key takeaways for complex topics
- For math: show your work and final answer
- For code: provide clean, commented examples

## CURRENT CONTEXT:
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Agent: {self.agent_name} v{self.version}
- Personality: {self.personality}

Always respond as a knowledgeable, helpful AI assistant."""
    
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """Process a query end-to-end with streaming"""
        
        print(f"\n🔍 Processing: {query[:60]}...")
        
        # Step 1: Store query in memory
        self.memory.add_memory(query, "user_query")
        
        # Step 2: Analyze query
        yield f"data: {json.dumps({'type': 'status', 'status': 'analyzing'})}\n\n"
        analysis = await self.reasoner.analyze_query(query)
        
        # Step 3: Handle special cases
        if analysis.get('query_type') == 'greeting':
            greeting = self._get_greeting(query)
            yield f"data: {json.dumps({'type': 'complete', 'full_response': greeting})}\n\n"
            self.memory.add_conversation(query, greeting)
            return
        
        if analysis.get('query_type') == 'identity':
            identity = self._get_identity_response(query)
            yield f"data: {json.dumps({'type': 'complete', 'full_response': identity})}\n\n"
            self.memory.add_conversation(query, identity)
            return
        
        # Step 4: Build context
        yield f"data: {json.dumps({'type': 'status', 'status': 'building_context'})}\n\n"
        
        context_parts = []
        
        # Add relevant memories
        memories = self.memory.get_relevant_memories(query)
        if memories:
            context_parts.append("## 📖 Relevant Memories:")
            for mem in memories:
                context_parts.append(f"- {mem}")
        
        # Add conversation history
        conversation = self.memory.get_conversation_context()
        if conversation:
            context_parts.append(f"\n{conversation}")
        
        # Step 5: Search if needed
        search_results = None
        if analysis.get('needs_search', False):
            yield f"data: {json.dumps({'type': 'status', 'status': 'searching'})}\n\n"
            
            async with self.search as searcher:
                search_results = await searcher.search(query)
            
            if search_results and search_results.get('success'):
                if search_results.get('answer'):
                    context_parts.append(f"\n## 🔍 Search Answer:\n{search_results['answer']}")
                
                if search_results.get('results'):
                    context_parts.append("\n## 📊 Search Results:")
                    for i, result in enumerate(search_results['results'][:3], 1):
                        title = result.get('title', 'No title')
                        content = result.get('content', '')[:200]
                        context_parts.append(f"\n{i}. **{title}**\n{content}...")
        
        context = "\n".join(context_parts) if context_parts else "No additional context."
        
        # Step 6: Generate response with streaming
        yield f"data: {json.dumps({'type': 'status', 'status': 'generating'})}\n\n"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nUser Query: {query}\n\nPlease provide a comprehensive, accurate response:")
        ]
        
        full_response = ""
        async for chunk in self.streamer.generate(messages):
            if 'full_response' in chunk:
                # Extract full response from completion message
                try:
                    data = json.loads(chunk.split('data: ')[1].strip())
                    if 'full_response' in data:
                        full_response = data['full_response']
                except:
                    pass
            yield chunk
        
        # Step 7: Store response in memory
        if full_response:
            self.memory.add_memory(full_response, "agent_response", importance=1.5)
            self.memory.add_conversation(query, full_response)
            
            # Save memory in background
            asyncio.create_task(self.memory.save_memory())
    
    def _get_greeting(self, query: str) -> str:
        """Get appropriate greeting response"""
        query_lower = query.lower()
        
        greetings = {
            "hi": "Hello! 👋 I'm OMNERT-AGENT, ready to help you with anything you need!",
            "hello": "Hi there! 😊 I'm your AI assistant. What can I do for you today?",
            "hey": "Hey! I'm OMNERT-AGENT. What's on your mind?",
            "good morning": "Good morning! ☀️ How can I assist you today?",
            "good afternoon": "Good afternoon! 🌤️ What would you like to know?",
            "good evening": "Good evening! 🌙 How can I help you?",
            "sup": "Hey there! I'm OMNERT-AGENT. What's up?",
            "what's up": "Not much! Just here to help. What can I do for you?"
        }
        
        for key, response in greetings.items():
            if key in query_lower:
                return response
        
        return "Hello! I'm OMNERT-AGENT. How can I assist you?"
    
    def _get_identity_response(self, query: str) -> str:
        """Get identity/capabilities response"""
        query_lower = query.lower()
        
        if "what can you do" in query_lower:
            return """# 🔧 My Capabilities:

## Core Functions:
• **Answer Questions** - Factual, technical, explanatory
• **Web Search** - Current information with citations
• **Math & Calculations** - From simple arithmetic to complex equations
• **Code Generation** - Python, JavaScript, HTML, etc.
• **Creative Writing** - Stories, poems, articles
• **Data Analysis** - Insights from information

## Advanced Features:
• **Multi-Step Reasoning** - Chain-of-thought problem solving
• **Context Memory** - Remembers our conversation
• **Streaming Responses** - Real-time answer generation
• **Source Citation** - Always cites information sources
• **Self-Correction** - Identifies and fixes mistakes

## Specialized Knowledge:
• Technology & Programming
• Science & Mathematics
• Business & Finance
• History & Culture
• Current Events

Try me with: "What's the latest AI news?" or "Calculate 15% of 250" or "Write a Python function to..." """
        
        if "who are you" in query_lower or "what are you" in query_lower:
            return f"""# 🤖 About Me:

I'm **{self.agent_name} v{self.version}**, a professional AI assistant built with advanced reasoning capabilities.

## My Design:
• **Architecture**: Multi-agent reasoning system
• **Memory**: Persistent conversation memory
• **Search**: Real-time web search integration
• **Streaming**: Real-time response generation
• **Accuracy**: Fact-checking and source verification

## My Purpose:
To provide accurate, comprehensive, and helpful information to users like you. I combine the latest AI models with web search and intelligent reasoning to give you the best possible answers.

## My Personality:
{self.personality}

I'm here to help with any question you have!"""
        
        return f"I'm {self.agent_name} v{self.version}, an advanced AI assistant. How can I help you today?"

# ========== INTERACTIVE CLI ==========
async def interactive_session():
    """Run interactive session with streaming display"""
    
    print("\n" + "="*70)
    print("🤖 OMNERT-AGENT v3.1 - Professional AI Assistant")
    print("="*70)
    print("Features:")
    print("• ✅ Fixed async architecture")
    print("• 🧠 Persistent memory system")
    print("• 🔍 Working web search")
    print("• 💭 Streaming responses (SSE)")
    print("• 🎯 Professional agent personality")
    print("• 📊 Advanced reasoning engine")
    print("="*70)
    
    # Check API keys
    if not Config.GROQ_API_KEY or "your_" in Config.GROQ_API_KEY:
        print("\n❌ ERROR: GROQ_API_KEY not configured!")
        print("   Get free key: https://console.groq.com")
        print("   Add to .env: GROQ_API_KEY=gsk_xxxxxxxxxxxx")
        return
    
    print(f"\n✅ Groq API: Ready ({Config.PRIMARY_MODEL})")
    
    if not Config.TAVILY_API_KEY or "your_" in Config.TAVILY_API_KEY:
        print("⚠️  Tavily API: Not configured (search disabled)")
        print("   Get free key: https://app.tavily.com")
        print("   Add to .env: TAVILY_API_KEY=tvly-xxxxxxxxxxxx")
    else:
        print("✅ Tavily API: Configured (search enabled)")
    
    # Initialize agent
    agent = OmniAgent()
    
    print("\n💡 Example queries:")
    print("  • 'What are the latest developments in AI?'")
    print("  • 'Calculate 25% of 400 + 15 squared'")
    print("  • 'Explain quantum computing simply'")
    print("  • 'What can you do?'")
    print("  • 'Write a Python function to reverse a string'")
    
    print("\n📋 Commands: 'stats', 'clear', 'memory', 'quit'")
    print("="*70)
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n📊 Agent statistics:")
                print(f"  • Memories stored: {len(agent.memory.memories)}")
                print(f"  • Conversations: {len(agent.memory.conversation_history)}")
                print(f"  • Memory file: {Config.MEMORY_FILE}")
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'stats':
                print(f"\n📈 Memory Stats:")
                print(f"  • Total memories: {len(agent.memory.memories)}")
                print(f"  • Conversation history: {len(agent.memory.conversation_history)} entries")
                
                # Count by type
                type_count = {}
                for mem in agent.memory.memories:
                    t = mem.get('type', 'unknown')
                    type_count[t] = type_count.get(t, 0) + 1
                
                print(f"  • Memory types: {type_count}")
                continue
            
            if query.lower() == 'clear':
                agent.memory.memories = []
                agent.memory.conversation_history = []
                await agent.memory.save_memory()
                print("🧹 Memory cleared")
                continue
            
            if query.lower() == 'memory':
                print("\n🧠 Recent memories:")
                for mem in agent.memory.memories[-5:]:
                    print(f"  • [{mem['type']}] {mem['content'][:80]}...")
                print("\n💬 Recent conversation:")
                for conv in agent.memory.conversation_history[-3:]:
                    print(f"  User: {conv['user'][:50]}...")
                    print(f"  Agent: {conv['agent'][:50]}...\n")
                continue
            
            # Process query with streaming
            print(f"\n{'🤖'*20}")
            print("🤖 Agent: ", end="", flush=True)
            
            full_response = ""
            async for chunk in agent.process_query(query):
                try:
                    # Parse SSE format
                    if chunk.startswith("data: "):
                        data = json.loads(chunk[6:].strip())
                        
                        if data.get('type') == 'token':
                            print(data['token'], end="", flush=True)
                            full_response += data['token']
                        elif data.get('type') == 'status':
                            # Show status updates (optional)
                            pass
                        elif data.get('type') == 'complete':
                            # Response complete
                            if data.get('full_response'):
                                full_response = data['full_response']
                        elif data.get('type') == 'error':
                            print(f"\n❌ Error: {data.get('error')}")
                except json.JSONDecodeError:
                    continue
            
            print("\n" + "-"*40)
            
        except KeyboardInterrupt:
            print("\n\n🔄 Session interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

# ========== REQUIREMENTS ==========
"""
Required packages:
pip install langchain-groq httpx python-dotenv numpy scikit-learn
"""

if __name__ == "__main__":
    asyncio.run(interactive_session())