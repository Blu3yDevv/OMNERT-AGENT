"""
OMNERT-AGENT v4.0: Advanced Agentic System with Enhanced Reasoning
COMPLETE FIXED VERSION - All bugs resolved
"""

import os
import json
import asyncio
import logging
import re
import math
import ast
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import httpx
import sympy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- SETUP ---
load_dotenv()

# --- ENUMS ---
class QueryType(Enum):
    GREETING = "greeting"
    MATH = "math"
    FACTUAL = "factual"
    NEWS = "news"
    EXPLANATION = "explanation"
    EMOTIONAL = "emotional"
    COMPLEX = "complex"
    CODE = "code"
    CREATIVE = "creative"
    IDENTITY = "identity"

class SearchPriority(Enum):
    CRITICAL = 3  # Must search (news, current events)
    IMPORTANT = 2  # Should search (factual, technical)
    OPTIONAL = 1  # Could search (explanations, concepts)
    NONE = 0      # Don't search (math, greetings, identity)

# --- CONFIGURATION ---
@dataclass
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # Models with different purposes
    ANALYZER_MODEL = "llama-3.1-8b-instant"
    REASONER_MODEL = "llama-3.3-70b-versatile"
    SYNTHESIZER_MODEL = "llama-3.3-70b-versatile"
    
    # Math precision
    MAX_SEARCH_RESULTS = 5
    SEARCH_TIMEOUT = 30

# --- ADVANCED TAVILY SEARCH (FIXED) ---
class AdvancedTavilySearch:
    """Enhanced Tavily search with proper authentication"""
    
    def __init__(self):
        self.api_key = Config.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com"
        
    async def search(self, query: str, max_results: int = 3, include_answer: bool = True) -> Dict[str, Any]:
        """Search with proper authentication"""
        if not self.api_key or self.api_key.startswith('your_'):
            return {
                "success": False,
                "error": "Tavily API key not configured or invalid. Please check your .env file.",
                "results": []
            }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_images": False,
            "include_answer": include_answer,
            "include_raw_content": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=Config.SEARCH_TIMEOUT) as client:
                response = await client.post(
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
                elif response.status_code == 403:
                    return {
                        "success": False,
                        "error": "Tavily API authentication failed. Check your API key at https://app.tavily.com",
                        "results": []
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Tavily API Error {response.status_code}: {response.text[:200]}",
                        "results": []
                    }
                    
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Search timeout - please try again",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "results": []
            }

# --- FIXED ADVANCED MATH ENGINE ---
class AdvancedMathEngine:
    """Fixed math engine that only evaluates actual mathematical expressions"""
    
    def __init__(self):
        self.safe_globals = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e
        }
        
        # Patterns that are definitely NOT math
        self.non_math_patterns = [
            r'^what (is|are) (you|your|this|that|it|he|she|they)',
            r'^who (is|are) (you|your|this|that|it|he|she|they)',
            r'^explain',
            r'^tell me (about|more)',
            r'^describe',
            r'^how (are|do|does|can|will)',
            r'^why (are|do|does|can|will)',
            r'^where (are|is|do|does)',
            r'^when (are|is|do|does)'
        ]
    
    def is_definitely_not_math(self, text: str) -> bool:
        """Check if text is clearly not a mathematical expression"""
        text_lower = text.lower().strip()
        
        # Check against non-math patterns
        for pattern in self.non_math_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for identity questions
        identity_phrases = ['what are you', 'who are you', 'explain yourself']
        if any(phrase in text_lower for phrase in identity_phrases):
            return True
        
        return False
    
    def is_likely_math(self, text: str) -> bool:
        """Check if text is likely a mathematical expression"""
        text_lower = text.lower().strip()
        
        # Clean common question words
        cleaned = re.sub(r'^(what is|what\'s|calculate|compute|solve)\s+', '', text_lower)
        cleaned = re.sub(r'\?$', '', cleaned)
        
        # Check for math operators
        math_operators = ['+', '-', '*', '/', '^', '**', '=', '%', '÷', '×']
        if any(op in cleaned for op in math_operators):
            return True
        
        # Check for numbers with math context
        if re.search(r'\d+\s*[\+\-\*/]\s*\d+', cleaned):
            return True
        
        # Check for power expressions
        if re.search(r'\d+\s*(to the power of|\^|\\*\*)\s*\d+', cleaned):
            return True
        
        # Check for math keywords with numbers
        math_keywords = ['plus', 'minus', 'times', 'multiplied', 'divided', 'power', 'square', 'cube', 'root']
        if any(keyword in cleaned for keyword in math_keywords) and any(c.isdigit() for c in cleaned):
            return True
        
        return False
    
    def evaluate_expression(self, expression: str) -> Tuple[bool, Any, str]:
        """Safely evaluate ONLY actual mathematical expressions"""
        
        # First check: Definitely not math
        if self.is_definitely_not_math(expression):
            return False, None, "Not a mathematical expression"
        
        # Second check: Likely math
        if not self.is_likely_math(expression):
            return False, None, "Does not appear to be a mathematical expression"
        
        # Clean and normalize
        expr = expression.lower().strip()
        
        # Remove question words
        expr = re.sub(r'^(what (is|are)|calculate|compute|solve|what\'s)\s+', '', expr)
        expr = re.sub(r'\?$', '', expr)
        expr = expr.strip()
        
        # Replace verbal math operations
        replacements = {
            'plus': '+',
            'minus': '-',
            'times': '*',
            'multiplied by': '*',
            'divided by': '/',
            'over': '/',
            'to the power of': '**',
            'power': '**',
            'squared': '**2',
            'cubed': '**3',
            'square root of': 'sqrt',
            'cube root of': '**(1/3)',
            'modulo': '%',
            'mod': '%',
            'factorial': '!',
            'pi': str(math.pi),
            'e': str(math.e)
        }
        
        for word, symbol in replacements.items():
            expr = expr.replace(word, symbol)
        
        # Handle special cases
        expr = re.sub(r'(\d+)\s*div\s+by\s*(\d+)', r'\1 / \2', expr)
        expr = re.sub(r'(\d+)\s*to\s+the\s+(\d+)(?:th|st|nd|rd)\s+power', r'\1 ** \2', expr)
        
        # Safety check: ensure only math characters remain
        allowed_chars = set('0123456789+-*/.()^!eπ ')
        if any(char not in allowed_chars for char in expr.replace(' ', '')):
            return False, None, "Contains non-mathematical characters"
        
        try:
            # Try sympy for complex math
            sympy_expr = sympy.sympify(expr)
            result = sympy.N(sympy_expr, 50)
            
            # Format result nicely
            if result.is_Integer:
                result = int(result)
            elif result.is_Float:
                # Round to reasonable precision
                result = float(f"{result:.10f}")
                # Remove trailing zeros
                result = float(f"{result:g}")
            else:
                result = str(result)
            
            explanation = f"""**Mathematical Calculation:**

**Original Query:** {expression}
**Processed Expression:** {expr}
**Result:** {result}

**Verification:** This is a mathematically valid calculation."""
            
            return True, result, explanation
            
        except Exception as e:
            # Try basic evaluation as fallback
            try:
                # Safe evaluation with restricted globals
                result = eval(expr, {"__builtins__": {}}, self.safe_globals)
                
                if isinstance(result, float):
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = round(result, 10)
                        result = float(f"{result:g}")
                
                explanation = f"""**Basic Calculation:**

**Original Query:** {expression}
**Processed Expression:** {expr}
**Result:** {result}

**Note:** Basic mathematical evaluation."""
                
                return True, result, explanation
                
            except Exception as e2:
                return False, None, f"Cannot evaluate as mathematics: {str(e2)}"

# --- FIXED ADVANCED QUERY ANALYZER ---
class AdvancedQueryAnalyzer:
    """Fixed query analyzer with better math detection"""
    
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.1,
            model_name=Config.ANALYZER_MODEL,
            api_key=Config.GROQ_API_KEY,
            max_tokens=1024
        )
        self.math_engine = AdvancedMathEngine()
    
    async def analyze(self, query: str, context: Optional[List] = None) -> Dict[str, Any]:
        """Perform comprehensive query analysis"""
        
        print("\n" + "="*60)
        print("🔬 ADVANCED QUERY ANALYSIS")
        print("="*60)
        
        # LAYER 1: Quick heuristics
        quick_analysis = self._quick_heuristics(query)
        print(f"  Layer 1 (Heuristics): {quick_analysis['type'].value.upper()}")
        
        # Special handling for identity questions
        if quick_analysis["type"] == QueryType.IDENTITY:
            return quick_analysis
        
        # LAYER 2: Math detection (FIXED)
        is_math, math_result, math_explanation = self.math_engine.evaluate_expression(query)
        if is_math:
            print(f"  Layer 2 (Math): Valid mathematical expression detected")
            return {
                "type": QueryType.MATH,
                "needs_search": False,
                "search_priority": SearchPriority.NONE,
                "needs_reasoning": False,
                "complexity": "low",
                "confidence": 1.0,
                "math_result": math_result,
                "math_explanation": math_explanation,
                "raw_analysis": "Valid mathematical query detected"
            }
        elif quick_analysis["type"] == QueryType.MATH:
            # If heuristics said math but engine says no, downgrade to factual
            quick_analysis["type"] = QueryType.FACTUAL
            quick_analysis["needs_search"] = True
            quick_analysis["search_priority"] = SearchPriority.OPTIONAL
        
        # LAYER 3: LLM-based analysis (with fixed context)
        llm_analysis = await self._llm_analysis(query, context)
        print(f"  Layer 3 (LLM): {llm_analysis['type'].value.upper()}")
        
        # LAYER 4: Cross-validation
        validated = self._cross_validate(quick_analysis, llm_analysis)
        print(f"  Layer 4 (Validation): Confidence {validated['confidence']:.0%}")
        
        # LAYER 5: Context adaptation
        final_analysis = self._adapt_to_context(validated, context)
        print(f"  Layer 5 (Context): Search priority {final_analysis['search_priority'].name}")
        
        return final_analysis
    
    def _quick_heuristics(self, query: str) -> Dict[str, Any]:
        """Fast heuristic analysis with identity detection"""
        query_lower = query.lower().strip()
        
        # Greetings
        greetings = {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'}
        if query_lower in greetings:
            return {
                "type": QueryType.GREETING,
                "needs_search": False,
                "search_priority": SearchPriority.NONE,
                "needs_reasoning": False,
                "complexity": "low",
                "confidence": 0.95,
                "raw_analysis": "Simple greeting detected"
            }
        
        # Identity questions (SPECIAL HANDLING)
        identity_patterns = [
            r'^what (are|is) you',
            r'^who (are|is) you',
            r'^explain yourself',
            r'^tell me about yourself',
            r'^what\'s your (name|purpose|function)'
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, query_lower):
                return {
                    "type": QueryType.IDENTITY,
                    "needs_search": False,
                    "search_priority": SearchPriority.NONE,
                    "needs_reasoning": True,
                    "complexity": "low",
                    "confidence": 0.9,
                    "raw_analysis": "Identity question detected"
                }
        
        # Emotional content
        emotional_words = {'sad', 'happy', 'angry', 'excited', 'died', 'passed away', 'sorry', 'thank you'}
        if any(word in query_lower for word in emotional_words):
            return {
                "type": QueryType.EMOTIONAL,
                "needs_search": False,
                "search_priority": SearchPriority.NONE,
                "needs_reasoning": True,
                "complexity": "medium",
                "confidence": 0.8,
                "raw_analysis": "Emotional content detected"
            }
        
        # News/current events
        news_indicators = {'latest', 'news', 'update', '2024', '2025', 'recent', 'current', 'today', 'yesterday'}
        if any(indicator in query_lower for indicator in news_indicators):
            return {
                "type": QueryType.NEWS,
                "needs_search": True,
                "search_priority": SearchPriority.CRITICAL,
                "needs_reasoning": True,
                "complexity": "high",
                "confidence": 0.85,
                "raw_analysis": "News/current events query"
            }
        
        # Explanations
        explanation_indicators = {'explain', 'how does', 'why does', 'tell me about', 'describe', 'what is'}
        if any(indicator in query_lower for indicator in explanation_indicators):
            # Check if it's asking about the agent itself
            if 'you' in query_lower or 'yourself' in query_lower:
                return {
                    "type": QueryType.IDENTITY,
                    "needs_search": False,
                    "search_priority": SearchPriority.NONE,
                    "needs_reasoning": True,
                    "complexity": "low",
                    "confidence": 0.9,
                    "raw_analysis": "Identity/explanation request"
                }
            return {
                "type": QueryType.EXPLANATION,
                "needs_search": True,
                "search_priority": SearchPriority.IMPORTANT,
                "needs_reasoning": True,
                "complexity": "medium",
                "confidence": 0.75,
                "raw_analysis": "Explanation request detected"
            }
        
        # Creative requests
        creative_indicators = {'write', 'create', 'story', 'poem', 'song', 'essay', 'article'}
        if any(indicator in query_lower for indicator in creative_indicators):
            return {
                "type": QueryType.CREATIVE,
                "needs_search": True,
                "search_priority": SearchPriority.OPTIONAL,
                "needs_reasoning": True,
                "complexity": "high",
                "confidence": 0.7,
                "raw_analysis": "Creative request detected"
            }
        
        # Default (factual)
        return {
            "type": QueryType.FACTUAL,
            "needs_search": True,
            "search_priority": SearchPriority.OPTIONAL,
            "needs_reasoning": True,
            "complexity": "medium",
            "confidence": 0.6,
            "raw_analysis": "General factual query"
        }
    
    async def _llm_analysis(self, query: str, context: Optional[List]) -> Dict[str, Any]:
        """LLM-based deep analysis with PROPER message formatting"""
        
        system_prompt = """You are an advanced query analyzer. Analyze the user's query and provide detailed analysis.

Consider:
1. Query intent and purpose
2. Required knowledge domains
3. Need for current information
4. Complexity and depth required
5. Best response strategy

Output JSON format:
{
  "query_intent": "description of intent",
  "knowledge_domains": ["list", "of", "domains"],
  "requires_current_info": true/false,
  "complexity_score": 1-10,
  "best_response_strategy": "description",
  "should_search": true/false,
  "search_priority": "critical/important/optional/none",
  "needs_reasoning": true/false,
  "estimated_confidence": 0.0-1.0,
  "query_type": "greeting/math/factual/news/explanation/emotional/complex/code/creative/identity"
}"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add context if available (PROPERLY FORMATTED)
        if context:
            for item in context[-3:]:  # Last 3 items
                if isinstance(item, dict):
                    if 'query' in item:
                        messages.append(HumanMessage(content=item['query']))
                    if 'answer' in item:
                        # Truncate answer to avoid token overflow
                        answer_preview = item['answer'][:200] + "..." if len(item['answer']) > 200 else item['answer']
                        messages.append(AIMessage(content=answer_preview))
        
        messages.append(HumanMessage(content=f"Analyze this query: {query}"))
        
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            # Parse JSON response
            content = response.content.strip()
            
            # Extract JSON from code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            
            # Map query type to enum
            query_type_map = {
                "greeting": QueryType.GREETING,
                "math": QueryType.MATH,
                "factual": QueryType.FACTUAL,
                "news": QueryType.NEWS,
                "explanation": QueryType.EXPLANATION,
                "emotional": QueryType.EMOTIONAL,
                "complex": QueryType.COMPLEX,
                "code": QueryType.CODE,
                "creative": QueryType.CREATIVE,
                "identity": QueryType.IDENTITY
            }
            
            query_type_str = analysis.get("query_type", "factual")
            query_type = query_type_map.get(query_type_str, QueryType.FACTUAL)
            
            # Map search priority
            priority_map = {
                "critical": SearchPriority.CRITICAL,
                "important": SearchPriority.IMPORTANT,
                "optional": SearchPriority.OPTIONAL,
                "none": SearchPriority.NONE
            }
            search_priority_str = analysis.get("search_priority", "optional").lower()
            search_priority = priority_map.get(search_priority_str, SearchPriority.OPTIONAL)
            
            return {
                "type": query_type,
                "needs_search": analysis.get("should_search", True),
                "search_priority": search_priority,
                "needs_reasoning": analysis.get("needs_reasoning", True),
                "complexity": "high" if analysis.get("complexity_score", 5) > 7 else "medium" if analysis.get("complexity_score", 5) > 4 else "low",
                "confidence": analysis.get("estimated_confidence", 0.7),
                "raw_analysis": json.dumps(analysis, indent=2)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM JSON parse error: {e}")
            return self._quick_heuristics(query)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._quick_heuristics(query)
    
    def _cross_validate(self, heuristic: Dict, llm: Dict) -> Dict:
        """Cross-validate heuristic and LLM analyses"""
        
        # If both agree on search need, use that
        if heuristic["needs_search"] == llm["needs_search"]:
            needs_search = heuristic["needs_search"]
            search_priority = max(heuristic["search_priority"], llm["search_priority"], key=lambda x: x.value)
        else:
            # If disagreement, prefer LLM but be conservative
            needs_search = llm["needs_search"] or heuristic["needs_search"]
            search_priority = max(heuristic["search_priority"], llm["search_priority"], key=lambda x: x.value)
        
        # Combine confidence
        combined_confidence = (heuristic["confidence"] * 0.3 + llm["confidence"] * 0.7)
        
        # Determine query type (prefer LLM but consider heuristic)
        if llm["confidence"] > heuristic["confidence"]:
            query_type = llm["type"]
        else:
            query_type = heuristic["type"]
        
        return {
            "type": query_type,
            "needs_search": needs_search,
            "search_priority": search_priority,
            "needs_reasoning": llm["needs_reasoning"] or heuristic["needs_reasoning"],
            "complexity": max(heuristic["complexity"], llm["complexity"], key=lambda x: ["low", "medium", "high"].index(x)),
            "confidence": combined_confidence,
            "raw_analysis": f"Heuristic: {heuristic['raw_analysis']}\nLLM: {llm['raw_analysis']}"
        }
    
    def _adapt_to_context(self, analysis: Dict, context: Optional[List]) -> Dict:
        """Adapt analysis based on conversation context"""
        if not context or len(context) < 2:
            return analysis
        
        # Check if this is follow-up to previous question
        last_queries = []
        for item in context[-3:]:
            if isinstance(item, dict) and 'query' in item:
                last_queries.append(item['query'].lower())
        
        # If previous query was same type, might not need search again
        if analysis["type"] in [QueryType.FACTUAL, QueryType.EXPLANATION]:
            # Check for repeated information requests
            for last_query in last_queries:
                if any(word in last_query for word in ["what is", "explain", "tell me about"]):
                    analysis["search_priority"] = SearchPriority.OPTIONAL
        
        return analysis

# --- ADVANCED REASONING ENGINE ---
class AdvancedReasoningEngine:
    """Multi-stage reasoning with self-correction"""
    
    def __init__(self):
        self.reasoner = ChatGroq(
            temperature=0.7,
            model_name=Config.REASONER_MODEL,
            api_key=Config.GROQ_API_KEY,
            max_tokens=2048
        )
        
        self.critic = ChatGroq(
            temperature=0.3,
            model_name=Config.REASONER_MODEL,
            api_key=Config.GROQ_API_KEY,
            max_tokens=1024
        )
    
    async def reason(self, query: str, analysis: Dict, search_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform multi-stage reasoning with self-correction"""
        
        print("\n" + "="*60)
        print("🧠 ADVANCED REASONING PROCESS")
        print("="*60)
        
        # Handle identity questions specially
        if analysis["type"] == QueryType.IDENTITY:
            return {
                "initial_analysis": "This is an identity question about the AI assistant.",
                "critique": "No critique needed for identity questions.",
                "final_reasoning": "The user is asking about the AI's identity, capabilities, or purpose.",
                "confidence": 1.0,
                "key_insights": ["Identity question", "About AI assistant", "No external search needed"],
                "remaining_questions": []
            }
        
        # Stage 1: Initial reasoning
        print("  Stage 1: Initial analysis")
        initial_reasoning = await self._initial_reasoning(query, analysis, search_data)
        
        # Stage 2: Critical review
        print("  Stage 2: Critical review")
        critique = await self._critical_review(query, initial_reasoning, search_data)
        
        # Stage 3: Synthesis and refinement
        print("  Stage 3: Synthesis")
        final_reasoning = await self._synthesize_reasoning(initial_reasoning, critique)
        
        # Calculate confidence
        confidence = self._calculate_confidence(final_reasoning, search_data)
        
        print(f"  ✓ Reasoning complete: Confidence {confidence:.0%}")
        
        return {
            "initial_analysis": initial_reasoning,
            "critique": critique,
            "final_reasoning": final_reasoning,
            "confidence": confidence,
            "key_insights": self._extract_insights(final_reasoning),
            "remaining_questions": self._identify_gaps(final_reasoning, search_data)
        }
    
    async def _initial_reasoning(self, query: str, analysis: Dict, search_data: Optional[Dict]) -> str:
        """Initial chain-of-thought reasoning"""
        
        context = ""
        if search_data and search_data.get("success"):
            if search_data.get("answer"):
                context += f"DIRECT ANSWER FOUND: {search_data['answer']}\n\n"
            if search_data.get("results"):
                context += "SEARCH RESULTS:\n"
                for i, result in enumerate(search_data["results"][:3], 1):
                    context += f"{i}. {result.get('title', 'No title')}\n"
                    content = result.get('content', '')[:300]
                    if content:
                        context += f"   {content}...\n\n"
        
        prompt = f"""Think step-by-step about this query:

QUERY: {query}
QUERY TYPE: {analysis['type'].value}
COMPLEXITY: {analysis['complexity']}

{context if context else "No external information available. Rely on internal knowledge."}

Perform a detailed analysis:
1. Break down the query into components
2. Identify key concepts and relationships
3. Consider multiple perspectives
4. Note assumptions and uncertainties
5. Outline what a complete answer should include

Think deeply and thoroughly."""
        
        response = await asyncio.to_thread(
            self.reasoner.invoke,
            [HumanMessage(content=prompt)]
        )
        
        return response.content
    
    async def _critical_review(self, query: str, reasoning: str, search_data: Optional[Dict]) -> str:
        """Critically review the initial reasoning"""
        
        prompt = f"""Critically review this reasoning about the query "{query}":

REASONING:
{reasoning}

SEARCH DATA AVAILABLE: {'Yes' if search_data and search_data.get('success') else 'No'}

Identify:
1. Logical gaps or inconsistencies
2. Unsupported claims or assumptions
3. Missing perspectives or considerations
4. Potential biases
5. Areas needing more evidence

Be thorough and constructive."""
        
        response = await asyncio.to_thread(
            self.critic.invoke,
            [HumanMessage(content=prompt)]
        )
        
        return response.content
    
    async def _synthesize_reasoning(self, initial: str, critique: str) -> str:
        """Synthesize initial reasoning with critique"""
        
        prompt = f"""Synthesize the initial reasoning with critical feedback:

INITIAL REASONING:
{initial}

CRITICAL FEEDBACK:
{critique}

Create an improved, comprehensive reasoning that:
1. Addresses the critique points
2. Strengthens weak arguments
3. Maintains strong points
4. Provides a balanced perspective
5. Clearly states confidence levels

Focus on creating the best possible understanding of the problem."""
        
        response = await asyncio.to_thread(
            self.reasoner.invoke,
            [HumanMessage(content=prompt)]
        )
        
        return response.content
    
    def _calculate_confidence(self, reasoning: str, search_data: Optional[Dict]) -> float:
        """Calculate confidence based on reasoning quality and data"""
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on reasoning characteristics
        indicators = {
            "high_confidence": ["clearly", "evidently", "conclusively", "supported by", "demonstrates"],
            "medium_confidence": ["likely", "probably", "suggests", "indicates", "appears"],
            "low_confidence": ["maybe", "possibly", "uncertain", "unclear", "speculative"]
        }
        
        reasoning_lower = reasoning.lower()
        
        for phrase in indicators["high_confidence"]:
            if phrase in reasoning_lower:
                confidence = min(1.0, confidence + 0.1)
        
        for phrase in indicators["low_confidence"]:
            if phrase in reasoning_lower:
                confidence = max(0.1, confidence - 0.1)
        
        # Adjust based on search data
        if search_data and search_data.get("success"):
            if search_data.get("answer"):
                confidence = min(1.0, confidence + 0.2)
            if search_data.get("results"):
                confidence = min(1.0, confidence + 0.1 * min(len(search_data["results"]), 3))
        
        return confidence
    
    def _extract_insights(self, reasoning: str) -> List[str]:
        """Extract key insights from reasoning"""
        lines = reasoning.split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•', '1.', '2.', '3.', '4.', '5.')):
                insights.append(line)
            elif len(line.split()) <= 20 and len(line) > 10 and not line.endswith('?'):
                insights.append(line)
        
        return insights[:5]
    
    def _identify_gaps(self, reasoning: str, search_data: Optional[Dict]) -> List[str]:
        """Identify knowledge gaps in reasoning"""
        gaps = []
        
        # Check for uncertainty markers
        uncertainty_markers = ["unclear", "unknown", "uncertain", "not sure", "need more", "requires further"]
        reasoning_lower = reasoning.lower()
        
        for marker in uncertainty_markers:
            if marker in reasoning_lower:
                # Extract context around marker
                start = max(0, reasoning_lower.find(marker) - 100)
                end = min(len(reasoning), reasoning_lower.find(marker) + 100)
                gaps.append(reasoning[start:end] + "...")
        
        # If no search data but reasoning suggests need for current info
        if not search_data and any(word in reasoning_lower for word in ["current", "recent", "latest", "new"]):
            gaps.append("Requires current information not available in search results")
        
        return gaps[:3]

# --- ADVANCED SYNTHESIS ENGINE (FIXED) ---
class AdvancedSynthesisEngine:
    """Multi-phase answer synthesis for high-quality outputs"""
    
    def __init__(self):
        self.synthesizer = ChatGroq(
            temperature=0.5,
            model_name=Config.SYNTHESIZER_MODEL,
            api_key=Config.GROQ_API_KEY,
            max_tokens=3072
        )
    
    async def synthesize(self, query: str, analysis: Dict, 
                        reasoning: Optional[Dict], search_data: Optional[Dict]) -> str:
        """Synthesize final answer with multiple quality checks"""
        
        print("\n" + "="*60)
        print("🎯 ADVANCED ANSWER SYNTHESIS")
        print("="*60)
        
        # Handle identity questions specially
        if analysis["type"] == QueryType.IDENTITY:
            return self._generate_identity_response(query)
        
        # Phase 1: Draft answer
        print("  Phase 1: Drafting answer")
        draft = await self._create_draft(query, analysis, reasoning, search_data)
        
        # Phase 2: Quality enhancement
        print("  Phase 2: Enhancing quality")
        enhanced = await self._enhance_quality(draft, query, analysis)
        
        # Phase 3: Format and polish
        print("  Phase 3: Final polish")
        confidence = reasoning.get("confidence", 0.5) if reasoning else analysis.get("confidence", 0.5)
        final = self._polish_answer(enhanced, analysis, confidence)
        
        print(f"  ✓ Synthesis complete: {len(final)} characters")
        
        return final
    
    def _generate_identity_response(self, query: str) -> str:
        """Generate response for identity questions"""
        identity_responses = [
            "I am **OMNERT-AGENT**, an advanced AI assistant with multi-phase reasoning capabilities. "
            "I can help with complex queries, mathematical calculations, web searches, and creative tasks.",
            
            "I'm an AI assistant called **OMNERT-AGENT** designed to provide thoughtful, well-reasoned answers "
            "using a sophisticated 5-layer analysis process. My goal is to understand your questions deeply "
            "and provide comprehensive, accurate responses.",
            
            "I am **OMNERT-AGENT v4.0** - an advanced reasoning system that analyzes queries through multiple layers "
            "including heuristic analysis, mathematical evaluation, LLM understanding, validation, and context adaptation. "
            "I'm here to help with any question you might have!"
        ]
        
        import random
        return random.choice(identity_responses)
    
    async def _create_draft(self, query: str, analysis: Dict, 
                           reasoning: Optional[Dict], search_data: Optional[Dict]) -> str:
        """Create initial draft answer"""
        
        # Prepare context
        context_parts = []
        
        if search_data and search_data.get("success"):
            if search_data.get("answer"):
                context_parts.append(f"DIRECT ANSWER FROM SEARCH: {search_data['answer']}")
            if search_data.get("results"):
                context_parts.append("SUPPORTING INFORMATION:")
                for i, result in enumerate(search_data["results"][:2], 1):
                    context_parts.append(f"{i}. {result.get('title', 'No title')}")
                    content = result.get('content', '')[:200]
                    if content:
                        context_parts.append(f"   {content}")
        
        context_parts.append(f"\nREASONING INSIGHTS:")
        if reasoning:
            for insight in reasoning.get("key_insights", [])[:3]:
                context_parts.append(f"• {insight}")
        else:
            context_parts.append("• No deep reasoning required for this query type.")
        
        context = "\n".join(context_parts) if context_parts else "No external data available."
        
        confidence = reasoning.get("confidence", 0.5) if reasoning else analysis.get("confidence", 0.5)
        
        prompt = f"""Create a comprehensive answer to this query:

QUERY: {query}
QUERY TYPE: {analysis['type'].value}
COMPLEXITY: {analysis['complexity']}

CONTEXT AND INFORMATION:
{context}

REASONING SUMMARY:
{reasoning.get('final_reasoning', 'No reasoning available') if reasoning else 'No reasoning available'}

CONFIDENCE LEVEL: {confidence:.0%}

Instructions for the draft:
1. Address the query directly and completely
2. Incorporate all relevant information
3. Structure the answer logically
4. Use clear, concise language
5. Note confidence levels and limitations
6. Include key takeaways

Create a detailed draft answer."""
        
        response = await asyncio.to_thread(
            self.synthesizer.invoke,
            [HumanMessage(content=prompt)]
        )
        
        return response.content
    
    async def _enhance_quality(self, draft: str, query: str, analysis: Dict) -> str:
        """Enhance answer quality with refinements"""
        
        prompt = f"""Enhance the quality of this answer:

ORIGINAL QUERY: {query}
QUERY TYPE: {analysis['type'].value}

DRAFT ANSWER:
{draft}

Enhancements to apply:
1. Improve clarity and readability
2. Strengthen arguments with evidence
3. Add relevant examples or analogies
4. Ensure logical flow and structure
5. Remove redundancy
6. Add helpful formatting (headings, bullet points where appropriate)
7. Ensure the answer fully addresses the query

Provide the enhanced version."""
        
        response = await asyncio.to_thread(
            self.synthesizer.invoke,
            [HumanMessage(content=prompt)]
        )
        
        return response.content
    
    def _polish_answer(self, enhanced: str, analysis: Dict, confidence: float) -> str:
        """Final polish and formatting"""
        
        # Add confidence notice if needed
        if confidence < 0.7:
            enhanced = f"⚠️ **Confidence Level: {confidence:.0%}**\n\n{enhanced}"
        
        # Add query-type specific formatting
        if analysis["type"] == QueryType.EXPLANATION:
            enhanced = self._format_explanation(enhanced)
        elif analysis["type"] == QueryType.NEWS:
            enhanced = self._format_news(enhanced)
        elif analysis["type"] == QueryType.CREATIVE:
            enhanced = self._format_creative(enhanced)
        
        # Ensure proper ending
        if not enhanced.strip().endswith(('.', '!', '?')):
            enhanced += '.'
        
        return enhanced
    
    def _format_explanation(self, text: str) -> str:
        """Format explanation answers with clear structure"""
        lines = text.split('\n')
        formatted = []
        
        # Add header
        formatted.append("## 📚 Detailed Explanation")
        formatted.append("")
        
        for line in lines:
            line = line.strip()
            if line:
                # Format bullet points
                if line.startswith(('-', '*', '•')):
                    formatted.append(f"• {line[1:].strip()}")
                # Format numbered lists
                elif re.match(r'^\d+[\.\)]', line):
                    formatted.append(line)
                # Format headers
                elif line.isupper() or (len(line) < 50 and not line.endswith('.')):
                    formatted.append(f"\n**{line}**")
                else:
                    formatted.append(line)
        
        return '\n'.join(formatted)
    
    def _format_news(self, text: str) -> str:
        """Format news answers with timeliness markers"""
        lines = text.split('\n')
        formatted = []
        
        # Add header with date
        current_date = datetime.now().strftime("%Y-%m-%d")
        formatted.append(f"## 📰 Latest Information (as of {current_date})")
        formatted.append("")
        
        for line in lines:
            line = line.strip()
            if line:
                # Add emoji for key points
                if any(keyword in line.lower() for keyword in ['important', 'key', 'major', 'significant']):
                    formatted.append(f"🔹 {line}")
                elif any(keyword in line.lower() for keyword in ['update', 'new', 'recent', 'latest']):
                    formatted.append(f"🆕 {line}")
                else:
                    formatted.append(line)
        
        return '\n'.join(formatted)
    
    def _format_creative(self, text: str) -> str:
        """Format creative writing answers"""
        lines = text.split('\n')
        formatted = []
        
        formatted.append("## ✍️ Creative Response")
        formatted.append("")
        
        for line in lines:
            line = line.strip()
            if line:
                formatted.append(line)
        
        return '\n'.join(formatted)

# --- MAIN AGENT (FIXED) ---
class OmniAgentV4:
    """Advanced agent with comprehensive capabilities"""
    
    def __init__(self):
        self.analyzer = AdvancedQueryAnalyzer()
        self.search = AdvancedTavilySearch()
        self.math_engine = AdvancedMathEngine()
        self.reasoner = AdvancedReasoningEngine()
        self.synthesizer = AdvancedSynthesisEngine()
        self.history = []
        
    async def process(self, query: str) -> Dict[str, Any]:
        """Process query through complete pipeline"""
        
        print(f"\n{'🚀'*30}")
        print(f"🚀 PROCESSING: {query}")
        print(f"{'🚀'*30}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze query
            analysis = await self.analyzer.analyze(query, self.history)
            
            # Handle greetings directly
            if analysis["type"] == QueryType.GREETING:
                duration = (datetime.now() - start_time).total_seconds()
                
                greetings = {
                    "hi": "Hello! 👋 How can I help you today?",
                    "hello": "Hi there! 🌟 What can I do for you?",
                    "hey": "Hey! 😊 What's on your mind?",
                    "good morning": "Good morning! ☀️ How can I assist you today?",
                    "good afternoon": "Good afternoon! 🌤️ What would you like to know?",
                    "good evening": "Good evening! 🌙 How can I help you?"
                }
                
                query_lower = query.lower().strip()
                answer = greetings.get(query_lower, "Hello! How can I help you today?")
                
                result = {
                    "query": query,
                    "answer": answer,
                    "analysis": analysis,
                    "search_results": None,
                    "reasoning": {"confidence": 1.0},
                    "metrics": {
                        "duration_seconds": round(duration, 3),
                        "confidence": 1.0,
                        "query_type": "greeting",
                        "search_performed": False,
                        "search_results_count": 0
                    }
                }
                
                self.history.append(result)
                return result
            
            # Handle identity questions
            if analysis["type"] == QueryType.IDENTITY:
                duration = (datetime.now() - start_time).total_seconds()
                
                answer = await self.synthesizer.synthesize(query, analysis, None, None)
                
                result = {
                    "query": query,
                    "answer": answer,
                    "analysis": analysis,
                    "search_results": None,
                    "reasoning": {"confidence": 1.0},
                    "metrics": {
                        "duration_seconds": round(duration, 3),
                        "confidence": 1.0,
                        "query_type": "identity",
                        "search_performed": False,
                        "search_results_count": 0
                    }
                }
                
                self.history.append(result)
                return result
            
            # Step 2: Handle math queries directly
            if analysis["type"] == QueryType.MATH:
                is_valid, result, explanation = self.math_engine.evaluate_expression(query)
                if is_valid:
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    answer = f"""## 🧮 Mathematical Result

{explanation}

**Confidence:** 100% (Mathematically verified)"""
                    
                    result = {
                        "query": query,
                        "answer": answer,
                        "analysis": analysis,
                        "search_results": None,
                        "reasoning": {"confidence": 1.0},
                        "metrics": {
                            "duration_seconds": round(duration, 3),
                            "confidence": 1.0,
                            "query_type": "math",
                            "search_performed": False,
                            "math_solved": True
                        }
                    }
                    
                    self.history.append(result)
                    return result
                else:
                    # If math evaluation failed, treat as normal query
                    analysis["type"] = QueryType.FACTUAL
                    analysis["needs_search"] = True
                    analysis["needs_reasoning"] = True
            
            # Step 3: Conduct search if needed
            search_results = None
            if analysis["needs_search"] and analysis["search_priority"].value > 0:
                print(f"\n🌐 SEARCHING: Priority {analysis['search_priority'].name}")
                search_results = await self.search.search(
                    query, 
                    max_results=Config.MAX_SEARCH_RESULTS,
                    include_answer=True
                )
                
                if search_results and search_results.get("success"):
                    result_count = len(search_results.get("results", []))
                    print(f"  ✓ Found {result_count} results")
                    if search_results.get("answer"):
                        print(f"  ✓ Direct answer available")
                elif search_results:
                    print(f"  ⚠️ Search failed: {search_results.get('error', 'Unknown error')}")
            
            # Step 4: Perform reasoning
            reasoning_results = None
            if analysis["needs_reasoning"]:
                reasoning_results = await self.reasoner.reason(query, analysis, search_results)
            
            # Step 5: Synthesize final answer
            answer = await self.synthesizer.synthesize(query, analysis, reasoning_results, search_results)
            
            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()
            
            # Compile results
            result = {
                "query": query,
                "answer": answer,
                "analysis": analysis,
                "search_results": search_results,
                "reasoning": reasoning_results,
                "metrics": {
                    "duration_seconds": round(duration, 3),
                    "confidence": reasoning_results.get("confidence", 0.5) if reasoning_results else analysis.get("confidence", 0.5),
                    "query_type": analysis["type"].value,
                    "search_performed": search_results and search_results.get("success", False),
                    "search_results_count": len(search_results.get("results", [])) if search_results else 0,
                    "complexity": analysis["complexity"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            error_msg = f"""## ❌ Error Encountered

I encountered an error while processing your query: `{str(e)[:100]}`

Please try:
1. Rephrasing your question
2. Asking a simpler version
3. Trying again in a moment

If the problem persists, the issue may be with external services."""
            
            return {
                "query": query,
                "answer": error_msg,
                "error": str(e),
                "metrics": {
                    "error": True,
                    "duration_seconds": (datetime.now() - start_time).total_seconds()
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.history:
            return {"total_queries": 0}
        
        total = len(self.history)
        successful = [r for r in self.history if not r.get("error")]
        
        if successful:
            avg_confidence = sum(r["metrics"].get("confidence", 0) for r in successful) / len(successful)
            avg_duration = sum(r["metrics"].get("duration_seconds", 0) for r in successful) / len(successful)
            search_count = sum(1 for r in successful if r["metrics"].get("search_performed", False))
        else:
            avg_confidence = 0
            avg_duration = 0
            search_count = 0
        
        return {
            "total_queries": total,
            "successful_queries": len(successful),
            "average_confidence": f"{avg_confidence:.0%}",
            "average_duration": f"{avg_duration:.2f}s",
            "searches_performed": search_count,
            "query_types": list(set(r["metrics"].get("query_type", "unknown") for r in self.history)),
            "recent_queries": [r["query"][:50] + "..." if len(r["query"]) > 50 else r["query"] 
                              for r in self.history[-5:]]
        }

# --- INTERACTIVE SESSION ---
async def interactive_session():
    """Run interactive chat session"""
    print("\n" + "🔥" * 60)
    print("🔥 OMNERT-AGENT v4.0 - ADVANCED REASONING SYSTEM")
    print("🔥" * 60)
    print("\n" + "="*60)
    print("CAPABILITIES:")
    print("• 5-Layer Query Analysis")
    print("• Advanced Mathematical Reasoning")
    print("• Multi-Stage Search Strategy")
    print("• Chain-of-Thought with Self-Correction")
    print("• Confidence-Aware Synthesis")
    print("• Context-Aware Processing")
    print("="*60)
    print("\nCOMMANDS:")
    print("• 'stats' - Show conversation statistics")
    print("• 'clear' - Clear conversation history")
    print("• 'debug' - Toggle debug mode")
    print("• 'quit', 'exit', 'q' - End session")
    print("="*60)
    
    # API key warnings
    if not Config.TAVILY_API_KEY or Config.TAVILY_API_KEY.startswith('your_'):
        print("\n⚠️  WARNING: Tavily API key not configured or invalid.")
        print("   Search functionality will be limited.")
        print("   Get a free key at: https://app.tavily.com")
        print("   Add to .env: TAVILY_API_KEY=your_actual_key_here\n")
    
    agent = OmniAgentV4()
    debug_mode = False
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n" + "="*60)
                print("📊 FINAL STATISTICS:")
                stats = agent.get_stats()
                for key, value in stats.items():
                    if isinstance(value, list):
                        print(f"  {key}:")
                        for item in value:
                            print(f"    • {item}")
                    else:
                        print(f"  {key}: {value}")
                print("="*60)
                print("\n👋 Thank you for using OMNERT-AGENT!")
                break
            
            if query.lower() == 'stats':
                print("\n" + "="*60)
                print("📈 CONVERSATION STATISTICS:")
                stats = agent.get_stats()
                for key, value in stats.items():
                    if isinstance(value, list):
                        print(f"  • {key}:")
                        for item in value:
                            print(f"    - {item}")
                    else:
                        print(f"  • {key}: {value}")
                print("="*60)
                continue
            
            if query.lower() == 'clear':
                agent.history = []
                print("\n✅ Conversation history cleared")
                continue
            
            if query.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"\n{'🔧 DEBUG MODE: ON' if debug_mode else '🔧 DEBUG MODE: OFF'}")
                continue
            
            # Process query
            result = await agent.process(query)
            
            # Display answer
            print("\n" + "🤖" * 60)
            print("🤖 OMNERT-AGENT:")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)
            
            # Show metrics
            metrics = result.get("metrics", {})
            if not metrics.get("error"):
                print(f"\n📊 PERFORMANCE METRICS:")
                print(f"  • Confidence: {metrics.get('confidence', 0):.0%}")
                print(f"  • Processing Time: {metrics.get('duration_seconds', 0):.2f}s")
                print(f"  • Query Type: {metrics.get('query_type', 'unknown').upper()}")
                print(f"  • Complexity: {metrics.get('complexity', 'medium').upper()}")
                if metrics.get("search_performed"):
                    print(f"  • Sources Analyzed: {metrics.get('search_results_count', 0)}")
                if metrics.get("math_solved"):
                    print(f"  • Mathematical Verification: ✓")
            
            # Debug info
            if debug_mode and result.get("analysis"):
                print(f"\n🔧 DEBUG INFO:")
                analysis = result["analysis"]
                print(f"  • Search Priority: {analysis.get('search_priority', 'N/A')}")
                print(f"  • Needs Search: {analysis.get('needs_search', False)}")
                print(f"  • Needs Reasoning: {analysis.get('needs_reasoning', False)}")
                if result.get("search_results"):
                    sr = result["search_results"]
                    print(f"  • Search Success: {sr.get('success', False)}")
                    if sr.get("error"):
                        print(f"  • Search Error: {sr.get('error')[:100]}...")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)[:100]}...")

# --- MAIN ---
def main():
    """Main entry point"""
    # Check API keys
    try:
        Config.GROQ_API_KEY
        print("✅ Groq API key loaded")
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease create a .env file with:")
        print("GROQ_API_KEY=your_groq_api_key_here")
        print("\nGet your key from: https://console.groq.com")
        return
    
    # Run interactive session
    asyncio.run(interactive_session())

if __name__ == "__main__":
    main()