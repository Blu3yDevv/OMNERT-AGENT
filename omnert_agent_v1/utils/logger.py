"""
Advanced logging configuration
"""
import logging
import sys
from datetime import datetime

class CustomLogger:
    def __init__(self, name="omnert_agent", log_file="agent.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_phase(self, phase: str, details: str):
        """Log a phase transition"""
        self.logger.info(f"PHASE: {phase.upper()} - {details}")
    
    def log_search(self, query: str, results: int):
        """Log search activity"""
        self.logger.info(f"SEARCH: '{query}' -> {results} results")
    
    def log_reasoning(self, steps: int, confidence: float):
        """Log reasoning activity"""
        self.logger.info(f"REASONING: {steps} steps -> {confidence:.0%} confidence")