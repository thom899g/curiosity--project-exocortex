"""
Robust Perception Agent for multi-source data ingestion.
Implements proxy rotation, user-agent spoofing, and adversarial detection.
"""
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from firebase_admin import firestore

from firebase_setup import firebase_manager

logger = logging.getLogger(__name__)

@dataclass
class SourceConfig:
    """Configuration for a data source"""
    source_type: str  # 'twitter', 'dex', 'freelance_board'
    endpoints: List[str]
    rate_limit_requests: int
    rate_limit_period: int  # seconds
    required_headers: Dict[str, str]
    parser_type: str

class RobustPerceptionAgent:
    """Robust agent for fetching data with stateful session management"""
    
    def __init__(self, agent_id: str, config: SourceConfig):
        """
        Initialize perception agent with configuration.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Source configuration
        
        Raises:
            ValueError: If configuration is invalid
        """
        self.agent_id = agent_id
        self.config = config
        self.session_state: Dict[str, Any] = {}
        self.circuit_breaker = False
        self.failure_count = 0
        self.last_success: Optional[datetime] = None
        self.request_count = 0
        self.rate_limit_reset: Optional[datetime] = None
        
        # Initialize session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Load configuration from Firestore
        self._load_agent_config()
        
        logger.info(f"Perception Agent {agent_id} initialized for {config.source_type}")
    
    def _load_agent_config(self) -> None:
        """Load agent configuration from Firestore"""
        try:
            db = firebase_manager.get_db()
            config_ref = db.collection('perception_config').document(self.agent_id)
            config_data = config_ref.get().to_dict()
            
            if config_data:
                # Update session state from persisted config
                self.session_state.update(config_data.get('session_state', {}))
                self.failure_count = config_data.get('failure_count', 0)
                logger.debug(f"Loaded persisted state for agent {self.agent_id}")
        except Exception as e:
            logger.warning(f"Could not load agent config: {e}")
    
    def _save_agent_state(self) -> None:
        """Persist agent state to Firestore"""
        try:
            db = firebase_manager.get_db()
            config_ref = db.collection('perception_config').document(self.agent_id)
            config_ref.set({
                'session_state': self.session_state,
                'failure_count': self.failure_count,
                'last_success': self.last_success,
                'updated_at': datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're rate limited, wait if necessary"""
        if self.rate_limit_reset and datetime.utcnow() < self.rate_limit_reset:
            wait_seconds = (self.rate_limit_reset - datetime.utcnow()).seconds
            logger.warning(f"Rate limited, waiting {wait_seconds} seconds")
            time.sleep(wait_seconds)
            return False
        
        # Check requests per period
        if (self.request_count >= self.config.rate_limit_requests and
            self.last_success and 
            (datetime.utcnow() - self.last_success).seconds < self.config.rate_limit_period):
            
            wait_time = self.config.rate_limit_period - (datetime.utcnow() - self.last_success).seconds
            logger.info(f"Approaching rate limit, cooling down for {wait_time}s")
            time.sleep(wait_time)
            self.request_count = 0
        
        return True
    
    def _generate_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid fingerprinting"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'application/json, text/html, application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # Add source-specific headers
        headers.update(self.config.required_headers)
        
        # Add cookies if we have them
        if 'cookies' in self.session_state:
            headers['Cookie'] = '; '.join([f"{k}={v}" for k, v in self.session_state['cookies'].items()])
        
        return headers
    
    def _detect_adversarial_response(self, response: requests.Response) -> bool:
        """Detect if we're being blocked or challenged"""
        indicators = [
            ('captcha', 'CAPTCHA' in response.text),
            ('blocked', response.status_code == 403),
            ('rate_limited', response.status_code == 429),
            ('cloudflare', 'cloudflare' in response.headers.get('server', '').lower()),
            ('access_denied', 'access denied' in response.text.lower()),
        ]
        
        for indicator_name, detected in indicators:
            if detected:
                logger.warning(f"Detected adversarial response: {indicator_name}")
                return True
        
        # Check for suspiciously small response (block page)
        if len(response.text) < 1000 and response.status_code != 200:
            return True
        
        return False
    
    def _handle_adversarial_response(self) -> None:
        """Handle detected blocking"""
        self.failure_count += 1
        
        if self.failure_count > 3:
            # Clear session state
            self.session_state = {}
            logger.info("Cleared session state due to repeated blocks")
        
        if self.failure_count > 5:
            self.circuit_breaker = True
            logger.error("Circuit breaker tripped due to adversarial responses")
        
        # Exponential backoff
        backoff = min(2 ** self.failure_count, 60)
        time.sleep(backoff)
    
    def _normalize_data(self, raw_data: Any, source_type: str) -> Dict[str, Any]:
        """Normalize data from different sources to common format"""
        normalized = {
            'source': source_type,
            'raw_data': raw_data,
            'timestamp': datetime.utcnow(),
            'agent_id': self.agent_id,
            'confidence': 1.0
        }
        
        try:
            if source_type == 'twitter':
                # Extract relevant fields from Twitter response
                if isinstance(raw_data, dict):
                    normalized.update({
                        'content': raw_data.get('text', ''),
                        'author': raw_data.get('user', {}).get('screen_name', ''),
                        'engagement': raw_data.get('favorite_count', 0) + raw_data.get('retweet_count', 0),
                        'sentiment_score': 0.0  # Placeholder for sentiment analysis
                    })
            elif source_type == 'dex':
                # DEX data normalization
                if isinstance(raw_data, dict):
                    normalized.update({
                        'token_pair': raw_data.get('symbol', ''),
                        'price': float(raw_data.get('last', 0)),
                        'volume_24h': float(raw_data.get('quoteVolume', 0)),
                        'liquidity': float(raw_data.get('baseVolume', 0))
                    })
            
            logger.debug(f"Normalized data from {source_type}")
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            normalized['confidence'] = 0.3
        
        return normalized
    
    def fetch(self, endpoint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch data from source with robust error handling.
        
        Args:
            endpoint: Specific endpoint to fetch from
        
        Returns:
            Normalized data dictionary or None if failed
        """
        if self.circuit_breaker:
            logger.warning(f"Circuit breaker open for agent {self.agent_id}")
            return None
        
        if not self._check_rate_limit():
            return None
        
        target_endpoint = endpoint or random.choice(self.config.endpoints)
        
        try:
            headers = self._generate_headers()
            
            logger.info(f"Fetching from {target_endpoint}")
            
            # Add jitter to avoid timing patterns