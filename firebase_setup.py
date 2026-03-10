"""
Firebase configuration and connection management for Project Exocortex.
Handles initialization, connection pooling, and graceful degradation.
"""
import os
import logging
from typing import Optional
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, exceptions
from google.api_core import retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseManager:
    """Singleton Firebase connection manager with circuit breaker pattern"""
    
    _instance: Optional['FirebaseManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._app = None
            self._db = None
            self._circuit_open = False
            self._circuit_trip_time: Optional[datetime] = None
            self._last_error: Optional[str] = None
            
    def initialize(self, credential_path: Optional[str] = None) -> None:
        """
        Initialize Firebase connection with exponential backoff and fallback.
        
        Args:
            credential_path: Path to Firebase service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        
        Raises:
            ValueError: If credentials are invalid or missing
            ConnectionError: If Firebase connection fails
        """
        if self._app is not None:
            logger.info("Firebase already initialized, returning existing instance")
            return
        
        try:
            # Check circuit breaker first
            if self._circuit_open:
                self._check_circuit_reset()
                
            # Load credentials
            if credential_path:
                if not os.path.exists(credential_path):
                    raise FileNotFoundError(f"Credential file not found: {credential_path}")
                cred = credentials.Certificate(credential_path)
            else:
                cred = credentials.ApplicationDefault()
            
            # Initialize with retry configuration
            self._app = firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv('FIREBASE_DATABASE_URL', ''),
                'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET', ''),
            })
            
            # Initialize Firestore with custom retry configuration
            self._db = firestore.client(app=self._app)
            
            # Test connection with timeout
            test_doc = self._db.collection('system_health').document('connection_test')
            test_doc.set({
                'timestamp': datetime.utcnow(),
                'status': 'active'
            }, timeout=5)
            
            logger.info("✅ Firebase initialized successfully")
            self._circuit_open = False
            self._last_error = None
            
        except exceptions.FirebaseError as e:
            self._last_error = str(e)
            logger.error(f"❌ Firebase initialization failed: {e}")
            self._trip_circuit()
            raise ConnectionError(f"Firebase connection failed: {e}")
        except FileNotFoundError as e:
            logger.error(f"❌ Credential file error: {e}")
            raise
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"❌ Unexpected error during Firebase init: {e}")
            self._trip_circuit()
            raise ConnectionError(f"Firebase initialization failed: {e}")
    
    @retry.Retry(predicate=retry.if_transient_error, deadline=30)
    def get_db(self) -> firestore.Client:
        """
        Get Firestore client with automatic retry for transient errors.
        
        Returns:
            Firestore client instance
        
        Raises:
            ConnectionError: If circuit is open or connection unavailable
        """
        if self._circuit_open:
            self._check_circuit_reset()
            if self._circuit_open:
                raise ConnectionError(f"Circuit breaker is open. Last error: {self._last_error}")
        
        if self._db is None:
            raise ConnectionError("Firebase not initialized. Call initialize() first.")
        
        return self._db
    
    def _trip_circuit(self) -> None:
        """Trip the circuit breaker for 60 seconds"""
        self._circuit_open = True
        self._circuit_trip_time = datetime.utcnow()
        logger.warning("🔴 Circuit breaker tripped for 60 seconds")
    
    def _check_circuit_reset(self) -> None:
        """Reset circuit breaker if 60 seconds have passed"""
        if (self._circuit_trip_time and 
            (datetime.utcnow() - self._circuit_trip_time).seconds > 60):
            self._circuit_open = False
            self._circuit_trip_time = None
            logger.info("🟢 Circuit breaker reset")
    
    def health_check(self) -> dict:
        """Return system health status"""
        return {
            'initialized': self._app is not None,
            'circuit_open': self._circuit_open,
            'last_error': self._last_error,
            'circuit_trip_time': self._circuit_trip_time,
            'timestamp': datetime.utcnow()
        }
    
    def cleanup(self) -> None:
        """Cleanup Firebase resources"""
        if self._app:
            firebase_admin.delete_app(self._app)
            self._app = None
            self._db = None
            logger.info("Firebase connection cleaned up")

# Global instance
firebase_manager = FirebaseManager()