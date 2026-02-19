"""
Request Queue System
Queues requests when models are loading
Processes queue when models become ready
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class QueuedRequest:
    """Represents a queued request"""
    request_id: str
    model_id: str
    request_data: Dict[str, Any]
    callback: Callable
    created_at: float
    timeout: Optional[float] = None

class RequestQueue:
    """Manages request queueing for loading models"""
    
    def __init__(self):
        """Initialize request queue"""
        self.queues: Dict[str, List[QueuedRequest]] = {}  # model_id -> queue
        self.processing: Dict[str, bool] = {}  # model_id -> is_processing
    
    def enqueue(
        self,
        model_id: str,
        request_data: Dict[str, Any],
        callback: Callable,
        timeout: Optional[float] = None
    ) -> str:
        """
        Enqueue a request for a model
        
        Args:
            model_id: Model identifier
            request_data: Request data dictionary
            callback: Callback function to process request
            timeout: Optional timeout in seconds
            
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        if model_id not in self.queues:
            self.queues[model_id] = []
        
        queued_request = QueuedRequest(
            request_id=request_id,
            model_id=model_id,
            request_data=request_data,
            callback=callback,
            created_at=datetime.now().timestamp(),
            timeout=timeout
        )
        
        self.queues[model_id].append(queued_request)
        logger.info(f"Enqueued request {request_id} for model {model_id}")
        
        return request_id
    
    def dequeue(self, model_id: str) -> Optional[QueuedRequest]:
        """
        Dequeue next request for a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            QueuedRequest or None if queue is empty
        """
        if model_id not in self.queues or not self.queues[model_id]:
            return None
        
        return self.queues[model_id].pop(0)
    
    def get_queue_size(self, model_id: str) -> int:
        """Get queue size for a model"""
        return len(self.queues.get(model_id, []))
    
    def clear_queue(self, model_id: str):
        """Clear queue for a model"""
        if model_id in self.queues:
            self.queues[model_id].clear()
    
    def process_queue(self, model_id: str):
        """
        Process queue for a model (mark as processing)
        
        Args:
            model_id: Model identifier
        """
        self.processing[model_id] = True
    
    def finish_processing(self, model_id: str):
        """
        Mark queue processing as finished
        
        Args:
            model_id: Model identifier
        """
        self.processing[model_id] = False
    
    def is_processing(self, model_id: str) -> bool:
        """Check if queue is being processed"""
        return self.processing.get(model_id, False)
    
    def get_all_queues(self) -> Dict[str, int]:
        """Get queue sizes for all models"""
        return {
            model_id: len(queue)
            for model_id, queue in self.queues.items()
        }



