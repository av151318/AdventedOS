"""
GPU Memory Monitor
Monitors GPU memory utilization using nvidia-smi
Triggers model unloading when memory exceeds 90% threshold
"""

import subprocess
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor GPU memory using nvidia-smi"""
    
    def __init__(self, threshold: float = 0.90):
        """
        Initialize memory monitor
        
        Args:
            threshold: Memory utilization threshold (default: 0.90 = 90%)
        """
        self.threshold = threshold
        self._check_nvidia_smi()
    
    def _check_nvidia_smi(self):
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    
    def get_gpu_memory_info(self) -> List[Dict]:
        """
        Get memory information for all GPUs
        
        Returns:
            List of GPU memory dictionaries with keys:
            - gpu_id: GPU index
            - total_memory_mb: Total memory in MB
            - used_memory_mb: Used memory in MB
            - free_memory_mb: Free memory in MB
            - utilization: Utilization percentage (0.0-1.0)
        """
        try:
            # Query GPU memory info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.error(f"nvidia-smi query failed: {result.stderr}")
                return []
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4:
                    continue
                
                # Skip lines with [N/A] values (GPU not available or error)
                if any(p in ['[N/A]', 'N/A', ''] for p in parts):
                    continue
                
                try:
                    gpu_id = int(parts[0])
                    total_mb = int(parts[1])
                    used_mb = int(parts[2])
                    free_mb = int(parts[3])
                except ValueError:
                    # Skip invalid lines
                    continue
                
                utilization = used_mb / total_mb if total_mb > 0 else 0.0
                
                gpus.append({
                    "gpu_id": gpu_id,
                    "total_memory_mb": total_mb,
                    "used_memory_mb": used_mb,
                    "free_memory_mb": free_mb,
                    "utilization": utilization
                })
            
            return gpus
            
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi query timed out")
            return []
        except Exception as e:
            logger.error(f"Error querying GPU memory: {e}")
            return []
    
    def check_memory_threshold(self) -> bool:
        """
        Check if any GPU exceeds memory threshold
        
        Returns:
            True if any GPU exceeds threshold, False otherwise
        """
        gpus = self.get_gpu_memory_info()
        
        for gpu in gpus:
            if gpu["utilization"] >= self.threshold:
                logger.warning(
                    f"GPU {gpu['gpu_id']} memory utilization {gpu['utilization']:.1%} "
                    f"exceeds threshold {self.threshold:.1%}"
                )
                return True
        
        return False
    
    def get_max_utilization_gpu(self) -> Optional[Dict]:
        """
        Get GPU with highest memory utilization
        
        Returns:
            GPU dictionary or None if no GPUs found
        """
        gpus = self.get_gpu_memory_info()
        
        if not gpus:
            return None
        
        return max(gpus, key=lambda g: g["utilization"])
    
    def get_total_memory_utilization(self) -> float:
        """
        Get average memory utilization across all GPUs
        
        Returns:
            Average utilization (0.0-1.0)
        """
        gpus = self.get_gpu_memory_info()
        
        if not gpus:
            return 0.0
        
        total_utilization = sum(gpu["utilization"] for gpu in gpus)
        return total_utilization / len(gpus)

