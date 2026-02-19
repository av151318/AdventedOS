"""
Memory Management System for EXO
Implements LRU eviction when RAM utilization exceeds threshold
"""

import asyncio
import logging
import psutil
import time
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_gb: float
    used_gb: float
    available_gb: float
    utilization_percent: float

class MemoryManager:
    """LRU memory management for model containers"""

    def __init__(
        self,
        threshold_percent: float = 90.0,
        check_interval: int = 60,
        evict_callback: Optional[Callable[[str], bool]] = None,
        min_idle_seconds: int = 60,
    ):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.evict_callback = evict_callback
        self.min_idle_seconds = min_idle_seconds
        self.model_access_times: Dict[str, float] = {}
        self._monitoring = False
        self._task: Optional[asyncio.Task] = None
        # Rate-limit noisy threshold warnings
        self._last_threshold_warning_ts = 0.0
        self._threshold_warning_interval_s = 10 * 60  # 10 minutes

    def start_monitoring(self):
        """Start background memory monitoring"""
        if self._monitoring:
            logger.warning("Memory monitoring already running")
            return

        self._monitoring = True
        self._task = asyncio.create_task(self._monitor_memory())
        logger.info(f"[MEM] Started memory monitoring (threshold: {self.threshold_percent}%, interval: {self.check_interval}s)")

    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self._monitoring = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("[MEM] Stopped memory monitoring")

    def update_access_time(self, model_id: str):
        """Update last access time for a model"""
        self.model_access_times[model_id] = time.time()

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        mem = psutil.virtual_memory()
        return MemoryStats(
            total_gb=round(mem.total / (1024**3), 2),
            used_gb=round(mem.used / (1024**3), 2),
            available_gb=round(mem.available / (1024**3), 2),
            utilization_percent=round(mem.percent, 1)
        )

    async def _monitor_memory(self):
        """Background memory monitoring task"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                logger.debug(f"[MEM] Memory: {stats.used_gb}GB/{stats.total_gb}GB ({stats.utilization_percent}%)")

                if stats.utilization_percent > self.threshold_percent:
                    now = time.time()
                    if now - self._last_threshold_warning_ts >= self._threshold_warning_interval_s:
                        logger.warning(f"[MEM] ⚠ Memory threshold exceeded: {stats.utilization_percent}% > {self.threshold_percent}%")
                        self._last_threshold_warning_ts = now
                    await self._evict_least_used_model()
                else:
                    logger.debug(f"[MEM] ✓ Memory within limits: {stats.utilization_percent}%")

            except Exception as e:
                logger.error(f"[MEM] Error in memory monitoring: {e}")

            await asyncio.sleep(self.check_interval)

    async def _evict_least_used_model(self):
        """Evict the least recently used model to free memory"""
        if not self.model_access_times:
            logger.warning("[MEM] No models to evict - no access times tracked")
            return

        # Try eviction candidates in LRU order until one succeeds.
        candidates = sorted(self.model_access_times.items(), key=lambda x: x[1])  # oldest first
        for model_id, last_access in candidates:
            idle_s = time.time() - last_access
            logger.info(f"[MEM] Eviction candidate: {model_id} (last accessed {idle_s:.0f}s ago)")

            if idle_s < self.min_idle_seconds:
                logger.warning(f"[MEM] Skipping candidate: {model_id} was accessed {idle_s:.0f}s ago (< {self.min_idle_seconds}s)")
                continue

            if not self.evict_callback:
                logger.warning(f"[MEM] ⚠ LRU eviction not yet implemented - would unload {model_id}")
                return

            try:
                # Unloading can involve subprocess/docker calls; run in a thread to avoid blocking the loop.
                ok = await asyncio.to_thread(self.evict_callback, model_id)
                if ok:
                    self.model_access_times.pop(model_id, None)
                    logger.info(f"[MEM] ✓ Evicted {model_id}")
                    return
                else:
                    logger.warning(f"[MEM] Candidate not evicted (callback returned false): {model_id}")
            except Exception as e:
                logger.error(f"[MEM] Error evicting {model_id}: {e}")

        logger.info("[MEM] No eviction candidates could be evicted")

    def get_eviction_candidates(self) -> List[str]:
        """Get list of models sorted by access time (oldest first)"""
        if not self.model_access_times:
            return []

        # Sort by access time (oldest first)
        sorted_models = sorted(self.model_access_times.items(), key=lambda x: x[1])
        return [model_id for model_id, _ in sorted_models]

    def force_gc(self):
        """Force garbage collection to free memory"""
        import gc
        collected = gc.collect()
        logger.info(f"[MEM] Forced GC collected {collected} objects")

    def log_memory_report(self):
        """Log detailed memory report"""
        stats = self.get_memory_stats()
        logger.info(f"[MEM] Memory Report:")
        logger.info(f"[MEM]   Total: {stats.total_gb}GB")
        logger.info(f"[MEM]   Used: {stats.used_gb}GB")
        logger.info(f"[MEM]   Available: {stats.available_gb}GB")
        logger.info(f"[MEM]   Utilization: {stats.utilization_percent}%")
        logger.info(f"[MEM]   Threshold: {self.threshold_percent}%")

        if self.model_access_times:
            logger.info(f"[MEM]   Model Access Times:")
            for model_id, access_time in sorted(self.model_access_times.items(), key=lambda x: x[1]):
                ago = time.time() - access_time
                logger.info(f"[MEM]     {model_id}: {ago:.0f}s ago")
        else:
            logger.info(f"[MEM]   No model access times tracked")

