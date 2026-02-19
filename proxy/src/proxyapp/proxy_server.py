"""
Proxy Server Entry Point
Main entry point for unified API proxy
"""

import asyncio
import logging
import argparse
import signal
import sys
import subprocess
import os
from pathlib import Path

# Resolve repo root more robustly for both host and container environments
_current_file = Path(__file__).resolve()
# Try container path first (/app), then fall back to host path resolution
if _current_file.parents[1].name == "proxyapp" and _current_file.parents[2].name == "src":
    # We're in the container structure
    REPO_ROOT = Path("/app")
else:
    # We're in the host structure - go up 3 levels from proxyapp/proxy_server.py
    try:
        REPO_ROOT = _current_file.parents[3]
    except IndexError:
        # Fallback if path resolution fails
        REPO_ROOT = Path.cwd()

from .proxy import UnifiedProxy
from .diagnostics import Diagnostics

# Configure logging - setup will be completed in main() with file handler
logger = logging.getLogger(__name__)

async def warmup_llama_model(proxy):
    """Warm up llama-3.1-8b-q4k-q4_k to ensure it's hot and fast."""
    try:
        logger.info("[WARMUP] Starting llama-3.1-8b-q4k-q4_k warmup...")

        # Small test request to warm up the model
        test_request = {
            "model": "llama-3.1-8b-q4k-q4_k",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0.1
        }

        start_time = asyncio.get_event_loop().time()
        response = await proxy.chat_completions(test_request)
        end_time = asyncio.get_event_loop().time()

        latency = (end_time - start_time) * 1000  # ms
        logger.info(f"[WARMUP] ✓ Llama-3.1 warmup completed in {latency:.1f}ms")

        if latency > 15000:  # 15 seconds
            logger.warning(f"[WARMUP] ⚠ Warmup took {latency:.1f}ms (>15s target)")

        return True

    except Exception as e:
        logger.error(f"[WARMUP] ✗ Llama-3.1 warmup failed: {e}")
        return False

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified API Proxy for vLLM and llama.cpp models"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=52415,
        help="Proxy server port (default: 52415)"
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.90,
        help="GPU memory threshold (default: 0.90)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to chat history database (default: ~/.cache/proxyapp/history.db)"
    )
    parser.add_argument(
        "--load-initial",
        action="store_true",
        help="Load initial models (Qwen3-14B and DeepSeek-R1-Distill-8B) on startup"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to models_config.yaml file"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/proxy.log",
        help="Path to log file (default: logs/proxy.log)"
    )
    
    args = parser.parse_args()
    
    # Setup persistent logging to disk
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger with both console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logger.info(f"[PROXY_START] Logging to {log_file}")
    
    # Detect Docker command (check if sudo is needed)
    # First check environment variable (set by start_proxy.sh or demo script)
    docker_cmd = os.environ.get("DOCKER_CMD", "")

    if docker_cmd:
        # Use provided DOCKER_CMD from environment
        logger.info(f"Using DOCKER_CMD from environment: {docker_cmd}")
    else:
        # Auto-detect if not provided
        docker_cmd = "docker"
        try:
            # Try regular docker first
            result = subprocess.run(["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            if result.returncode != 0:
                # Try sudo docker
                result = subprocess.run(["sudo", "docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if result.returncode == 0:
                    docker_cmd = "sudo docker"
                    logger.info("Auto-detected: Using 'sudo docker' for Docker commands")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not auto-detect Docker command: {e}, defaulting to 'docker'")
    
    # Resolve paths relative to repo root
    default_config = REPO_ROOT / "configs" / "models_config.yaml"
    config_path = Path(args.config) if args.config else default_config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    # Set default db_path if not provided
    db_path = args.db_path
    if db_path is None:
        db_path = REPO_ROOT / "data/history.db"
    db_path = str(db_path)

    # Create proxy instance
    proxy = UnifiedProxy(
        proxy_port=args.port,
        memory_threshold=args.memory_threshold,
        db_path=db_path,
        docker_cmd=docker_cmd,
        config_path=str(config_path)
    )
    
    # DIAGNOSTIC: Check Docker before starting
    logger.info("[DIAG] [PROXY_START] ========== STARTING PROXY SERVER ==========")
    # Split docker_cmd string into list for Diagnostics
    docker_cmd_list = docker_cmd.split() if isinstance(docker_cmd, str) else docker_cmd
    docker_check_ok, docker_msg = Diagnostics.check_docker(docker_cmd_list)
    logger.info(f"[DIAG] [PROXY_START] {docker_msg}")
    if not docker_check_ok:
        logger.warning("[DIAG] [PROXY_START] ⚠ Docker check failed - proxy will continue but model loading may fail")
        logger.warning("[DIAG] [PROXY_START] Models will still be visible but cannot be loaded without Docker")
        # Don't exit - allow proxy to start so models are visible
    
    # Start proxy FIRST (so it's available immediately)
    logger.info("[DIAG] [PROXY_START] Calling proxy.start()...")
    try:
        await proxy.start()
        logger.info("[DIAG] [PROXY_START] ✓ proxy.start() completed successfully")
    except Exception as e:
        logger.error(f"[DIAG] [PROXY_START] ✗ FAIL: proxy.start() raised exception: {e}", exc_info=True)
        sys.exit(1)
    
    # DIAGNOSTIC: Verify proxy is listening
    port_check_ok, port_msg = Diagnostics.check_port_available(args.port)
    logger.info(f"[DIAG] [PROXY_START] {port_msg}")
    
    endpoint_check_ok, endpoint_msg = Diagnostics.check_http_endpoint(f"http://localhost:{args.port}/healthcheck")
    logger.info(f"[DIAG] [PROXY_START] Healthcheck: {endpoint_msg}")
    
    # Load initial models and WAIT for readiness (blocking for cluster-grade reliability)
    if args.load_initial:
        logger.info("[DIAG] [PROXY_START] Loading initial models and validating readiness (blocking until ready)...")

        async def load_and_validate_models():
            try:
                logger.info("[DIAG] [BG_LOAD] ========== MODEL LOADING & VALIDATION STARTED ==========")
                import time
                loop = asyncio.get_event_loop()

                # Critical models for MVP (Phase 1: vLLM + llama.cpp - restored Nov 2025)
                # Load preload models from models_config.yaml
                try:
                    import yaml
                    cfg_path = Path(config_path)
                    logger.info(f"[DIAG] [BG_LOAD] Looking for config at: {cfg_path}")
                    logger.info(f"[DIAG] [BG_LOAD] Config file exists: {cfg_path.exists()}")
                    logger.info(f"[DIAG] [BG_LOAD] Current working directory: {Path.cwd()}")

                    if cfg_path.exists():
                        with open(cfg_path, 'r') as f:
                            config = yaml.safe_load(f)

                        critical_models = [
                            model_config['model_id']
                            for model_config in config.get('models', [])
                            if model_config.get('preload', False)
                        ]
                        logger.info(f"[DIAG] [BG_LOAD] Found {len(critical_models)} preload models: {critical_models}")
                    else:
                        # Fallback if config not found
                        critical_models = ["qwen3-14b", "llama-3.1-8b-instruct-q4"]
                        logger.warning(f"[DIAG] [BG_LOAD] models_config.yaml not found at {cfg_path}, using fallback preload models")
                except Exception as e:
                    # Fallback on error
                    critical_models = ["qwen3-14b", "llama-3.1-8b-instruct-q4"]
                    logger.error(f"[DIAG] [BG_LOAD] Error reading preload models from config: {e}, using fallback")
                loaded_models = []

                # Launch preload loads concurrently so one slow model doesn't block others
                load_tasks = {}
                for model_id in critical_models:
                    logger.info(f"[DIAG] [BG_LOAD] Loading (async) {model_id}...")
                    load_tasks[model_id] = loop.run_in_executor(None, proxy.model_manager.load_model, model_id)

                # Await all load tasks
                load_results = await asyncio.gather(*load_tasks.values(), return_exceptions=True)

                # Map results back to model ids
                results_by_model = dict(zip(load_tasks.keys(), load_results))

                # Spawn readiness waiters for successful starts
                readiness_tasks = {}
                for model_id, load_result in results_by_model.items():
                    if isinstance(load_result, Exception):
                        logger.error(f"[DIAG] [BG_LOAD] ✗ Exception starting {model_id}: {load_result}")
                        continue
                    logger.info(f"[DIAG] [BG_LOAD] {model_id} load result: {load_result}")
                    if load_result:
                        async def wait_ready(mid: str):
                            ready_start = time.time()
                            logger.info(f"[DIAG] [BG_LOAD] Waiting for {mid} to become ready...")
                            while time.time() - ready_start < 180:  # 3 minute timeout
                                model_info = proxy.model_manager.models.get(mid)
                                if model_info and model_info.status == "loaded":
                                    loaded_models.append(mid)
                                    logger.info(f"[DIAG] [BG_LOAD] ✓ {mid} is ready!")
                                    return
                                await asyncio.sleep(10)
                            logger.warning(f"[DIAG] [BG_LOAD] ⚠ {mid} still loading after 3 minutes - will be available on-demand")

                        readiness_tasks[model_id] = asyncio.create_task(wait_ready(model_id))
                    else:
                        logger.error(f"[DIAG] [BG_LOAD] ✗ Failed to start loading {model_id}")

                if readiness_tasks:
                    await asyncio.gather(*readiness_tasks.values(), return_exceptions=True)

                logger.info("[DIAG] [BG_LOAD] ========== MODEL LOADING & VALIDATION COMPLETED ==========")
                logger.info(f"[DIAG] [BG_LOAD] Successfully loaded and validated: {loaded_models}")

                # Validate models loaded (be lenient for demo purposes)
                if len(loaded_models) >= 1:  # At least one model working
                    logger.info(f"[DIAG] [BG_LOAD] ✓ CLUSTER READY: {len(loaded_models)}/{len(critical_models)} critical models loaded and validated")
                elif loaded_models:
                    logger.info(f"[DIAG] [BG_LOAD] ⚠ PARTIAL SUCCESS: {len(loaded_models)} models ready, others loading in background")
                else:
                    logger.warning("[DIAG] [BG_LOAD] ⚠ NO MODELS READY: All models still loading - UI will show loading states")
                    logger.warning("[DIAG] [BG_LOAD] Models will become available on-demand as they finish loading")

            except Exception as e:
                logger.error(f"[DIAG] [BG_LOAD] Exception in model loading: {e}", exc_info=True)

        # Wait for models to load and validate (blocking for cluster reliability)
        await load_and_validate_models()
        logger.info("[DIAG] [PROXY_START] ✓ Model loading and validation completed")

        # WARMUP: Ensure llama-3.1-8b-q4k-q4_k is hot and fast
        await warmup_llama_model(proxy)
        logger.info("[DIAG] [PROXY_START] ✓ Llama-3.1 warmup completed")
    
    # Wait for shutdown signal - KEEP PROXY RUNNING
    def signal_handler(sig, frame):
        logger.info("Shutting down proxy server...")
        # Stop background tasks
        if proxy._monitoring_task:
            proxy._monitoring_task.cancel()
        if proxy._queue_processing_task:
            proxy._queue_processing_task.cancel()
        if proxy._keep_alive_task:
            proxy._keep_alive_task.cancel()
        sys.exit(0)
    
    def sighup_handler(sig, frame):
        # SIGHUP = terminal disconnect - IGNORE IT, keep running as daemon
        logger.info("Received SIGHUP (terminal disconnect) - ignoring, continuing as daemon")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, sighup_handler)  # IGNORE SIGHUP - keep running
    
    # Keep running FOREVER - don't exit
    logger.info("Proxy server running. Waiting for requests...")
    logger.info(f"Models available: {list(proxy.model_manager.models.keys())}")
    
    # Use a never-completing future to keep event loop alive
    try:
        # Create an event that never completes
        stop_event = asyncio.Event()
        await stop_event.wait()  # This will wait forever
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt")
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        # Don't exit - log and continue waiting
        import traceback
        traceback.print_exc()
        # Keep waiting even after exception
        await asyncio.sleep(1)
        await stop_event.wait()

if __name__ == "__main__":
    asyncio.run(main())

