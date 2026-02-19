"""
Model Manager
Manages model lifecycle: loading, unloading, tracking usage
Handles both vLLM (NGC containers) and llama.cpp backends
vLLM support restored for DGX Spark (Nov 2025)
"""

import subprocess
import time
import logging
import os
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Resolve repo root more robustly for both host and container environments
_current_file = Path(__file__).resolve()
# Try container path first (/app), then fall back to host path resolution
if _current_file.parents[1].name == "proxyapp" and _current_file.parents[2].name == "src":
    # We're in the container structure
    REPO_ROOT = Path("/app")
else:
    # We're in the host structure - go up 3 levels from proxyapp/model_manager.py
    try:
        REPO_ROOT = _current_file.parents[3]
    except IndexError:
        # Fallback if path resolution fails
        REPO_ROOT = Path.cwd()

logger = logging.getLogger(__name__)

class ModelBackend(Enum):
    """Model backend types"""
    VLLM = "vllm"
    LLAMACPP = "llamacpp"

@dataclass
class ModelInfo:
    """Model information"""
    model_id: str
    backend: ModelBackend
    port: int
    model_path: str
    config_path: Optional[str] = None
    gpu_layers: int = 0
    context_size: int = 4096
    threads: int = 8
    container_name: Optional[str] = None  # Process/container name
    process: Optional[subprocess.Popen] = None  # Native process for llama.cpp
    status: str = "unloaded"  # unloaded, loading, loaded, unloading
    last_access_time: float = 0.0
    request_count: int = 0
    openapi_extra: Optional[Dict] = None  # Additional OpenAPI schema fields

class ModelManager:
    """Manages model lifecycle and routing"""
    
    def __init__(self, vllm_env_path: Optional[str] = None, docker_cmd: str = "docker", config_path: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            vllm_env_path: Path to vLLM virtual environment (for vLLM models)
            docker_cmd: Docker command to use (default: "docker", can be "sudo docker")
            config_path: Path to models_config.yaml file
        """
        self.vllm_env_path = vllm_env_path or "vllm_env"
        self.docker_cmd = docker_cmd.split()  # Split into list for subprocess
        self.models: Dict[str, ModelInfo] = {}
        self.config_path = config_path
        
        if config_path and Path(config_path).exists():
            self._load_models_from_config(config_path)
        else:
            self._init_default_models()
    
    def _load_models_from_config(self, config_path: str):
        """Load models from YAML config file"""
        import yaml
        
        logger.info(f"Loading models from config: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for model_config in config.get('models', []):
            model_id = model_config['model_id']
            backend = ModelBackend(model_config['backend'])
            
            model_info = ModelInfo(
                model_id=model_id,
                backend=backend,
                port=model_config['port'],
                model_path=model_config['model_path'],
                config_path=model_config.get('config_path'),
                gpu_layers=model_config.get('gpu_layers', 0),
                context_size=model_config.get('context_size', 4096),
                threads=model_config.get('threads', 8),
                openapi_extra=model_config.get('openapi_extra')
            )
            
            self.models[model_id] = model_info
            logger.info(f"Registered {backend.value} model: {model_id} on port {model_info.port}")
    
    def _init_default_models(self):
        """Initialize ALL available models from configs and GGUF files"""
        # Determine workspace directory (where vllm_models/ exists)
        workspace_dir = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))

        # Start port assignments (vLLM: 8001+, llama.cpp: 8080+)
        vllm_port = 8001
        llamacpp_port = 8080
        
        # Register all vLLM models from configs/vllm_configs (fallback to legacy root)
        # RESTORED: vLLM now supported on DGX Spark with NGC containers (Nov 2025)
        possible_config_dirs = [
            workspace_dir / "configs" / "vllm_configs",
            workspace_dir / "vllm_configs",
        ]
        configs_dir = next((p for p in possible_config_dirs if p.exists()), possible_config_dirs[0])
        used_ports = set()  # Track used ports to avoid conflicts

        if configs_dir.exists():  # ENABLED - NGC containers now support ARM64
            for config_file in configs_dir.glob("*.yaml"):
                model_id = config_file.stem  # e.g., "qwen3-14b" from "qwen3-14b.yaml"
                
                # Read config to get model name and port
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    # Get model name from config
                    model_name = config.get("model", model_id.replace("-", "/"))
                    # Extract model directory name (last part after /)
                    if "/" in model_name:
                        model_dir_name = model_name.split("/")[-1]
                    else:
                        model_dir_name = model_name
                    
                    # Check if directory exists, try variations
                    possible_paths = [
                        f"vllm_models/{model_dir_name}",
                        f"vllm_models/{model_name.replace('/', '-')}",
                        f"vllm_models/{model_id.replace('-', '-')}"
                    ]
                    model_path = possible_paths[0]  # Default to first
                    for path_candidate in possible_paths:
                        if (workspace_dir / path_candidate).exists():
                            model_path = path_candidate
                            break
                    
                    # Try to use port from config, but ensure uniqueness
                    config_port = config.get("port")
                    if config_port and config_port not in used_ports:
                        assigned_port = config_port
                    else:
                        # Assign next available port
                        while vllm_port in used_ports:
                            vllm_port += 1
                        assigned_port = vllm_port
                        vllm_port += 1
                    used_ports.add(assigned_port)
                except Exception as e:
                    logger.warning(f"Error reading config {config_file}: {e}, using defaults")
                    # Fallback: use model_id as directory name
                    model_path = f"vllm_models/{model_id.replace('-', '-')}"
                    while vllm_port in used_ports:
                        vllm_port += 1
                    assigned_port = vllm_port
                    used_ports.add(assigned_port)
                    vllm_port += 1
                
                self.models[model_id] = ModelInfo(
                    model_id=model_id,
                    backend=ModelBackend.VLLM,
                    port=assigned_port,
                    model_path=model_path,
                    config_path=str(config_file.relative_to(workspace_dir)),
                    status="unloaded"
                )
                logger.info(f"Registered vLLM model: {model_id} on port {assigned_port}")
        
        # Register all llama.cpp models from vllm_models/ (GGUF files)
        models_dir = workspace_dir / "vllm_models"
        if models_dir.exists():
            # Find all directories containing GGUF files
            for model_dir in models_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                gguf_files = list(model_dir.glob("*.gguf"))
                if not gguf_files:
                    continue
                
                # Use directory name as model_id base
                model_id_base = model_dir.name.lower().replace("_", "-")
                
                # Register best quantization first (Q8_0 > Q6_K > Q4_K_M > others)
                quant_priority = ["Q8_0", "Q6_K", "Q4_K_M", "Q4_K", "Q3_K_L", "Q3_K_M"]
                
                registered_quants = set()
                for quant in quant_priority:
                    for gguf_file in gguf_files:
                        if quant in gguf_file.name and quant not in registered_quants:
                            model_id = f"{model_id_base}-{quant.lower()}"
                            if model_id not in self.models:  # Avoid duplicates
                                self.models[model_id] = ModelInfo(
                                    model_id=model_id,
                                    backend=ModelBackend.LLAMACPP,
                                    port=llamacpp_port,
                                    model_path=str(gguf_file.relative_to(workspace_dir)),
                                    status="unloaded"
                                )
                                llamacpp_port += 1
                                registered_quants.add(quant)
                                logger.info(f"Registered llama.cpp model: {model_id} on port {llamacpp_port - 1}")
                                break
                
                # Register any remaining GGUF files not already registered
                for gguf_file in gguf_files:
                    # Check if this file is already registered
                    already_registered = any(
                        model.model_path == str(gguf_file.relative_to(workspace_dir))
                        for model in self.models.values()
                    )
                    if not already_registered:
                        # Extract quantization from filename or use default
                        quant_match = None
                        for q in quant_priority:
                            if q in gguf_file.name:
                                quant_match = q.lower()
                                break
                        
                        if quant_match:
                            model_id = f"{model_id_base}-{quant_match}"
                        else:
                            # Use filename without extension as model_id
                            model_id = f"{model_id_base}-{gguf_file.stem.split('-')[-1].lower()}"
                        
                        if model_id not in self.models:
                            self.models[model_id] = ModelInfo(
                                model_id=model_id,
                                backend=ModelBackend.LLAMACPP,
                                port=llamacpp_port,
                                model_path=str(gguf_file.relative_to(workspace_dir)),
                                status="unloaded"
                            )
                            llamacpp_port += 1
                            logger.info(f"Registered llama.cpp model: {model_id} on port {llamacpp_port - 1}")
        
        logger.info(f"Total models registered: {len(self.models)}")
    
    def add_model(
        self,
        model_id: str,
        backend: ModelBackend,
        port: int,
        model_path: str,
        config_path: Optional[str] = None,
        gpu_layers: int = 0,
        context_size: int = 4096,
        threads: int = 8
    ):
        """Add a model to the manager"""
        self.models[model_id] = ModelInfo(
            model_id=model_id,
            backend=backend,
            port=port,
            model_path=model_path,
            config_path=config_path,
            gpu_layers=gpu_layers,
            context_size=context_size,
            threads=threads,
            status="unloaded"
        )
    
    def load_model(self, model_id: str) -> bool:
        """
        Load a model (start server process)
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if loading started successfully, False otherwise
        """
        logger.info(f"[DIAG] [LOAD_MODEL] Starting load_model for {model_id}")
        
        if model_id not in self.models:
            logger.error(f"[DIAG] [LOAD_MODEL] ✗ FAIL: Model {model_id} not found in models dict")
            return False
        logger.info(f"[DIAG] [LOAD_MODEL] ✓ Model {model_id} found in models dict")
        
        model = self.models[model_id]
        
        if model.status == "loaded":
            logger.info(f"[DIAG] [LOAD_MODEL] ✓ Model {model_id} already loaded")
            return True
        
        if model.status == "loading":
            logger.info(f"[DIAG] [LOAD_MODEL] ✓ Model {model_id} already loading")
            return True
        
        model.status = "loading"
        logger.info(f"[DIAG] [LOAD_MODEL] Setting status=loading for {model_id} on port {model.port}")
        
        try:
            logger.info(f"[DIAG] [LOAD_MODEL] Backend: {model.backend}")
            if model.backend == ModelBackend.VLLM:
                logger.info(f"[DIAG] [LOAD_MODEL] Calling _load_vllm_model")
                success = self._load_vllm_model(model)
                logger.info(f"[DIAG] [LOAD_MODEL] _load_vllm_model returned: {success}")
            elif model.backend == ModelBackend.LLAMACPP:
                logger.info(f"[DIAG] [LOAD_MODEL] Calling _load_llamacpp_model")
                success = self._load_llamacpp_model(model)
                logger.info(f"[DIAG] [LOAD_MODEL] _load_llamacpp_model returned: {success}")
            else:
                logger.error(f"[DIAG] [LOAD_MODEL] ✗ FAIL: Unknown backend {model.backend} for model {model_id}")
                model.status = "unloaded"
                return False
            
            if success:
                # Perform prompt-based readiness test before marking as loaded
                if self._test_model_readiness(model):
                    model.status = "loaded"
                    model.last_access_time = time.time()
                    logger.info(f"[DIAG] [LOAD_MODEL] ✓ SUCCESS: Model {model_id} loaded and ready for inference")
                    return True
                else:
                    logger.error(f"[DIAG] [LOAD_MODEL] ✗ FAIL: Model {model_id} health check passed but prompt test failed")
                    # For vLLM, if the container is running, keep it in 'loading' and allow queued requests
                    if model.backend == ModelBackend.VLLM and model.container_name:
                        try:
                            check_result = subprocess.run(
                                self.docker_cmd + ["ps", "--filter", f"name={model.container_name}", "--format", "{{.Names}}"],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if model.container_name in check_result.stdout:
                                model.status = "loading"
                                logger.info(f"[DIAG] [LOAD_MODEL] {model_id} container running; keeping status=loading for deferred readiness")
                                return True
                        except Exception as e:
                            logger.warning(f"[DIAG] [LOAD_MODEL] Error checking container status after readiness failure: {e}")
                    model.status = "error"
                    return False
            else:
                # Check if container is actually running (might have started but health check failed)
                # If container is running, mark as "loading" and let background health checker update it
                if model.container_name:
                    try:
                        check_result = subprocess.run(
                            self.docker_cmd + ["ps", "--filter", f"name={model.container_name}", "--format", "{{.Names}}"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if model.container_name in check_result.stdout:
                            # Container is running - keep status as "loading" and return True
                            # Background health checker will update to "loaded" when ready
                            logger.info(f"[DIAG] [LOAD_MODEL] ✓ Container {model.container_name} is running (health check will complete in background)")
                            logger.info(f"[DIAG] [LOAD_MODEL] Model {model_id} status remains 'loading' - background health checker will update when ready")
                            return True  # Return True so request gets queued
                    except Exception as e:
                        logger.warning(f"[DIAG] [LOAD_MODEL] Error checking container status: {e}")
                
                # Container failed to start or doesn't exist
                logger.error(f"[DIAG] [LOAD_MODEL] ✗ FAIL: Model {model_id} failed to start")
                model.status = "unloaded"
                return False
                
        except Exception as e:
            logger.error(f"[DIAG] [LOAD_MODEL] ✗ EXCEPTION loading model {model_id}: {e}", exc_info=True)
            model.status = "unloaded"
            return False
    
    def _load_vllm_model(self, model: ModelInfo) -> bool:
        """Load vLLM model using NGC Docker container (CUDA 13.0 compatible)"""
        from pathlib import Path
        import os
        
        # NGC Container configuration (matches start_vllm.sh)
        VLLM_IMAGE = "nvcr.io/nvidia/vllm:25.09-py3"  # gx2 proven reference
        # Use absolute path from repo root (or EXO_WORKSPACE override)
        WORKSPACE_DIR = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))

        # Container name: vllm-{model_id} (lowercase, matches start_vllm.sh pattern)
        container_name = f"vllm-{model.model_id.lower()}"
        model.container_name = container_name
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Container name: {container_name}")
        
        # Check if container already exists and is running
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Checking if container exists...")
        check_result = subprocess.run(
            self.docker_cmd + ["ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Container check stdout: {check_result.stdout}")
        
        if container_name in check_result.stdout:
            # Container exists, check if running
            ps_result = subprocess.run(
                self.docker_cmd + ["ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            if container_name in ps_result.stdout:
                logger.info(f"Container {container_name} already running")
                return True
            else:
                # Start existing container
                logger.info(f"Starting existing container {container_name}")
                start_result = subprocess.run(
                    self.docker_cmd + ["start", container_name],
                    capture_output=True,
                    text=True
                )
                if start_result.returncode == 0:
                    # Wait for health check
                    return self._wait_for_container_health(model)
                else:
                    logger.error(f"Failed to start container: {start_result.stderr}")
                    return False
        
        # Check if config file exists (resolve relative to workspace)
        config_file = WORKSPACE_DIR / model.config_path if model.config_path else None
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Config path: {config_file}")
        if not config_file or not config_file.exists():
            logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] ✗ FAIL: Config file not found: {config_file}")
            return False
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] ✓ Config file exists")
        
        # Check if model directory exists (resolve relative to workspace)
        model_dir = WORKSPACE_DIR / model.model_path
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Model dir: {model_dir}")
        if not model_dir.exists():
            logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] ✗ FAIL: Model directory not found: {model_dir}")
            return False
        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] ✓ Model directory exists")
        
        # Create required directories (matches start_vllm.sh)
        logs_dir = WORKSPACE_DIR / "logs"
        pids_dir = WORKSPACE_DIR / "pids"
        logs_dir.mkdir(exist_ok=True)
        pids_dir.mkdir(exist_ok=True)
        
        # Build docker run command with individual arguments (vLLM doesn't support --config)
        workspace_abs = WORKSPACE_DIR.resolve()

        # Read config and build command line arguments
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] ✗ FAIL: Error reading config file {config_file}: {e}")
            return False

        # Build command line arguments from config
        # Use local model path if available, otherwise HuggingFace name
        if model_dir.exists() and any(model_dir.glob("*.safetensors")):
            # Use local mounted path
            cmd_args = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", "/app/model",
                "--host", "0.0.0.0",
                "--port", str(model.port)
            ]
        else:
            # Use HuggingFace name
            cmd_args = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", config.get("model", model.model_path),
                "--host", "0.0.0.0",
                "--port", str(model.port)
            ]

        # Add optional config parameters
        if "tensor_parallel_size" in config:
            cmd_args.extend(["--tensor-parallel-size", str(config["tensor_parallel_size"])])
        if "gpu_memory_utilization" in config:
            cmd_args.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])
        if "max_model_len" in config:
            cmd_args.extend(["--max-model-len", str(config["max_model_len"])])
        if "max_concurrent_requests" in config:
            cmd_args.extend(["--max-num-seqs", str(config["max_concurrent_requests"])])

        if config.get("trust_remote_code"):
            cmd_args.extend(["--trust-remote-code"])
        if "quantization" in config:
            cmd_args.extend(["--quantization", str(config["quantization"])])
        if "max_num_seqs" in config:
            cmd_args.extend(["--max-num-seqs", str(config["max_num_seqs"])])
        if "kv_cache_dtype" in config:
            cmd_args.extend(["--kv-cache-dtype", str(config["kv_cache_dtype"])])

        # Tool calling configuration
        if config.get("enable_auto_tool_choice"):
            cmd_args.extend(["--enable-auto-tool-choice"])
        if "tool_call_parser" in config:
            cmd_args.extend(["--tool-call-parser", str(config["tool_call_parser"])])

        # Reasoning parser configuration
        if "reasoning_parser" in config:
            cmd_args.extend(["--reasoning-parser", str(config["reasoning_parser"])])
        if "reasoning_parser_plugin" in config:
            cmd_args.extend(["--reasoning-parser-plugin", f"/app/plugins/{config['reasoning_parser_plugin']}"]) 
        # Mount model directory for local loading
        model_dir_abs = model_dir.resolve()

        
        # Per-model container image override (falls back to default)
        container_image = config.get("container_image", VLLM_IMAGE)
        cmd = self.docker_cmd + [
            "run", "--user", "root", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
            "--gpus", "all",
            "--shm-size", "16GB",
            "-p", f"{model.port}:{model.port}",
            "-v", f"{model_dir_abs}:/app/model:ro",
            "-v", f"{workspace_abs}/logs:/app/logs",
            "-v", f"{workspace_abs}/pids:/app/pids",
            "-v", f"{workspace_abs}/proxy:/app/plugins:ro",
            "--env", "PYTHONPATH=/app",
            "--env", "NVIDIA_DISABLE_REQUIRE=true",
            "--env", "VLLM_USE_FLASHINFER_MOE_FP4=1",
            "--env", "VLLM_FLASHINFER_MOE_BACKEND=throughput",
            container_image
        ] + cmd_args
        
        try:
            logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Running docker command: {' '.join(cmd[:5])}...")
            # Start container
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(WORKSPACE_DIR)
            )
            
            logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Docker returncode: {result.returncode}")
            if result.returncode != 0:
                logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] ✗ FAIL: Docker command failed")
                logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] stderr: {result.stderr}")
                logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] stdout: {result.stdout}")
                return False
            
            container_id = result.stdout.strip()
            logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] ✓ Container started: {container_name} (ID: {container_id[:12]}...)")
            
            # Wait for server to be ready (non-blocking - returns False if not ready yet)
            logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Starting health check wait")
            health_result = self._wait_for_container_health(model)
            logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Health check result: {health_result}")
            
            # If health check failed but container is running, return True anyway
            # Background health checker will update status when ready
            if not health_result:
                # Verify container is actually running
                try:
                    ps_result = subprocess.run(
                        self.docker_cmd + ["ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                        capture_output=True,
                        text=True,
                        timeout=3
                    )
                    if container_name in ps_result.stdout:
                        logger.info(f"[DIAG] [_LOAD_VLLM_MODEL] Container is running but not ready yet - background health checker will update")
                        return True  # Container started successfully, health check will complete in background
                except:
                    pass
            
            return health_result
            
        except Exception as e:
            logger.error(f"[DIAG] [_LOAD_VLLM_MODEL] ✗ EXCEPTION: {e}", exc_info=True)
            return False
    
    def _wait_for_container_health(self, model: ModelInfo, timeout: int = None) -> bool:
        """Wait for container health endpoint to be ready - backend-specific timeouts"""
        import requests
        import time as time_module

        # Set backend-specific timeouts (per user requirements)
        if timeout is None:
            if model.backend == ModelBackend.VLLM:
                timeout = 180  # 3 minutes for vLLM (large models need more time)
            elif model.backend == ModelBackend.LLAMACPP:
                timeout = 90   # 1.5 minutes for llama.cpp
            else:
                timeout = 60   # 1 minute default

        logger.info(f"[DIAG] [_WAIT_HEALTH] Waiting up to {timeout}s for {model.container_name} health check")

        # First check if container is running
        try:
            check_result = subprocess.run(
                self.docker_cmd + ["ps", "--filter", f"name={model.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "Up" not in check_result.stdout:
                logger.warning(f"Container {model.container_name} is not running")
                return False
        except Exception as e:
            logger.error(f"Error checking container status: {e}")
            return False

        # Health check with backend-specific timeout
        health_endpoints = [
            f"http://localhost:{model.port}/v1/models",
            f"http://localhost:{model.port}/health",
            f"http://localhost:{model.port}/healthcheck"
        ]

        # Check every 5 seconds until timeout
        start_time = time_module.time()
        check_count = 0
        while (time_module.time() - start_time) < timeout:
            check_count += 1
            for health_url in health_endpoints:
                try:
                    response = requests.get(health_url, timeout=1)
                    if response.status_code == 200:
                        elapsed = int(time_module.time() - start_time)
                        logger.info(f"[DIAG] [_WAIT_HEALTH] ✓ Container {model.container_name} ready after {elapsed}s")
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                except Exception as e:
                    logger.debug(f"Health check error on {health_url}: {e}")

            # Sleep between checks (but not on last iteration)
            if (time_module.time() - start_time) + 5 < timeout:
                time_module.sleep(5)
        
        # Container exists but not ready yet - mark as loading, will check on first request
        logger.info(f"Container {model.container_name} exists but not ready yet - will check on first request")
        return False  # Return False so status stays "loading"

    def _test_model_readiness(self, model: ModelInfo) -> bool:
        """Test model readiness with a simple prompt - cluster-grade validation"""
        import requests
        import time

        logger.info(f"[DIAG] [_TEST_READINESS] Testing readiness for {model.model_id} on port {model.port}")

        # Simple test prompt - be more lenient for large models
        # Use correct model ID for different backends
        if model.backend == ModelBackend.VLLM:
            test_model_id = "/app/model"  # vLLM uses mounted path as model ID
        elif model.backend == ModelBackend.LLAMACPP:
            # llama.cpp uses the full path to the GGUF file as model ID
            from pathlib import Path
            workspace_dir = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))
            model_path = Path(workspace_dir) / model.model_path
            # For GGUF models, find the actual .gguf file if path is a directory
            if model_path.is_dir():
                gguf_files = list(model_path.glob("*.gguf"))
                if gguf_files:
                    # Use the same logic as in model_manager
                    preferred_order = ["Q8_0", "Q6_K", "Q4_K_M", "Q4_K"]
                    for pref in preferred_order:
                        for gguf_file in gguf_files:
                            if pref in gguf_file.name:
                                model_path = gguf_file
                                break
                        if model_path.suffix == ".gguf":
                            break
                    if model_path.suffix != ".gguf":
                        model_path = gguf_files[0]
            test_model_id = str(model_path)
        else:
            test_model_id = model.model_id

        test_payload = {
            "model": test_model_id,
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 20,  # Increased for reasoning models (DeepSeek-R1)
            "temperature": 0
        }
        
        logger.info(f"[DIAG] [_TEST_READINESS] Test payload model ID: {test_model_id}")

        # Allow slower backends to come up (vLLM can take longer)
        if model.backend == ModelBackend.VLLM:
            max_attempts = 6
            base_sleep = 15
        else:
            max_attempts = 4
            base_sleep = 10

        for attempt in range(max_attempts):
            try:
                # Test the model via its direct port (not through proxy)
                url = f"http://localhost:{model.port}/v1/chat/completions"
                start_time = time.time()
                
                logger.info(f"[DIAG] [_TEST_READINESS] Sending request to {url}")
                response = requests.post(url, json=test_payload, timeout=60)
                logger.info(f"[DIAG] [_TEST_READINESS] Response status: {response.status_code}, body: {response.text[:200]}")

                if response.status_code == 200:
                    result = response.json()
                    elapsed = time.time() - start_time

                    # Check if we got a valid response with content
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        # Handle None values from API (vLLM may return None instead of empty string)
                        content_raw = message.get("content")
                        reasoning_raw = message.get("reasoning_content")
                        content = (content_raw or "").strip()
                        reasoning = (reasoning_raw or "").strip()

                        # Accept either content OR reasoning (for DeepSeek-R1/Qwen3 style models)
                        # For reasoning models, accept partial responses during startup
                        has_content = content and len(content) > 0
                        has_reasoning = reasoning and len(reasoning) > 0
                        is_reasoning_model_partial = (model.backend == ModelBackend.VLLM and
                                                    len(content) >= 5)  # Accept "<think>" prefix

                        if has_content or has_reasoning or is_reasoning_model_partial:
                            logger.info(f"[DIAG] [_TEST_READINESS] ✓ {model.model_id} ready in {elapsed:.1f}s (attempt {attempt+1})")
                            logger.info(f"[DIAG] [_TEST_READINESS] Response: content='{content[:50]}...', reasoning='{reasoning[:50]}...'")
                            return True
                        else:
                            logger.warning(f"[DIAG] [_TEST_READINESS] Empty response on attempt {attempt+1}: content='{content}', reasoning='{reasoning}'")
                    else:
                        logger.warning(f"[DIAG] [_TEST_READINESS] Invalid response format on attempt {attempt+1}: {result}")
                else:
                    logger.warning(f"[DIAG] [_TEST_READINESS] HTTP {response.status_code} on attempt {attempt+1}")

                # Wait before retry
                if attempt < max_attempts - 1:
                    time.sleep(base_sleep)

            except requests.exceptions.ConnectionError:
                logger.warning(f"[DIAG] [_TEST_READINESS] Connection error on attempt {attempt+1} - model may still be starting")
                if attempt < max_attempts - 1:
                    time.sleep(base_sleep)
            except requests.exceptions.Timeout:
                logger.warning(f"[DIAG] [_TEST_READINESS] Timeout on attempt {attempt+1} - model may still be loading")
                if attempt < max_attempts - 1:
                    time.sleep(base_sleep + 5)
            except Exception as e:
                logger.error(f"[DIAG] [_TEST_READINESS] Exception on attempt {attempt+1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(base_sleep)

        # If we get here, all attempts failed
        logger.error(f"[DIAG] [_TEST_READINESS] ✗ All readiness test attempts failed for {model.model_id}")
        return False

    def _load_llamacpp_model(self, model: ModelInfo) -> bool:
        """Load llama.cpp model using native ARM64 binary"""
        from pathlib import Path
        import os

        # Path to native llama-server binary (relative to repo root)
        LLAMA_SERVER_BINARY = Path(REPO_ROOT) / "llama.cpp" / "build" / "bin" / "llama-server"

        # Use absolute path - work from repo root where vllm_models/ exists
        WORKSPACE_DIR = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))

        # Per-model log file to capture server output for readiness debugging
        logs_dir = WORKSPACE_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"llamacpp-{model.model_id}.log"

        # Process name for tracking
        process_name = f"llama-server-{model.port}"
        model.container_name = process_name

        # Check if process is already running on this port
        try:
            # Check if port is in use
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', model.port))
            sock.close()
            if result == 0:
                logger.info(f"llama-server already running on port {model.port}")
                return True
        except Exception as e:
            logger.warning(f"Error checking port {model.port}: {e}")

        # Model path - handle different model formats
        model_path = Path(WORKSPACE_DIR) / model.model_path

        # For GGUF models, find the actual .gguf file if path is a directory
        if model_path.is_dir():
            gguf_files = list(model_path.glob("*.gguf"))
            if gguf_files:
                # Prefer Q8_0, then Q6_K, then any GGUF
                preferred_order = ["Q8_0", "Q6_K", "Q4_K_M", "Q4_K"]
                for pref in preferred_order:
                    for gguf_file in gguf_files:
                        if pref in gguf_file.name:
                            model_path = gguf_file
                            break
                    if model_path.suffix == ".gguf":
                        break
                # If no preferred found, use first GGUF
                if model_path.suffix != ".gguf":
                    model_path = gguf_files[0]

        logger.info(f"Starting native llama-server for {model.model_id} on port {model.port}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Binary: {LLAMA_SERVER_BINARY}")
        logger.info(f"Log file: {log_file}")

        # Start the native binary
        cmd = [
            str(LLAMA_SERVER_BINARY),
            "-m", str(model_path),
            "--port", str(model.port),
            "--host", "127.0.0.1",
            "--n-gpu-layers", str(model.gpu_layers),
            "--ctx-size", str(model.context_size),
            "--threads", str(model.threads)
        ]

        try:
            # Set CUDA library paths to avoid missing libmtmd.so (CUDA 13 toolchain)
            env = os.environ.copy()
            cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
            ld_paths = [
                env.get("LD_LIBRARY_PATH", ""),
                f"{cuda_home}/lib64",
                f"{cuda_home}/targets/aarch64-linux/lib",
                f"{cuda_home}/targets/x86_64-linux/lib",
                str(LLAMA_SERVER_BINARY.parent),  # include llama.cpp build/bin for libmtmd.so
            ]
            env["LD_LIBRARY_PATH"] = ":".join([p for p in ld_paths if p])

            # Start process in background, capture stdout/stderr to log file
            with open(log_file, "a", encoding="utf-8") as lf:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(WORKSPACE_DIR),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )

            # Store process info
            model.process = process
            logger.info(f"Started llama-server process (PID: {process.pid}) for {model.model_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to start llama-server for {model.model_id}: {e}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model (stop Docker container)
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        model = self.models[model_id]
        
        if model.status == "unloaded":
            logger.info(f"Model {model_id} already unloaded")
            return True
        
        model.status = "unloading"
        logger.info(f"Unloading model {model_id}")
        
        try:
            success = False

            # Handle vLLM models (Docker containers)
            if model.backend == ModelBackend.VLLM and model.container_name:
                container_name = model.container_name

                # Stop container
                stop_result = subprocess.run(
                    self.docker_cmd + ["stop", container_name],
                    capture_output=True,
                    text=True
                )

                if stop_result.returncode != 0:
                    logger.warning(f"Failed to stop container {container_name}: {stop_result.stderr}")

                # Remove container
                rm_result = subprocess.run(
                    self.docker_cmd + ["rm", container_name],
                    capture_output=True,
                    text=True
                )

                if rm_result.returncode != 0:
                    logger.warning(f"Failed to remove container {container_name}: {rm_result.stderr}")

                success = True

            # Handle llama.cpp models (native processes)
            elif model.backend == ModelBackend.LLAMACPP and model.process:
                process = model.process

                # Terminate the process
                try:
                    process.terminate()
                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        process.wait(timeout=10)
                        logger.info(f"Process {process.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        process.kill()
                        process.wait()
                        logger.warning(f"Process {process.pid} force killed")
                except Exception as e:
                    logger.warning(f"Error terminating process {process.pid}: {e}")

                success = True

            if success:
                model.container_name = None
                model.process = None
                model.status = "unloaded"
                logger.info(f"Model {model_id} unloaded successfully")
                return True
            else:
                logger.warning(f"No valid container/process found for model {model_id}")
                model.status = "unloaded"
                return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            model.status = "unloaded"
            return False
    
    def get_least_recently_used_model(self) -> Optional[str]:
        """
        Get model ID of least recently used loaded model
        
        Returns:
            Model ID or None if no loaded models
        """
        loaded_models = [
            (model_id, model)
            for model_id, model in self.models.items()
            if model.status == "loaded"
        ]
        
        if not loaded_models:
            return None
        
        # Sort by last access time (oldest first)
        loaded_models.sort(key=lambda x: x[1].last_access_time)
        return loaded_models[0][0]
    
    def update_access_time(self, model_id: str):
        """Update last access time for a model"""
        if model_id in self.models:
            self.models[model_id].last_access_time = time.time()
            self.models[model_id].request_count += 1
    
    def get_model_status(self, model_id: str) -> Optional[Dict]:
        """Get status information for a model"""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        return {
            "model_id": model.model_id,
            "backend": model.backend.value,
            "port": model.port,
            "status": model.status,
            "last_access_time": model.last_access_time,
            "request_count": model.request_count
        }
    
    def list_models(self) -> List[Dict]:
        """List all models with their status"""
        return [
            self.get_model_status(model_id)
            for model_id in self.models.keys()
        ]

