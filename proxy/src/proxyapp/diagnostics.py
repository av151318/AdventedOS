"""
Diagnostic Utilities
Provides comprehensive health checks and diagnostics for debugging
"""

import subprocess
import requests
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class Diagnostics:
    """Comprehensive diagnostic checks"""
    
    @staticmethod
    def check_docker(docker_cmd: List[str]) -> Tuple[bool, str]:
        """Check Docker availability"""
        try:
            result = subprocess.run(
                docker_cmd + ["ps"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "✓ Docker available"
            else:
                return False, f"✗ Docker failed: {result.stderr}"
        except Exception as e:
            return False, f"✗ Docker check failed: {e}"
    
    @staticmethod
    def check_container_running(docker_cmd: List[str], container_name: str) -> Tuple[bool, str]:
        """Check if container is running"""
        try:
            result = subprocess.run(
                docker_cmd + ["ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if container_name in result.stdout:
                status_result = subprocess.run(
                    docker_cmd + ["ps", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                status = status_result.stdout.strip()
                return True, f"✓ Container {container_name} running: {status}"
            else:
                return False, f"✗ Container {container_name} not running"
        except Exception as e:
            return False, f"✗ Container check failed: {e}"
    
    @staticmethod
    def check_port_available(port: int) -> Tuple[bool, str]:
        """Check if port is available/listening"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True, f"✓ Port {port} is listening"
            else:
                return False, f"✗ Port {port} not listening"
        except Exception as e:
            return False, f"✗ Port check failed: {e}"
    
    @staticmethod
    def check_http_endpoint(url: str, timeout: int = 2) -> Tuple[bool, str]:
        """Check if HTTP endpoint responds"""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, f"✓ {url} responding (200)"
            else:
                return False, f"✗ {url} returned {response.status_code}"
        except requests.exceptions.Timeout:
            return False, f"✗ {url} timeout after {timeout}s"
        except requests.exceptions.ConnectionError:
            return False, f"✗ {url} connection refused"
        except Exception as e:
            return False, f"✗ {url} error: {e}"
    
    @staticmethod
    def check_model_config(config_path: Path) -> Tuple[bool, str]:
        """Check if model config file exists"""
        if config_path.exists():
            return True, f"✓ Config file exists: {config_path}"
        else:
            return False, f"✗ Config file missing: {config_path}"
    
    @staticmethod
    def check_model_directory(model_path: Path) -> Tuple[bool, str]:
        """Check if model directory exists"""
        if model_path.exists():
            safetensors = list(model_path.glob("*.safetensors*"))
            if safetensors:
                return True, f"✓ Model directory exists with {len(safetensors)} safetensors files"
            else:
                return False, f"✗ Model directory exists but no safetensors files found"
        else:
            return False, f"✗ Model directory missing: {model_path}"
    
    @staticmethod
    def check_gguf_file(gguf_path: Path) -> Tuple[bool, str]:
        """Check if GGUF file exists"""
        if gguf_path.exists():
            size_gb = gguf_path.stat().st_size / (1024**3)
            return True, f"✓ GGUF file exists: {gguf_path.name} ({size_gb:.2f}GB)"
        else:
            return False, f"✗ GGUF file missing: {gguf_path}"
    
    @staticmethod
    def run_full_diagnostics(
        docker_cmd: List[str],
        model_info: Dict,
        workspace_dir: Path
    ) -> List[Tuple[str, bool, str]]:
        """Run full diagnostic suite for a model"""
        results = []
        
        # Check Docker
        docker_ok, docker_msg = Diagnostics.check_docker(docker_cmd)
        results.append(("Docker", docker_ok, docker_msg))
        
        if not docker_ok:
            return results  # Can't continue without Docker
        
        # Check container
        if model_info.get("container_name"):
            container_ok, container_msg = Diagnostics.check_container_running(
                docker_cmd, model_info["container_name"]
            )
            results.append(("Container", container_ok, container_msg))
        
        # Check port
        port = model_info.get("port")
        if port:
            port_ok, port_msg = Diagnostics.check_port_available(port)
            results.append(("Port", port_ok, port_msg))
            
            # Check HTTP endpoints
            endpoints = [
                f"http://localhost:{port}/v1/models",
                f"http://localhost:{port}/health",
                f"http://localhost:{port}/healthcheck"
            ]
            for endpoint in endpoints:
                endpoint_ok, endpoint_msg = Diagnostics.check_http_endpoint(endpoint, timeout=2)
                results.append((f"Endpoint {endpoint}", endpoint_ok, endpoint_msg))
                if endpoint_ok:
                    break  # Found working endpoint
        
        # Check config/model files
        if model_info.get("backend") == "vllm":
            config_path = workspace_dir / model_info.get("config_path", "")
            config_ok, config_msg = Diagnostics.check_model_config(config_path)
            results.append(("Config", config_ok, config_msg))
            
            model_path = workspace_dir / model_info.get("model_path", "")
            model_ok, model_msg = Diagnostics.check_model_directory(model_path)
            results.append(("Model Directory", model_ok, model_msg))
        elif model_info.get("backend") == "llamacpp":
            gguf_path = workspace_dir / model_info.get("model_path", "")
            gguf_ok, gguf_msg = Diagnostics.check_gguf_file(gguf_path)
            results.append(("GGUF File", gguf_ok, gguf_msg))
        
        return results



