#!/bin/bash

# ProxyApp Watchdog Script - 24/7 Uptime Monitor
# Monitors entire demo_proxy_platform.sh stack (proxy + containers + WebUI)
# Handles timeouts, edge cases, and automatic restarts

set -euo pipefail

# Configuration
WATCHDOG_LOG="${WATCHDOG_LOG:-logs/watchdog.log}"
PROXY_PORT="${PROXY_PORT:-52415}"
WEBUI_PORT="${WEBUI_PORT:-3000}"
DISABLE_FILE="/tmp/disable_watchdog"
STATE_DIR="/tmp/proxyapp_watchdog"
LAST_RESTART_FILE="${STATE_DIR}/last_restart_epoch"
WATCHDOG_AUTOSTART="${WATCHDOG_AUTOSTART:-false}"  # if true, watchdog may restart/start the demo stack
MAX_LOAD_TIME=900        # 15 minutes for model loading
MAX_INFERENCE_TIME=900   # 15 minutes for inference requests
MAX_STREAMING_IDLE=300   # 5 minutes streaming inactivity
MONITOR_INTERVAL="${MONITOR_INTERVAL:-180}"  # Default 3 minutes; override env MONITOR_INTERVAL
RESTART_GRACE_PERIOD=60  # Wait 60s after restart before monitoring
STARTUP_GRACE_PERIOD=180 # Ignore WebUI/container health during initial startup window

# Tailscale connectivity configuration
TAILSCALE_DEVICES=("100.83.248.106" "100.73.12.33")  # Known tailnet device IPs
TAILSCALE_PING_TIMEOUT=3  # Seconds to wait for ping response

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Global state
LAST_ACTIVITY=$(date +%s)
RESTART_COUNT=0
LAST_RESTART=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PID=$$
SCRIPT_START_TS=$(date +%s)
DISABLED_LOGGED=false

# Persisted state init (prevents systemd restart loops)
mkdir -p "$STATE_DIR"
if [ -f "$LAST_RESTART_FILE" ]; then
    LAST_RESTART="$(cat "$LAST_RESTART_FILE" 2>/dev/null || echo "0")"
fi

# Tailscale connectivity state
TAILSCALE_LAST_UP=$(date +%s)
TAILSCALE_LAST_DOWN=$(date +%s)
TAILSCALE_STATE="unknown"  # "up", "down", "unknown"
TAILSCALE_STABLE_SINCE=$(date +%s)
TAILSCALE_MIN_STABLE_TIME=180  # 3 minutes continuous up before restart
TAILSCALE_RESTART_DELAY=300    # 5 minutes stable before restart trigger

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Color based on level
    local color="$NC"
    case "$level" in
        "ERROR") color="$RED" ;;
        "WARN") color="$YELLOW" ;;
        "INFO") color="$BLUE" ;;
        "SUCCESS") color="$GREEN" ;;
        "CRITICAL") color="$PURPLE" ;;
    esac

    # Log to file
    echo "[$timestamp] [$level] $message" >> "$WATCHDOG_LOG"

    # Log to console with color (if not daemon mode)
    if [ "${DAEMON_MODE:-false}" != "true" ]; then
        echo -e "${color}[$timestamp] [$level] $message${NC}"
    fi
}

# Check if watchdog is disabled
is_disabled() {
    if [ -f "$DISABLE_FILE" ]; then
        if [ "$DISABLED_LOGGED" != "true" ]; then
            log "INFO" "Watchdog disabled via $DISABLE_FILE"
            DISABLED_LOGGED=true
        fi
        return 0
    fi
    DISABLED_LOGGED=false
    return 1
}

# Get process info for logging
get_process_info() {
    local pid="$1"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        ps -p "$pid" -o pid,ppid,cmd --no-headers 2>/dev/null || echo "PID:$pid"
    else
        echo "PID:$pid (not running)"
    fi
}

# Check HTTP endpoint health
check_http_endpoint() {
    local url="$1"
    local timeout="${2:-10}"
    local expected_code="${3:-200}"

    local response
    response=$(timeout "$timeout" curl -s -w "%{http_code}" -o /dev/null "$url" 2>/dev/null || echo "000")

    if [ "$response" = "$expected_code" ]; then
        return 0
    else
        log "WARN" "HTTP check failed: $url returned $response (expected $expected_code)"
        return 1
    fi
}

# Check if demo process is running
check_demo_running() {
    # Check for main demo script process
    local demo_pid
    demo_pid=$(pgrep -f "demo_proxy_platform.sh" | head -1 || true)

    if [ -n "$demo_pid" ]; then
        log "INFO" "Demo script running: $(get_process_info "$demo_pid")"
        return 0
    fi

    log "WARN" "Demo script not found running"
    return 1
}

# Check proxy server health
check_proxy_health() {
    local proxy_url="http://localhost:$PROXY_PORT/healthcheck"

    if check_http_endpoint "$proxy_url" 5; then
        log "INFO" "Proxy health check passed"
        return 0
    else
        log "ERROR" "Proxy health check failed"
        return 1
    fi
}

# Check WebUI health
check_webui_health() {
    local webui_url="http://localhost:$WEBUI_PORT"

    if check_http_endpoint "$webui_url" 5; then
        log "INFO" "WebUI health check passed"
        return 0
    else
        log "ERROR" "WebUI health check failed"
        return 1
    fi
}

# Check model containers
check_model_containers() {
    local container_count
    container_count=$(docker ps --filter "name=vllm-" --filter "name=llamacpp-" --format "{{.Names}}" 2>/dev/null | wc -l)

    if [ "$container_count" -gt 0 ]; then
        log "INFO" "Found $container_count model containers running"
        return 0
    else
        log "ERROR" "No model containers found running"
        return 1
    fi
}

# Check Tailscale connectivity and handle restoration restart
check_tailscale_connectivity() {
    local current_time=$(date +%s)

    # Check if Tailscale daemon is running
    if ! pgrep -f "tailscaled" >/dev/null 2>&1; then
        if [ "$TAILSCALE_STATE" != "daemon_down" ]; then
            log "ERROR" "Tailscale daemon not running"
            TAILSCALE_STATE="daemon_down"
            TAILSCALE_LAST_DOWN=$current_time
        fi
        return 1
    fi

    # Get Tailscale IP to verify it's assigned
    local tailscale_ip
    tailscale_ip=$(tailscale ip -4 2>/dev/null | head -1 || echo "")
    if [ -z "$tailscale_ip" ]; then
        if [ "$TAILSCALE_STATE" != "no_ip" ]; then
            log "ERROR" "No Tailscale IP assigned"
            TAILSCALE_STATE="no_ip"
            TAILSCALE_LAST_DOWN=$current_time
        fi
        return 1
    fi

    # Check connectivity to known tailnet devices
    local connectivity_ok=false

    for device_ip in "${TAILSCALE_DEVICES[@]}"; do
        if ping -c 1 -W $TAILSCALE_PING_TIMEOUT "$device_ip" >/dev/null 2>&1; then
            connectivity_ok=true
            break
        fi
    done

    # Handle state transitions
    if [ "$connectivity_ok" = true ]; then
        # Connectivity is UP
        if [ "$TAILSCALE_STATE" != "up" ]; then
            # State change: down -> up
            log "SUCCESS" "Tailscale connectivity restored (can reach tailnet devices)"
            TAILSCALE_STATE="up"
            TAILSCALE_LAST_UP=$current_time
            TAILSCALE_STABLE_SINCE=$current_time
        fi

        # Check if stable enough for restart
        local stable_duration=$((current_time - TAILSCALE_STABLE_SINCE))
        if [ $stable_duration -ge $TAILSCALE_MIN_STABLE_TIME ]; then
            # Check if we were previously down and need to restart
            local time_since_down=$((current_time - TAILSCALE_LAST_DOWN))
            if [ $time_since_down -le $TAILSCALE_RESTART_DELAY ]; then
                # We were down recently and now stable - trigger restart
                log "CRITICAL" "Tailscale connectivity stable after $stable_duration seconds - restarting services"
                return 2  # Special return code for connectivity-restoration restart
            fi
        fi

        log "INFO" "Tailscale connectivity OK (stable for ${stable_duration}s)"
        return 0
    else
        # Connectivity is DOWN
        if [ "$TAILSCALE_STATE" != "down" ]; then
            # State change: up -> down
            log "CRITICAL" "Tailscale connectivity lost - cannot reach tailnet devices"
            TAILSCALE_STATE="down"
            TAILSCALE_LAST_DOWN=$current_time
        fi
        return 1
    fi
}

# Check for stuck requests (basic proxy request monitoring)
check_stuck_requests() {
    # This is a simplified check - in production you'd want more sophisticated monitoring
    # Check if there are any long-running processes that might be stuck

    # Try to get detailed status from proxy watchdog endpoint
    local watchdog_url="http://localhost:$PROXY_PORT/watchdog/status"
    local response
    response=$(timeout 10 curl -s "$watchdog_url" 2>/dev/null || echo "")

    if [ -n "$response" ]; then
        # Parse JSON response for stuck requests
        local long_running_count
        long_running_count=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    long_running = data.get('active_requests', {}).get('long_running', [])
    print(len(long_running))
    if long_running:
        for req in long_running:
            print(f\"{req['request_id']}:{req['duration_seconds']:.0f}s\", end=' ')
except:
    print('0')
" 2>/dev/null || echo "0")

        if [ "$long_running_count" -gt 0 ]; then
            local request_details
            request_details=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    long_running = data.get('active_requests', {}).get('long_running', [])
    for req in long_running:
        print(f\"{req['request_id']} ({req['duration_human']})\", end=' ')
except:
    pass
" 2>/dev/null || echo "")

            log "CRITICAL" "Found $long_running_count requests running longer than 15 minutes: $request_details"
            return 1
        else
            log "INFO" "No long-running requests detected via watchdog endpoint"
        fi
    else
        # Fallback to basic process check if watchdog endpoint unavailable
        log "WARN" "Watchdog status endpoint unavailable, using fallback process check"
        local long_running
    long_running=$(ps aux | grep -E "(python.*proxyapp|llama-server)" | grep -v grep | awk '$9 > 0.1 {print $2 ":" $11}' || true)

        if [ -n "$long_running" ]; then
            log "INFO" "Found potentially long-running processes: $long_running"
        fi
    fi

    return 0
}

# Monitor inference timeouts (simplified approach)
monitor_inference_timeouts() {
    # This would need integration with the proxy to track active requests
    # For MVP, we'll use a simpler approach: check for processes that have been running too long

    local now=$(date +%s)

    # Check for any python processes running longer than MAX_INFERENCE_TIME
    local old_processes
    old_processes=$(ps -eo pid,etimes,cmd | grep -E "python.*proxyapp" | grep -v grep | awk -v max="$MAX_INFERENCE_TIME" '$2 > max {print $1 ":" $2 ":" $3}' || true)

    if [ -n "$old_processes" ]; then
        log "CRITICAL" "Found processes running longer than ${MAX_INFERENCE_TIME}s: $old_processes"
        log "CRITICAL" "Triggering restart due to suspected stuck inference"
        return 1
    fi

    return 0
}

# Restart the demo stack
restart_demo_stack() {
    local now=$(date +%s)

    # Check restart frequency to avoid restart loops
    local time_since_last=$((now - LAST_RESTART))
    if [ $time_since_last -lt $RESTART_GRACE_PERIOD ]; then
        # IMPORTANT: Never exit here. Exiting causes systemd to restart us instantly and we can
        # end up in a destructive restart loop that kills the demo during normal startup.
        log "ERROR" "Restart suppressed (cooldown): ${time_since_last}s < ${RESTART_GRACE_PERIOD}s"
        return 0
    fi

    log "CRITICAL" "RESTARTING DEMO STACK - Reason: $1"
    RESTART_COUNT=$((RESTART_COUNT + 1))
    LAST_RESTART=$now
    echo "$LAST_RESTART" > "$LAST_RESTART_FILE" 2>/dev/null || true

    # Kill existing processes
    log "INFO" "Stopping existing demo processes..."
    pkill -f "demo_proxy_platform.sh" || true
    pkill -f "start_proxy.sh" || true
    pkill -f "llama-server" || true
    pkill -f "python.*proxyapp" || true

    # Stop containers
    docker stop $(docker ps -q --filter "name=vllm-") 2>/dev/null || true
    docker stop $(docker ps -q --filter "name=llamacpp-") 2>/dev/null || true
    docker stop open-webui 2>/dev/null || true

    # Clean up containers
    docker rm $(docker ps -aq --filter "name=vllm-") 2>/dev/null || true
    docker rm $(docker ps -aq --filter "name=llamacpp-") 2>/dev/null || true

    # Wait for cleanup
    sleep 5

    # Start fresh demo stack
    log "INFO" "Starting fresh demo stack..."
    cd "$SCRIPT_DIR"
    nohup ./demo_proxy_platform.sh > logs/demo_restart.log 2>&1 &
    local new_pid=$!

    log "SUCCESS" "Demo stack restart initiated (PID: $new_pid, restart #$RESTART_COUNT)"

    # Wait for startup
    sleep $RESTART_GRACE_PERIOD
}

# Main monitoring loop
main_monitoring_loop() {
    log "INFO" "Watchdog monitoring started (PID: $SCRIPT_PID)"

    while true; do
        # Check if disabled
        if is_disabled; then
            sleep $MONITOR_INTERVAL
            continue
        fi

        local now
        now=$(date +%s)

        # During initial startup, do not restart the stack just because WebUI
        # hasn't come up yet (image pull / init can take time).
        local in_startup_grace=false
        if [ $((now - SCRIPT_START_TS)) -lt $STARTUP_GRACE_PERIOD ]; then
            in_startup_grace=true
        fi

        # Also skip aggressive checks immediately after a restart attempt.
        if [ $((now - LAST_RESTART)) -lt $RESTART_GRACE_PERIOD ]; then
            sleep $MONITOR_INTERVAL
            continue
        fi

        local all_healthy=true
        local failure_reason=""
        local connectivity_restart_needed=false

        # Check 0: Tailscale connectivity (special handling)
        local tailscale_result
        # Capture only the exit code, suppress stdout logging during checks
        tailscale_result=$(check_tailscale_connectivity >/dev/null 2>&1; echo $?)
        if [ "$tailscale_result" -eq 2 ]; then
            # Special case: connectivity restored and stable - restart needed
            connectivity_restart_needed=true
            failure_reason="${failure_reason}Tailscale connectivity restored and stable. "
            log "CRITICAL" "Tailscale connectivity restored and stable - restarting services"
        elif [ "$tailscale_result" -eq 1 ]; then
            # Connectivity down - just log, don't restart
            log "WARN" "Tailscale connectivity issues detected - monitoring continues"
        fi

        # Check 1: Demo script running
        local demo_running=true
        if ! check_demo_running; then
            demo_running=false
            all_healthy=false
            failure_reason="${failure_reason}Demo script not running. "
        fi

        # Check 2: Proxy health
        if ! check_proxy_health; then
            all_healthy=false
            failure_reason="${failure_reason}Proxy unhealthy. "
        fi

        # Check 3: WebUI health
        if [ "$in_startup_grace" = true ]; then
            log "INFO" "Startup grace active (${STARTUP_GRACE_PERIOD}s) - skipping WebUI health enforcement"
        else
            if ! check_webui_health; then
                all_healthy=false
                failure_reason="${failure_reason}WebUI unhealthy. "
            fi
        fi

        # Check 4: Model containers
        if [ "$in_startup_grace" = true ]; then
            log "INFO" "Startup grace active (${STARTUP_GRACE_PERIOD}s) - skipping model container enforcement"
        else
            if ! check_model_containers; then
                all_healthy=false
                failure_reason="${failure_reason}No model containers. "
            fi
        fi

        # Check 5: Stuck requests/inference timeouts
        if ! monitor_inference_timeouts; then
            all_healthy=false
            failure_reason="${failure_reason}Stuck inference detected. "
        fi

        # Restart if local health checks failed OR connectivity restoration restart needed
        if [ "$all_healthy" != "true" ] || [ "$connectivity_restart_needed" = true ]; then
            # Safety: never kill/restart the stack unless explicitly enabled.
            if [ "$WATCHDOG_AUTOSTART" != "true" ]; then
                log "WARN" "Health check failed but WATCHDOG_AUTOSTART=false; not restarting. Reason: $failure_reason"
            else
                # Only restart if the demo is/was running, or if connectivity restoration requires it.
                # This avoids surprising autostarts on boot.
                if [ "$demo_running" = true ] || [ "$connectivity_restart_needed" = true ]; then
                    restart_demo_stack "$failure_reason"
                else
                    log "WARN" "Demo not running; WATCHDOG_AUTOSTART=true but skipping autostart to avoid surprise start"
                fi
            fi
        else
            LAST_ACTIVITY=$(date +%s)
            log "INFO" "All health checks passed"
        fi

        # Sleep before next check
        sleep $MONITOR_INTERVAL
    done
}

# Handle signals
cleanup() {
    log "INFO" "Watchdog shutting down (PID: $SCRIPT_PID)"
    exit 0
}

trap cleanup INT TERM

# Create log directory
mkdir -p "$(dirname "$WATCHDOG_LOG")"

# Main entry point
case "${1:-}" in
    "start")
        # Daemon mode
        export DAEMON_MODE=true
        log "INFO" "Starting watchdog in daemon mode"
        main_monitoring_loop
        ;;
    "stop")
        log "INFO" "Stopping watchdog"
        pkill -f "watchdog.sh" || true
        ;;
    "status")
        if pgrep -f "watchdog.sh" > /dev/null; then
            echo "Watchdog is running"
            exit 0
        else
            echo "Watchdog is not running"
            exit 1
        fi
        ;;
    "disable")
        touch "$DISABLE_FILE"
        log "INFO" "Watchdog disabled via $DISABLE_FILE"
        echo "Watchdog disabled. Create $DISABLE_FILE to re-enable."
        ;;
    "enable")
        rm -f "$DISABLE_FILE"
        log "INFO" "Watchdog re-enabled (removed $DISABLE_FILE)"
        echo "Watchdog re-enabled."
        ;;
    *)
        # Interactive mode
        echo "ProxyApp Watchdog - 24/7 Uptime Monitor"
        echo "Usage: $0 {start|stop|status|disable|enable}"
        echo ""
        echo "Commands:"
        echo "  start   - Start watchdog in background"
        echo "  stop    - Stop watchdog"
        echo "  status  - Check if watchdog is running"
        echo "  disable - Disable watchdog (/tmp/disable_watchdog)"
        echo "  enable  - Enable watchdog (remove /tmp/disable_watchdog)"
        echo ""
        echo "Configuration:"
        echo "  PROXY_PORT=$PROXY_PORT"
        echo "  WEBUI_PORT=$WEBUI_PORT"
        echo "  MAX_LOAD_TIME=${MAX_LOAD_TIME}s"
        echo "  MAX_INFERENCE_TIME=${MAX_INFERENCE_TIME}s"
        echo "  MAX_STREAMING_IDLE=${MAX_STREAMING_IDLE}s"
        echo "  MONITOR_INTERVAL=${MONITOR_INTERVAL}s"
        echo "  WATCHDOG_LOG=$WATCHDOG_LOG"
        echo "  TAILSCALE_DEVICES=${TAILSCALE_DEVICES[*]}"
        echo "  TAILSCALE_MIN_STABLE_TIME=${TAILSCALE_MIN_STABLE_TIME}s"
        echo "  TAILSCALE_RESTART_DELAY=${TAILSCALE_RESTART_DELAY}s"
        echo ""
        # Run one monitoring cycle in foreground for testing
        echo "Running one monitoring cycle..."
        main_monitoring_loop &
        sleep 35  # Run for one cycle + buffer
        kill %1 2>/dev/null || true
        ;;
esac
