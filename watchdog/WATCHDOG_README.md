# ProxyApp Watchdog - 24/7 Uptime Monitor

The ProxyApp watchdog provides automatic monitoring and restart capabilities for the entire AI platform stack.

## Features

- **24/7 Monitoring**: Continuous health checks of all components
- **Timeout Protection**: 15-minute max for model loads and inference requests
- **Streaming Protection**: 5-minute inactivity timeout for streaming responses
- **Full Stack Restart**: Restarts entire demo stack on any failure
- **Manual Override**: Simple file-based disable/enable control
- **Comprehensive Logging**: All events logged for debugging

## Installation

The watchdog is automatically installed with the main demo:

```bash
./demo_proxy_platform.sh
```

For manual installation:

```bash
./install_watchdog.sh
```

## Usage

### Automatic (Systemd Service)

The watchdog runs automatically as a systemd service:

```bash
# Check status
sudo systemctl status proxyapp-watchdog.service

# View logs
sudo systemctl journalctl -u proxyapp-watchdog.service -f

# Restart service
sudo systemctl restart proxyapp-watchdog.service
```

### Manual Control

Use the control script for manual operations:

```bash
# Start/stop/restart
./watchdog-control.sh start
./watchdog-control.sh stop
./watchdog-control.sh restart

# Check status
./watchdog-control.sh status

# View detailed logs
./watchdog-control.sh logs

# Test watchdog (runs one monitoring cycle)
./watchdog-control.sh test
```

### Manual Override

```bash
# Disable watchdog monitoring
./watchdog-control.sh disable
# or
touch /tmp/disable_watchdog

# Re-enable watchdog
./watchdog-control.sh enable
# or
rm -f /tmp/disable_watchdog
```

## Configuration

Edit `watchdog.sh` to modify timeouts and monitoring settings:

```bash
# Timeouts (in seconds)
MAX_LOAD_TIME=900        # 15 minutes for model loading
MAX_INFERENCE_TIME=900   # 15 minutes for inference requests
MAX_STREAMING_IDLE=300   # 5 minutes streaming inactivity
MONITOR_INTERVAL=30      # Check every 30 seconds
```

## Monitoring

### Health Checks

The watchdog monitors:

1. **Demo Script**: `demo_proxy_platform.sh` process running
2. **Proxy Server**: HTTP health check on port 52415
3. **WebUI**: HTTP check on port 3000
4. **Model Containers**: Docker containers running
5. **Stuck Requests**: Processes running longer than timeouts

### Log Files

- **Systemd Logs**: `journalctl -u proxyapp-watchdog.service`
- **Watchdog Logs**: `logs/watchdog.log`
- **Demo Logs**: `logs/demo_restart.log`

## Troubleshooting

### Watchdog Not Starting

```bash
# Check service status
sudo systemctl status proxyapp-watchdog.service

# Check logs
sudo systemctl journalctl -u proxyapp-watchdog.service -n 50

# Test manually
./watchdog.sh
```

### False Restarts

If watchdog is restarting too frequently:

```bash
# Temporarily disable
touch /tmp/disable_watchdog

# Adjust timeouts in watchdog.sh
# Restart service
sudo systemctl restart proxyapp-watchdog.service

# Re-enable
rm -f /tmp/disable_watchdog
```

### Restart Loops

If watchdog detects restart loops (restarts too close together):

1. Check logs for the cause
2. Fix underlying issue (network, Docker, etc.)
3. Manually restart: `./watchdog-control.sh restart`

## Architecture

```
┌─────────────────┐
│   Watchdog      │
│   Monitor       │
├─────────────────┤
│ • Health Checks │
│ • Timeout       │
│ • Auto-restart  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│  Demo Script    │ -> │   Components    │
│  (Full Stack)   │    │ • Proxy Server  │
├─────────────────┤    │ • WebUI         │
│ • Proxy         │    │ • Containers    │
│ • Models        │    │ • Processes     │
│ • WebUI         │    └─────────────────┘
└─────────────────┘
```

The watchdog monitors the entire stack and restarts everything if any component fails or times out.
