#!/bin/bash

# ProxyApp Watchdog Systemd Installation Script
# Installs and configures the watchdog service for 24/7 uptime monitoring

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_FILE_TEMPLATE="$SCRIPT_DIR/proxyapp-watchdog.service"
SERVICE_FILE_GEN="/tmp/proxyapp-watchdog.service"
WATCHDOG_SCRIPT="$SCRIPT_DIR/watchdog.sh"
SERVICE_USER="${SUDO_USER:-$USER}"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"

echo -e "${BLUE}ProxyApp Watchdog Installation${NC}"
echo -e "${BLUE}==============================${NC}"
echo ""

# Check comprehensive permissions like original demo script
if [ "$EUID" -eq 0 ]; then
    echo -e "${GREEN}Running as root - full permissions available${NC}"
    SUDO=""
else
    echo -e "${YELLOW}Running as user - checking sudo access${NC}"
    SUDO="sudo"

    # Test sudo access (non-interactive)
    if ! $SUDO -n true 2>/dev/null; then
        echo -e "${YELLOW}⚠ sudo access required for systemd service installation${NC}"
        echo -e "${YELLOW}You may be prompted for password, or run this script with sudo${NC}"
        echo -e "${YELLOW}Alternatively, run: sudo $0${NC}"

        # Try once to see if we can get sudo access
        if ! $SUDO true 2>/dev/null; then
            echo -e "${RED}❌ Cannot get sudo access. Systemd service installation requires root privileges.${NC}"
            echo -e "${YELLOW}Please run this script with sudo: sudo $0${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ sudo access confirmed${NC}"
    fi
fi

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}❌ Error: systemctl not found. This script requires systemd.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: docker not found. Docker is required.${NC}"
    exit 1
fi

# Check if watchdog script exists and is executable
if [ ! -f "$WATCHDOG_SCRIPT" ]; then
    echo -e "${RED}❌ Error: Watchdog script not found at $WATCHDOG_SCRIPT${NC}"
    exit 1
fi

if [ ! -x "$WATCHDOG_SCRIPT" ]; then
    echo -e "${YELLOW}Making watchdog script executable...${NC}"
    chmod +x "$WATCHDOG_SCRIPT"
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"

# Create logs directory
echo -e "${YELLOW}Creating logs directory...${NC}"
mkdir -p "$SCRIPT_DIR/logs"

# Check if service is already installed
if $SUDO systemctl list-units --all | grep -q "proxyapp-watchdog.service"; then
    echo -e "${YELLOW}Service already exists - updating...${NC}"
    $SUDO systemctl stop proxyapp-watchdog.service 2>/dev/null || true
else
    echo -e "${YELLOW}Installing systemd service...${NC}"
fi

# Generate systemd service with current paths
cat > "$SERVICE_FILE_GEN" <<EOF
[Unit]
Description=ProxyApp AI Platform Watchdog - 24/7 Uptime Monitor
After=network.target docker.service
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$REPO_ROOT
ExecStart=$WATCHDOG_SCRIPT start
ExecStop=$WATCHDOG_SCRIPT stop
Restart=always
RestartSec=10
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=$SERVICE_HOME
WatchdogSec=300
WatchdogSignal=SIGTERM
LimitNOFILE=65536
LimitNPROC=4096
StandardOutput=journal
StandardError=journal
SyslogIdentifier=proxyapp-watchdog

[Install]
WantedBy=multi-user.target
EOF

# Install/update systemd service
$SUDO cp "$SERVICE_FILE_GEN" /etc/systemd/system/proxyapp-watchdog.service
$SUDO systemctl daemon-reload

echo -e "${GREEN}✓ Systemd service installed${NC}"

# Enable service for auto-start on boot (only if not already enabled)
if ! $SUDO systemctl is-enabled proxyapp-watchdog.service &>/dev/null; then
    echo -e "${YELLOW}Enabling service for auto-start on boot...${NC}"
    $SUDO systemctl enable proxyapp-watchdog.service
    echo -e "${GREEN}✓ Service enabled for auto-start${NC}"
else
    echo -e "${GREEN}✓ Service already enabled for auto-start${NC}"
fi

# Test service installation (but don't fail if it doesn't start yet)
echo -e "${YELLOW}Testing service installation...${NC}"
if $SUDO systemctl list-units --all | grep -q "proxyapp-watchdog.service"; then
    echo -e "${GREEN}✓ Service installed successfully${NC}"
else
    echo -e "${RED}❌ Service installation failed${NC}"
    exit 1
fi

# Create manual control scripts
echo -e "${YELLOW}Creating manual control scripts...${NC}"

cat > "$SCRIPT_DIR/watchdog-control.sh" << 'EOF'
#!/bin/bash
# ProxyApp Watchdog Control Script

case "${1:-help}" in
    "start")
        sudo systemctl start proxyapp-watchdog.service
        echo "Watchdog started"
        ;;
    "stop")
        sudo systemctl stop proxyapp-watchdog.service
        echo "Watchdog stopped"
        ;;
    "restart")
        sudo systemctl restart proxyapp-watchdog.service
        echo "Watchdog restarted"
        ;;
    "status")
        sudo systemctl status proxyapp-watchdog.service
        ;;
    "disable")
        touch /tmp/disable_watchdog
        echo "Watchdog disabled (/tmp/disable_watchdog created)"
        ;;
    "enable")
        rm -f /tmp/disable_watchdog
        echo "Watchdog enabled (/tmp/disable_watchdog removed)"
        ;;
    "logs")
        journalctl -u proxyapp-watchdog.service -f
        ;;
    "test")
        cd "$(dirname "$0")"
        ./watchdog.sh
        ;;
    *)
        echo "ProxyApp Watchdog Control"
        echo "Usage: $0 {start|stop|restart|status|disable|enable|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start watchdog service"
        echo "  stop    - Stop watchdog service"
        echo "  restart - Restart watchdog service"
        echo "  status  - Show service status"
        echo "  disable - Disable watchdog monitoring"
        echo "  enable  - Enable watchdog monitoring"
        echo "  logs    - Show systemd logs"
        echo "  test    - Run watchdog test cycle"
        ;;
esac
EOF

chmod +x "$SCRIPT_DIR/watchdog-control.sh"

echo -e "${GREEN}✓ Manual control script created${NC}"

# Create README for watchdog
cat > "$SCRIPT_DIR/WATCHDOG_README.md" << 'EOF'
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
EOF

echo -e "${GREEN}✓ README created${NC}"

# Final instructions
echo ""
echo -e "${GREEN}🎉 Watchdog installation complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Start the demo: ${YELLOW}./demo_proxy_platform.sh${NC}"
echo -e "  2. Check watchdog: ${YELLOW}./watchdog-control.sh status${NC}"
echo -e "  3. View logs: ${YELLOW}./watchdog-control.sh logs${NC}"
echo ""
echo -e "${BLUE}Service will auto-start on boot${NC}"
echo -e "${BLUE}Manual override: ${YELLOW}touch /tmp/disable_watchdog${NC}"
echo ""
echo -e "${PURPLE}📖 Documentation: WATCHDOG_README.md${NC}"
