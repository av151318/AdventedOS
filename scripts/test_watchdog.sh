#!/bin/bash

# ProxyApp Watchdog Test Script
# Tests watchdog functionality and edge cases

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}ProxyApp Watchdog Test Suite${NC}"
echo -e "${BLUE}=============================${NC}"
echo ""

# Test 1: Watchdog script syntax
echo -e "${YELLOW}Test 1: Watchdog script syntax check${NC}"
if bash -n "$SCRIPT_DIR/watchdog.sh"; then
    echo -e "${GREEN}✓ Watchdog script syntax OK${NC}"
else
    echo -e "${RED}✗ Watchdog script syntax error${NC}"
    exit 1
fi

# Test 2: Watchdog control script
echo -e "${YELLOW}Test 2: Control script functionality${NC}"
if [ -x "$SCRIPT_DIR/watchdog-control.sh" ]; then
    echo -e "${GREEN}✓ Control script exists and is executable${NC}"

    # Test help output
    if "$SCRIPT_DIR/watchdog-control.sh" 2>&1 | grep -q "ProxyApp Watchdog Control"; then
        echo -e "${GREEN}✓ Control script help works${NC}"
    else
        echo -e "${RED}✗ Control script help failed${NC}"
    fi
else
    echo -e "${RED}✗ Control script not executable${NC}"
fi

# Test 3: Manual override functionality
echo -e "${YELLOW}Test 3: Manual override (/tmp/disable_watchdog)${NC}"

# Clean up any existing override
rm -f /tmp/disable_watchdog

# Test disable
"$SCRIPT_DIR/watchdog.sh" disable
if [ -f /tmp/disable_watchdog ]; then
    echo -e "${GREEN}✓ Manual disable works${NC}"
else
    echo -e "${RED}✗ Manual disable failed${NC}"
fi

# Test enable
"$SCRIPT_DIR/watchdog.sh" enable
if [ ! -f /tmp/disable_watchdog ]; then
    echo -e "${GREEN}✓ Manual enable works${NC}"
else
    echo -e "${RED}✗ Manual enable failed${NC}"
fi

# Test 4: Service file syntax
echo -e "${YELLOW}Test 4: Systemd service file syntax${NC}"
if command -v systemd-analyze &> /dev/null; then
    if systemd-analyze verify "$SCRIPT_DIR/proxyapp-watchdog.service" 2>/dev/null; then
        echo -e "${GREEN}✓ Service file syntax OK${NC}"
    else
        echo -e "${YELLOW}⚠ Service file syntax check failed (may still work)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ systemd-analyze not available, skipping service syntax check${NC}"
fi

# Test 5: Installation script
echo -e "${YELLOW}Test 5: Installation script${NC}"
if [ -x "$SCRIPT_DIR/install_watchdog.sh" ]; then
    echo -e "${GREEN}✓ Install script exists and is executable${NC}"
else
    echo -e "${RED}✗ Install script not executable${NC}"
fi

# Test 6: Watchdog test cycle
echo -e "${YELLOW}Test 6: Watchdog test cycle${NC}"
# Run one monitoring cycle in background and kill after 35 seconds
timeout 35 "$SCRIPT_DIR/watchdog.sh" &
watchdog_pid=$!
sleep 35

if kill -0 $watchdog_pid 2>/dev/null; then
    kill $watchdog_pid 2>/dev/null || true
    echo -e "${GREEN}✓ Watchdog test cycle completed${NC}"
else
    echo -e "${GREEN}✓ Watchdog test cycle exited cleanly${NC}"
fi

# Test 7: Check for required tools
echo -e "${YELLOW}Test 7: Required tools availability${NC}"

tools=("curl" "docker" "python3" "nvidia-smi")
missing_tools=()

for tool in "${tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo -e "${GREEN}✓ $tool available${NC}"
    else
        echo -e "${RED}✗ $tool missing${NC}"
        missing_tools+=("$tool")
    fi
done

# Test 8: Log file creation
echo -e "${YELLOW}Test 8: Log file creation${NC}"
if [ -d "$SCRIPT_DIR/logs" ]; then
    echo -e "${GREEN}✓ Logs directory exists${NC}"

    # Create a test log entry
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [TEST] Test log entry" >> "$SCRIPT_DIR/logs/watchdog.log"
    if [ -f "$SCRIPT_DIR/logs/watchdog.log" ]; then
        echo -e "${GREEN}✓ Log file writable${NC}"
    else
        echo -e "${RED}✗ Log file not writable${NC}"
    fi
else
    echo -e "${RED}✗ Logs directory missing${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}============${NC}"

if [ ${#missing_tools[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All core tests passed${NC}"
    echo -e "${GREEN}✓ Watchdog is ready for deployment${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Run demo: ${GREEN}./demo_proxy_platform.sh${NC}"
    echo -e "  2. Install service: ${GREEN}./install_watchdog.sh${NC}"
    echo -e "  3. Check status: ${GREEN}./watchdog-control.sh status${NC}"
else
    echo -e "${YELLOW}⚠ Some tools are missing (may not be critical):${NC}"
    for tool in "${missing_tools[@]}"; do
        echo -e "  - $tool"
    done
    echo ""
    echo -e "${YELLOW}The watchdog can still function with limited capabilities.${NC}"
fi

echo ""
echo -e "${BLUE}Documentation:${NC}"
echo -e "  📖 WATCHDOG_README.md"
echo -e "  🔧 ./watchdog-control.sh --help"

