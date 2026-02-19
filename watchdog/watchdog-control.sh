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
