#!/bin/bash

SERVICE_NAME="reward_services"
SCRIPT="reward_services.py"
LOGFILE="reward_service.log"

start_service() {
    echo "Starting TikTok Reward Calculation Service..."
    # nohup python3 $SCRIPT > $LOGFILE 2>&1 &
    nohup python3 $SCRIPT 2>&1 &
    echo $! > "${SERVICE_NAME}.pid"
    echo "Service started with PID $(cat ${SERVICE_NAME}.pid)"
}

stop_service() {
    if [ -f "${SERVICE_NAME}.pid" ]; then
        PID=$(cat "${SERVICE_NAME}.pid")
        echo "Stopping service with PID $PID..."
        kill $PID && rm -f "${SERVICE_NAME}.pid"
        echo "Service stopped."
    else
        echo "No PID file found. Service may not be running."
    fi
}

restart_service() {
    stop_service
    start_service
}

status_service() {
    if [ -f "${SERVICE_NAME}.pid" ]; then
        PID=$(cat "${SERVICE_NAME}.pid")
        if ps -p $PID > /dev/null; then
            echo "Service is running with PID $PID."
        else
            echo "PID file exists but process is not running."
        fi
    else
        echo "Service is not running."
    fi
}

case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac