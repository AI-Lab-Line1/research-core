#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
FRONTEND_PID_FILE="$RUNTIME_DIR/frontend.pid"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"

is_running() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

stop_service() {
  local label="$1"
  local pid_file="$2"
  local pid=""
  local attempt

  if [[ ! -f "$pid_file" ]]; then
    printf '%s未由一键脚本启动，无需关闭。\n' "$label"
    return
  fi

  pid="$(tr -d '[:space:]' < "$pid_file")"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    printf '%s PID 文件无效，已清理。\n' "$label"
    rm -f "$pid_file"
    return
  fi

  if ! is_running "$pid"; then
    printf '%s进程已经退出，已清理 PID 文件。\n' "$label"
    rm -f "$pid_file"
    return
  fi

  printf '正在关闭%s，PID=%s...\n' "$label" "$pid"
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true

  for ((attempt = 1; attempt <= 20; attempt++)); do
    if ! is_running "$pid"; then
      break
    fi
    sleep 0.25
  done

  if is_running "$pid"; then
    printf '%s未在等待时间内退出，正在终止本项目进程组。\n' "$label"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi

  rm -f "$pid_file"
  printf '%s已关闭。\n' "$label"
}

stop_service "前端" "$FRONTEND_PID_FILE"
stop_service "后端" "$BACKEND_PID_FILE"

printf 'RAG Demo 一键启动的服务已全部关闭。日志保留在 %s/logs。\n' "$RUNTIME_DIR"
