#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
LOG_DIR="$RUNTIME_DIR/logs"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
FRONTEND_PID_FILE="$RUNTIME_DIR/frontend.pid"

CONDA_ENV="${RAG_CONDA_ENV:-blue}"
BACKEND_HOST="${RAG_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${RAG_BACKEND_PORT:-8001}"
FRONTEND_HOST="${RAG_FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${RAG_FRONTEND_PORT:-5174}"
API_BASE="${VITE_API_BASE:-http://${BACKEND_HOST}:${BACKEND_PORT}/api}"

BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

mkdir -p "$LOG_DIR"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '错误：缺少命令 %s。\n' "$1" >&2
    exit 1
  fi
}

is_running() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

read_managed_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    tr -d '[:space:]' < "$pid_file"
  fi
}

clear_stale_pid() {
  local pid_file="$1"
  local pid
  pid="$(read_managed_pid "$pid_file")"
  if [[ -n "$pid" ]] && ! is_running "$pid"; then
    rm -f "$pid_file"
  fi
}

port_is_open() {
  local host="$1"
  local port="$2"
  (exec 3<>"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_for_url() {
  local url="$1"
  local attempts="${2:-40}"
  local index
  for ((index = 1; index <= attempts; index++)); do
    if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

stop_process_group() {
  local pid="${1:-}"
  if is_running "$pid"; then
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  fi
}

require_command conda
require_command curl
require_command setsid

if [[ ! -f "$ROOT_DIR/backend/app/main.py" || ! -f "$ROOT_DIR/frontend/package.json" ]]; then
  printf '错误：项目目录不完整：%s\n' "$ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
  printf '错误：前端依赖尚未安装，请先在 blue 环境运行 npm install。\n' >&2
  exit 1
fi

if ! conda run -n "$CONDA_ENV" python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  printf '错误：Conda 环境 %s 不存在或后端依赖不完整。\n' "$CONDA_ENV" >&2
  exit 1
fi

clear_stale_pid "$BACKEND_PID_FILE"
clear_stale_pid "$FRONTEND_PID_FILE"

backend_started=0
frontend_started=0
backend_pid="$(read_managed_pid "$BACKEND_PID_FILE")"

if is_running "$backend_pid"; then
  printf '后端已由本项目管理，PID=%s。\n' "$backend_pid"
else
  if port_is_open "$BACKEND_HOST" "$BACKEND_PORT"; then
    printf '错误：后端端口 %s:%s 已被非本脚本管理的进程占用。\n' "$BACKEND_HOST" "$BACKEND_PORT" >&2
    printf '请先关闭旧服务，或设置 RAG_BACKEND_PORT 使用其他端口。\n' >&2
    exit 1
  fi

  : > "$BACKEND_LOG"
  (
    cd "$ROOT_DIR/backend"
    exec setsid conda run --no-capture-output -n "$CONDA_ENV" \
      uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
  ) >>"$BACKEND_LOG" 2>&1 &
  backend_pid=$!
  printf '%s\n' "$backend_pid" > "$BACKEND_PID_FILE"
  backend_started=1
  printf '正在启动后端，PID=%s...\n' "$backend_pid"
fi

if ! wait_for_url "$BACKEND_URL/api/health"; then
  printf '错误：后端未能通过健康检查。日志末尾如下：\n' >&2
  tail -n 30 "$BACKEND_LOG" >&2 || true
  if [[ "$backend_started" -eq 1 ]]; then
    stop_process_group "$backend_pid"
    rm -f "$BACKEND_PID_FILE"
  fi
  exit 1
fi

frontend_pid="$(read_managed_pid "$FRONTEND_PID_FILE")"
if is_running "$frontend_pid"; then
  printf '前端已由本项目管理，PID=%s。\n' "$frontend_pid"
else
  if port_is_open "$FRONTEND_HOST" "$FRONTEND_PORT"; then
    printf '错误：前端端口 %s:%s 已被非本脚本管理的进程占用。\n' "$FRONTEND_HOST" "$FRONTEND_PORT" >&2
    printf '请先关闭旧服务，或设置 RAG_FRONTEND_PORT 使用其他端口。\n' >&2
    if [[ "$backend_started" -eq 1 ]]; then
      stop_process_group "$backend_pid"
      rm -f "$BACKEND_PID_FILE"
    fi
    exit 1
  fi

  : > "$FRONTEND_LOG"
  (
    cd "$ROOT_DIR/frontend"
    exec setsid env VITE_API_BASE="$API_BASE" \
      conda run --no-capture-output -n "$CONDA_ENV" \
      npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
  ) >>"$FRONTEND_LOG" 2>&1 &
  frontend_pid=$!
  printf '%s\n' "$frontend_pid" > "$FRONTEND_PID_FILE"
  frontend_started=1
  printf '正在启动前端，PID=%s...\n' "$frontend_pid"
fi

if ! wait_for_url "$FRONTEND_URL"; then
  printf '错误：前端未能通过访问检查。日志末尾如下：\n' >&2
  tail -n 30 "$FRONTEND_LOG" >&2 || true
  if [[ "$frontend_started" -eq 1 ]]; then
    stop_process_group "$frontend_pid"
    rm -f "$FRONTEND_PID_FILE"
  fi
  if [[ "$backend_started" -eq 1 ]]; then
    stop_process_group "$backend_pid"
    rm -f "$BACKEND_PID_FILE"
  fi
  exit 1
fi

printf '\nRAG Demo 已启动。\n'
printf '前端工作台：%s\n' "$FRONTEND_URL"
printf 'API 文档：  %s/docs\n' "$BACKEND_URL"
printf '运行日志：  %s\n' "$LOG_DIR"
printf '关闭服务：  %s/stop.sh\n' "$ROOT_DIR"
