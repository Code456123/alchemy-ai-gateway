#!/usr/bin/env bash
set -e

PORT=${PORT:-8080}

export TERM=xterm-256color
export LANG=C.UTF-8

exec ttyd -W -p "$PORT" python -m frontend.cli.app