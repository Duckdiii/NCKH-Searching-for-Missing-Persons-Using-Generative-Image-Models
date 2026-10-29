#!/bin/bash
tmux kill-session -t frontend 2>/dev/null && echo "Đã dừng frontend"
tmux kill-session -t backend 2>/dev/null && echo "Đã dừng backend"
true
