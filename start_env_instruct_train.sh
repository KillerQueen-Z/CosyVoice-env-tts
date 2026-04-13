#!/usr/bin/env bash

# 一键后台启动 env-instruct 训练的脚本
# 使用方式：
#   bash start_env_instruct_train.sh
#
# 会自动：
#   1) cd 到仓库根目录
#   2) 给 run_env_instruct_train.sh 加执行权限
#   3) 用 nohup 在后台启动训练，日志写到 train_env_instruct.log

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

chmod +x run_env_instruct_train.sh

echo ">>> [$(date)] Launch training with nohup, log: $REPO_ROOT/train_env_instruct.log"
nohup "$REPO_ROOT/run_env_instruct_train.sh" > "$REPO_ROOT/train_env_instruct.log" 2>&1 &

echo ">>> PID: $!  已在后台运行"

