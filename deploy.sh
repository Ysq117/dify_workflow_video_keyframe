#!/bin/bash

set -e  # 遇到错误立即退出

# 尝试停止并删除名为 'keyframe-service' 的容器（推荐用名字而非 ID）
echo "🗑️ 步骤 1: 停止并删除旧容器（如果存在）"
docker stop keyframe-service 2>/dev/null || true
docker rm keyframe-service 2>/dev/null || true


# 删除旧镜像（按标签名，更安全）
echo "🧹 步骤 2: 删除旧镜像（如果存在）"
docker rmi keyframe:latest 2>/dev/null || true

echo "🏗️ 步骤 3: 构建新镜像"
docker build -t keyframe .

echo "🚀 步骤 4: 启动服务"
docker compose up -d

echo "✅ 部署完成！"
