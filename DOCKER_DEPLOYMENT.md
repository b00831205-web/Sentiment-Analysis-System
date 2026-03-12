# Docker 部署指南

## 快速开始

### 前置条件

- Docker 18.06+ 或 Docker Desktop
- Docker Compose（选项，但推荐）
- 4GB RAM 最小（推荐 8GB）
- 2GB 磁盘空间

### 5 分钟快速部署

```bash
# 1. 从项目根目录构建镜像
docker build -t sentiment-api:v2 .

# 2. 运行容器
docker run -d \
  -p 8000:8000 \
  --name sentiment-api \
  sentiment-api:v2

# 3. 验证服务运行
curl http://localhost:8000/health

# 输出应该是：
# {"status": "ok", "timestamp": "2024-..."}
```

## 详细部署指南

### 方案 1：直接使用 Docker

#### 构建步骤

```bash
# 构建镜像（第一次运行需要 2-5 分钟）
docker build -t sentiment-api:v2 .

# 验证镜像
docker images | grep sentiment
```

#### 运行步骤

**基础运行：**
```bash
docker run -p 8000:8000 sentiment-api:v2
```

**生产运行（后台 + 数据卷）：**
```bash
# 创建数据卷存储日志和模型
docker volume create sentiment-logs
docker volume create sentiment-models

# 后台运行
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  -v sentiment-logs:/app/logs \
  -v sentiment-models:/app/v0:ro \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  sentiment-api:v2
```

#### 常用命令

```bash
# 查看容器状态
docker ps

# 查看容器日志
docker logs -f sentiment-api

# 进入容器 shell
docker exec -it sentiment-api /bin/bash

# 停止容器
docker stop sentiment-api

# 删除容器
docker rm sentiment-api

# 查看容器详细信息
docker inspect sentiment-api
```

### 方案 2：使用 Docker Compose（推荐）

#### 文件结构

```
project-root/
├── docker-compose.yml
├── Dockerfile
└── ... (其他项目文件)
```

#### 启动服务

```bash
# 启动服务（第一次会构建镜像）
docker-compose up

# 后台启动
docker-compose up -d

# 查看日志
docker-compose logs -f sentiment-api

# 重启服务
docker-compose restart sentiment-api

# 停止服务
docker-compose stop

# 停止并删除容器
docker-compose down

# 重建镜像（清除缓存）
docker-compose build --no-cache

# 显示当前状态
docker-compose ps
```

## 网络配置

### 端口映射

| 容器端口 | 主机端口 | 用途 |
|---------|---------|------|
| 8000 | 8000 | Flask API |

### 修改端口

**Docker 运行：**
```bash
docker run -p 9000:8000 sentiment-api:v2
# 现在访问 http://localhost:9000
```

**Docker Compose：**
编辑 `docker-compose.yml` 中的端口配置：
```yaml
ports:
  - "9000:8000"  # 改为 9000
```

### 网络模式

```bash
# 使用 host 网络（Linux 仅）
docker run --network host sentiment-api:v2

# 使用自定义网络
docker network create sentiment-net
docker run --network sentiment-net -p 8000:8000 sentiment-api:v2
```

## 数据持久化

### 挂载主机目录

```bash
docker run -d \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/v0:/app/v0:ro \
  -v $(pwd)/v1:/app/v1:ro \
  -p 8000:8000 \
  sentiment-api:v2
```

### 使用 Docker 数据卷

```bash
# 创建卷
docker volume create sentiment-logs

# 挂载卷
docker run -d \
  -v sentiment-logs:/app/logs \
  -p 8000:8000 \
  sentiment-api:v2

# 查看卷信息
docker volume inspect sentiment-logs
```

## API 使用示例

### 健康检查

```bash
curl http://localhost:8000/health
# 响应: {"status": "ok", "timestamp": "..."}
```

### 预测请求

```bash
# v0 模型预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This movie is amazing!",
    "model_kind": "v0"
  }'

# 响应示例：
# {
#   "label": 1,
#   "prob_pos": 0.95,
#   "model": "v0",
#   "model_path": "/app/v0/best_model_v0_20240110.joblib",
#   "model_name": "logistic_regression"
# }
```

### 使用 Python requests

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "text": "Great product!",
    "model_kind": "v0"
}

response = requests.post(url, json=payload)
print(response.json())
```

## 监控和日志

### 查看日志

```bash
# 最后 50 行日志
docker logs --tail 50 sentiment-api

# 持续查看日志
docker logs -f sentiment-api

# 查看特定时间段的日志
docker logs --since 10m sentiment-api
```

### 日志位置

容器内日志位置：`/app/logs/v2.log`

主机上挂载的日志（如果使用卷）：
```bash
# 查看卷中的日志
docker volume inspect sentiment-logs
# 查看 Mountpoint 路径
cat "Mountpoint/v2.log"
```

### 健康检查

```bash
# 检查容器健康状态
docker inspect --format='{{.State.Health.Status}}' sentiment-api

# 可能的状态：starting, healthy, unhealthy, none

# 查看健康检查历史
docker inspect sentiment-api | grep -A 5 Health
```

## 环境变量

### 配置示例

```bash
docker run -d \
  -e PORT=8000 \
  -e HOST=0.0.0.0 \
  -e LOG_LEVEL=DEBUG \
  -p 8000:8000 \
  sentiment-api:v2
```

## 模型和工件管理

### 预装模型

如果要在镜像中包含预建模型：

```bash
# 建立前确保 v0/v1 目录存在
cp -r v0 ./v0
cp -r v1 ./v1

# 构建镜像（会包含模型）
docker build -t sentiment-api:v2 .
```

### 运行时加载模型

```bash
docker run -d \
  -v /path/to/v0:/app/v0:ro \
  -v /path/to/v1:/app/v1:ro \
  -p 8000:8000 \
  sentiment-api:v2
```

## 性能优化

### 镜像大小

当前多阶段构建的镜像大小约 500MB。优化方法：

```dockerfile
# 使用更精简的基础镜像
FROM python:3.11-slim-alpine

# 移除不必要的包
RUN pip install --no-cache-dir ...
```

### 资源限制

```bash
docker run \
  -m 1g \
  --cpus="1" \
  -p 8000:8000 \
  sentiment-api:v2
```

## 故障排除

### 容器启动失败

```bash
# 查看详细错误
docker logs sentiment-api

# 常见原因：
# 1. 端口已被占用
docker ps | grep 8000
lsof -i :8000  # 查看占用进程

# 2. 内存不足
docker stats sentiment-api

# 3. 模型加载失败
# 确保 v0/v1 目录存在且有模型工件
```

### API 无响应

```bash
# 测试连接
curl -v http://localhost:8000/health

# 查看容器是否仍在运行
docker ps | grep sentiment-api

# 检查网络
docker network inspect bridge
```

### 高 CPU/内存使用

```bash
# 监控资源使用
docker stats sentiment-api

# 限制资源
docker update -m 512m --cpus="0.5" sentiment-api

# 重启容器
docker restart sentiment-api
```

## 升级和维护

### 升级镜像

```bash
# 拉取最新代码
git pull

# 重建镜像
docker build -t sentiment-api:v2 --no-cache .

# 停止旧容器
docker-compose down

# 启动新版本
docker-compose up -d
```

### 备份

```bash
# 备份日志
docker volume inspect sentiment-logs
cp -r <Mountpoint> sentiment-logs-backup

# 备份数据卷
docker run --rm \
  -v sentiment-logs:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/sentiment-logs.tar.gz /data
```

### 清理

```bash
# 删除已停止的容器
docker container prune

# 删除未使用的镜像
docker image prune

# 删除未使用的卷
docker volume prune

# 完全清理（谨慎！）
docker system prune -a
```

## 生产部署清单

- [ ] Docker 和 Docker Compose 已安装
- [ ] 镜像已成功构建并测试
- [ ] 模型工件（v0/v1）已验证
- [ ] 日志卷已创建
- [ ] 端口未被占用
- [ ] 健康检查响应正常
- [ ] 环境变量已正确配置
- [ ] 监控和日志聚合已设置
- [ ] 备份策略已实施
- [ ] 容器重启策略已配置

## 相关资源

- [Docker 官方文档](https://docs.docker.com)
- [Docker Compose 文档](https://docs.docker.com/compose)
- [Docker 最佳实践](https://docs.docker.com/develop/dev-best-practices)
- [Flask 在 Docker 中部署](https://flask.palletsprojects.com/deployment)
