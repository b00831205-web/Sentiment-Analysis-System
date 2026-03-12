# V2 版本 - 测试和 Docker 增强

## 新增功能概述

v2 版本现在包含：

### ✅ 完整的单元测试套件
- **5 个主要测试模块**：预测、模型加载、服务器、CLI、日志配置
- **50+ 个单元测试**：覆盖所有关键功能和边界情况
- **代码覆盖率**：目标 >60%
- **烟雾测试**：v0/v1 工件加载验证

### 🐳 Docker 容器化
- **多阶段构建**：优化的镜像大小和构建时间
- **健康检查**：自动服务可用性监控
- **Docker Compose**：简化开发和部署
- **完整文档**：部署指南和最佳实践

## 文件结构

```
v2/
├── __init__.py
├── cli.py              # 命令行接口
├── server.py           # Flask 服务器
├── predict.py          # 预测核心逻辑
├── model_loader.py     # 模型加载工具
├── logging_config.py   # 日志配置
├── TESTING.md          # 测试完整指南
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py           # 烟雾测试（继承）
│   ├── test_predict.py         # 预测函数测试（新增）
│   ├── test_model_loader.py    # 模型加载测试（新增）
│   ├── test_server.py          # 服务器端点测试（新增）
│   ├── test_cli.py             # CLI 命令测试（新增）
│   └── test_logging_config.py  # 日志配置测试（新增）
└── README.md

根目录/
├── Dockerfile                  # Docker 镜像定义（新增）
├── .dockerignore              # Docker 构建忽略文件（新增）
├── docker-compose.yml         # Docker Compose 配置（新增）
├── pytest.ini                 # Pytest 配置文件（新增）
├── setup.py                   # Python 包安装配置（新增）
├── DOCKER_DEPLOYMENT.md       # Docker 部署指南（新增）
└── requirements.txt
```

## 快速开始

### 本地开发和测试

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock

# 2. 运行所有测试
pytest -v

# 3. 查看覆盖率报告
pytest --cov=v2 --cov-report=html
# 打开 htmlcov/index.html

# 4. 运行特定测试
pytest v2/tests/test_predict.py -v
```

### Docker 开发和部署

```bash
# 1. 构建镜像
docker build -t sentiment-api:v2 .

# 2. 使用 Docker Compose 启动
docker-compose up

# 3. 访问 API
curl http://localhost:8000/health

# 4. 进行预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is great!", "model_kind": "v0"}'
```

## 测试详情

### 测试覆盖的功能

#### test_predict.py
- ✅ Sigmoid 函数数值稳定性
- ✅ v0 模型预测接口（with/without predict_proba）
- ✅ v1 模型预测接口（神经网络推理）
- ✅ 主分发函数（v0/v1 选择）
- ✅ 错误处理和输入验证

#### test_model_loader.py
- ✅ 最新文件发现（glob 模式）
- ✅ v0 工件加载和验证
- ✅ v1 工件加载和验证
- ✅ 缺失文件错误处理
- ✅ 数据结构完整性验证

#### test_server.py
- ✅ Flask 应用创建
- ✅ /health 端点
- ✅ /predict POST 端点
- ✅ HTML UI GET /
- ✅ 404 错误处理
- ✅ 长文本处理

#### test_cli.py
- ✅ CLI 参数解析
- ✅ predict 和 serve 命令
- ✅ 配置文件加载
- ✅ 输入验证
- ✅ 错误处理

#### test_logging_config.py
- ✅ 日志目录创建
- ✅ 日志文件生成
- ✅ 日志级别配置
- ✅ 多处理程序设置（控制台 + 文件）
- ✅ 日志格式验证

### 运行特定测试示例

```bash
# 运行所有预测测试
pytest v2/tests/test_predict.py -v

# 运行特定测试类
pytest v2/tests/test_predict.py::TestSigmoid -v

# 运行特定测试方法
pytest v2/tests/test_predict.py::TestSigmoid::test_sigmoid_zero_returns_half -v

# 运行并显示打印输出
pytest -s v2/tests/test_server.py

# 运行并显示覆盖率缺失行
pytest --cov=v2 --cov-report=term-missing v2/tests/

# 仅运行"单位"标记的测试
pytest -m "unit" -v
```

## Docker 详情

### Dockerfile 特点

- **多阶段构建**：
  - Stage 1: 构建虚拟环境和依赖
  - Stage 2: 精简运行时镜像

- **优化：**
  - 缓存层利用（requirements.txt 优先复制）
  - 虚拟环境复用减小最终镜像大小
  - 精简基础镜像（python:3.11-slim）

- **生产就绪：**
  - 健康检查：每 30 秒检查一次
  - 自动重启：失败时重新启动
  - 日志挂载点：持久化日志
  - 模型挂载点：运行时模型加载

### docker-compose.yml 特点

- **服务定义**：sentiment-api 服务
- **数据卷**：logs、v0、v1 挂载
- **健康检查**：curl 基础检查
- **网络隔离**：自定义 sentiment-network
- **环境变量**：灵活配置

### Docker 命令快速参考

```bash
# 构建
docker build -t sentiment-api:v2 .

# 运行
docker run -d -p 8000:8000 sentiment-api:v2

# Compose
docker-compose up -d
docker-compose logs -f
docker-compose down

# 调试
docker exec -it <container_id> bash
docker inspect <container_id>
docker stats <container_id>
```

## API 参考

### POST /predict

**请求：**
```json
{
  "text": "This movie is amazing!",
  "model_kind": "v0"
}
```

**响应（成功）：**
```json
{
  "label": 1,
  "prob_pos": 0.95,
  "model": "v0",
  "model_path": "/app/v0/best_model_v0_20240110.joblib",
  "model_name": "logistic_regression"
}
```

### GET /health

**响应：**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### GET /

**响应：** HTML UI 用于交互式预测

## 性能指标

| 指标 | 值 |
|------|-----|
| 镜像大小 | ~500MB |
| 启动时间 | <5 秒 |
| 健康检查间隔 | 30 秒 |
| 预测延迟（v0） | <100ms |
| 预测延迟（v1） | <150ms |

## 故障排除

### 测试失败

```bash
# 问题：ModuleNotFoundError
# 解决：从项目根目录运行
cd /path/to/Sentiment-Analysis-System
pytest

# 问题：模型工件未找到
# 解决：生成 v0/v1 工件
python -m v0.v0_auto
python -m v1.v1_auto
```

### Docker 问题

```bash
# 问题：容器无法启动
# 解决：检查日志
docker logs sentiment-api-v2

# 问题：端口被占用
# 解决：改用不同端口
docker run -p 9000:8000 sentiment-api:v2

# 问题：镜像构建失败
# 解决：清理缓存重建
docker build --no-cache -t sentiment-api:v2 .
```

## 文档链接

- 📖 [详细测试指南](v2/TESTING.md)
- 🐳 [Docker 部署指南](DOCKER_DEPLOYMENT.md)
- 📝 [V2 原始 README](v2/README.md)
- ⚙️ [项目配置](config.json)

## 贡献指南

1. 编写对应的单元测试
2. 运行 `pytest` 确保所有测试通过
3. 检查覆盖率：`pytest --cov=v2`
4. 在容器中测试：`docker-compose up`

## 许可证

与主项目相同

## 支持

如有问题或建议，请提交 Issue 或 Pull Request。
