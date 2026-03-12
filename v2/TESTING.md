# V2版本单元测试和Docker指南

## 概述

本目录包含 v2 情感分析系统的完整单元测试套件和 Docker 配置。

## 单元测试

### 测试覆盖范围

v2 版本包含以下测试模块：

| 模块 | 文件 | 覆盖内容 |
|------|------|---------|
| **预测函数** | `test_predict.py` | Sigmoid 函数、v0/v1 预测接口、主分发函数 |
| **模型加载** | `test_model_loader.py` | 文件发现、v0/v1 模型加载、错误处理 |
| **Flask 服务器** | `test_server.py` | 应用创建、API 端点、集成测试 |
| **命令行** | `test_cli.py` | 参数解析、命令分发、输入验证 |
| **日志配置** | `test_logging_config.py` | 日志设置、文件处理、级别配置 |
| **烟雾测试** | `test_smoke.py` | 数据管道、工件加载（来自 v0） |

### 运行测试

**运行所有测试：**
```bash
pytest
```

**运行特定测试文件：**
```bash
pytest v2/tests/test_predict.py -v
```

**运行特定测试类：**
```bash
pytest v2/tests/test_predict.py::TestSigmoid -v
```

**运行特定测试方法：**
```bash
pytest v2/tests/test_predict.py::TestSigmoid::test_sigmoid_zero_returns_half -v
```

**运行带覆盖率报告：**
```bash
pytest --cov=v2 --cov-report=html
```

覆盖率报告将生成在 `htmlcov/index.html`。

**运行并显示打印输出：**
```bash
pytest -s
```

**运行特定标记的测试：**
```bash
pytest -m "unit"
pytest -m "integration"
pytest -m "not slow"
```

### 测试依赖

测试需要以下包：
- `pytest>=7.4`
- `pytest-cov`（用于覆盖率报告）
- `pytest-mock`（用于模拟）

从 `requirements.txt` 安装依赖：
```bash
pip install -r requirements.txt
pip install pytest-cov pytest-mock
```

### 测试结构

每个测试模块遵循 AAA 模式（Arrange-Act-Assert）：

```python
def test_example():
    # Arrange: 设置测试数据和模拟
    mock_model = Mock()
    
    # Act: 执行要测试的代码
    result = predict_v0(mock_model, "test")
    
    # Assert: 验证结果
    assert result["label"] in (0, 1)
```

## Docker

### 构建 Docker 镜像

**使用 Dockerfile：**
```bash
docker build -t sentiment-api:v2 .
```

**使用标签和缓存：**
```bash
docker build --no-cache -t sentiment-api:v2 -t sentiment-api:latest .
```

### 运行容器

**交互模式运行服务器：**
```bash
docker run -p 8000:8000 sentiment-api:v2
```

**后台运行（守护进程模式）：**
```bash
docker run -d -p 8000:8000 --name sentiment-api sentiment-api:v2
```

**挂载本地模型和日志：**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/v0:/app/v0:ro \
  -v $(pwd)/v1:/app/v1:ro \
  -v $(pwd)/logs:/app/logs \
  sentiment-api:v2
```

**运行一次性预测命令：**
```bash
docker run --rm sentiment-api:v2 \
  python -m v2.cli predict \
  --text "This movie is amazing!" \
  --model v0
```

### Docker Compose

使用 Docker Compose 简化管理：

**启动服务：**
```bash
docker-compose up
```

**后台启动：**
```bash
docker-compose up -d
```

**查看日志：**
```bash
docker-compose logs -f sentiment-api
```

**停止服务：**
```bash
docker-compose down
```

**重建镜像：**
```bash
docker-compose build --no-cache
```

### DockerFile 特点

- **多阶段构建**：减小最终镜像大小
- **虚拟环境**：在构建阶段创建并复制到运行时
- **健康检查**：自动检查服务可用性
- **非 Root 用户**：提高安全性（可选扩展）
- **缓存优化**：首先复制 requirements.txt

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | 8000 | 服务监听端口 |
| `HOST` | 0.0.0.0 | 服务监听地址 |
| `LOG_LEVEL` | INFO | 日志级别 |

### 健康检查

容器包含内置健康检查，定期验证 `/health` 端点：

```bash
docker ps  # 查看容器状态
```

状态将为 `Up` 或 `Unhealthy`。

### 容器化的设置要求

1. **模型工件**：在运行前，通过运行 v0/v1 生成模型工件
2. **配置文件**：可选的 `config.json` 会被加载
3. **依赖**：所有 Python 依赖在镜像中自动安装

## 集成工作流

### 本地开发

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 2. 运行测试
pytest -v

# 3. 启动开发服务器
python -m v2.cli serve
```

### Docker 工作流

```bash
# 1. 构建镜像
docker build -t sentiment-api:v2 .

# 2. 在容器中运行测试
docker run --rm sentiment-api:v2 pytest v2/tests/

# 3. 启动服务
docker-compose up -d

# 4. 验证健康状态
curl http://localhost:8000/health
```

### CI/CD 集成

在 GitHub Actions 或 GitLab CI 中使用：

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest --cov=v2
```

## 故障排除

### 测试失败

**问题**：`ModuleNotFoundError: No module named 'v2'`
```bash
# 解决：从项目根目录运行 pytest
cd /path/to/project
pytest
```

**问题**：模型加载测试跳过
```bash
# 解决：生成 v0/v1 工件
python -m v0.v0_auto
python -m v1.v1_auto
```

### Docker 构建失败

**问题**：`COPY failed: file not found`
```bash
# 解决：从项目根目录构建
docker build -t sentiment-api:v2 .
```

**问题**：`ModuleNotFoundError` 在容器中
```bash
# 解决：重建镜像
docker build --no-cache -t sentiment-api:v2 .
```

### 健康检查失败

```bash
# 检查容器日志
docker logs sentiment-api-v2

# 手动测试端点
curl http://localhost:8000/health
```

## 最佳实践

1. **单元测试**：在开发每个功能时编写测试
2. **覆盖率**：目标是 >80% 的代码覆盖率
3. **模拟**：使用 `unittest.mock` 隔离外部依赖
4. **Docker**：在生产前在容器中测试
5. **日志**：配置日志进行调试而非 `print()` 语句

## 相关资源

- [Pytest 文档](https://docs.pytest.org/)
- [Docker 最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [继续集成](https://docs.github.com/en/actions)
