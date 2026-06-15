# 环境配置信息

## 工具路径
- **uv**: `/root/aigame/.tools/uv`
- **工作目录**: `/root/aigame/dannyyan/cs336/assignment2-systems`

## 常用命令
- 运行 Python: `/root/aigame/.tools/uv run python`
- 运行测试: `/root/aigame/.tools/uv run pytest tests/`
- 安装依赖: `/root/aigame/.tools/uv sync`

## 项目结构
- `cs336_systems/` — 学生实现代码目录（从零开始写）
- `cs336-basics/` — 作业1官方参考实现（直接使用，不需要修改）
- `tests/adapters.py` — 适配器接口，将实现连接到测试框架
- `pyproject.toml` — 项目配置，已指向本地 `cs336-basics`

## 注意事项
- Python 版本要求: >=3.12, <3.14
- PyTorch 版本: ~2.11.0
- `cs336-basics` 作为 editable 本地依赖，可直接 `import cs336_basics.model`
