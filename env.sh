#!/bin/bash

# 创建并激活虚拟环境
echo "正在创建虚拟环境..."
if [ ! -d "./.venv" ]; then
    python3 -m venv .venv
    echo "虚拟环境创建成功!"
else
    echo "虚拟环境已存在，直接激活"
fi

# 激活虚拟环境 (使用完整路径)
. ./.venv/bin/activate

# 检查激活是否成功
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: 虚拟环境激活失败!"
    exit 1
fi

# 验证Python环境
echo "Python 路径："
command -v python
echo "Python 版本："
python -V

echo "环境激活成功!"

# 安装依赖
echo "正在安装依赖..."

# 确保pip是最新的
./.venv/bin/pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 检查requirements.txt是否存在
if [ -f "requirements.txt" ]; then
    echo "开始安装依赖..."
    ./.venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "依赖安装完成!"
else
    echo "Error: requirements.txt 文件不存在!"
    exit 1
fi

echo "环境配置完成!"


