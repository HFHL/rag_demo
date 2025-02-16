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





# 目标链接（修复URL赋值语法）
URL="https://shu80-my.sharepoint.com/personal/download_shu80_onmicrosoft_com/_layouts/52/download.aspx?share=EU0O61VloSBPvNbs3Lr3A_wBe1dlmfEtJYUDw5vO7fo8yw"

# 创建数据目录
mkdir -p ./data

# 检查文件是否已经下载
if [ -f ./data/downloaded_file.tar.bz2 ]; then
    echo "文件已经存在，跳过下载步骤。"
else
    # 下载文件到 ./data 目录
    echo "开始下载文件..."
    # 使用 curl 替代 wget，添加引号避免URL特殊字符问题
    curl -L -o "./data/downloaded_file.tar.bz2" "${URL}"

    # 检查文件是否下载成功
    if [ $? -eq 0 ]; then
        echo "文件下载成功!"
    else
        echo "文件下载失败!"
        exit 1
    fi
fi

# 检查是否已经解压过
if [ -d "./data/enwiki-20171001-pages-meta-current-withlinks-processed" ]; then
    echo "文件已经解压过，跳过解压步骤。"
else
    # 解压文件
    echo "开始解压文件..."
    tar -xjf ./data/downloaded_file.tar.bz2 -C ./data

    # 检查解压是否成功
    if [ $? -eq 0 ]; then
        echo "文件解压成功!"
    else
        echo "文件解压失败!"
        exit 1
    fi
fi

# 运行Python脚本时使用虚拟环境中的Python
echo "运行 Python 脚本解压 .bz2 文件..."
# python ./decompress.py

echo "运行 Python 脚本完成!"

echo "开始构建bm25索引..."
python ./build_index.py

echo "bm25索引构建完成!"