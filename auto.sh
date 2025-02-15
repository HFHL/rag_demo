#!/bin/bash

# 配置环境
echo "正在创建并激活环境..."
# 创建环境
conda env create -f environment.yml

# 激活环境
source activate rag

echo "环境创建并激活成功!"

# 目标链接
# url="https://shu80-my.sharepoint.com/personal/download_shu80_onmicrosoft_com/_layouts/52/download.aspx?share=EU0O61VloSBPvNbs3Lr3A_wBe1dlmfEtJYUDw5vO7fo8yw"

url="https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"


# 创建数据目录，如果目录不存在
mkdir -p ./data

# 下载文件到 ./data 目录
echo "开始下载文件..."
wget -O ./data/downloaded_file.tar.bz2 "$url"

# 检查文件是否下载成功
if [ $? -eq 0 ]; then
  echo "文件下载成功!"
else
  echo "文件下载失败!"
  exit 1
fi

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


echo "运行 Python 脚本解压 .bz2 文件..."
python ./decompress.py

echo "运行 Python 脚本完成!"

echo "开始构建bm25索引..."
python ./build_index.py

echo "bm25索引构建完成!"