#!/bin/bash

echo "========================================"
echo "启动XGBoost糖尿病预测API服务"
echo "========================================"

# 检查模型文件是否存在
if [ ! -f "xgboost_model.pkl" ]; then
    echo "错误: 找不到模型文件 xgboost_model.pkl"
    echo "请确保模型文件在当前目录中"
    exit 1
fi

echo "模型文件检查通过..."

# 检查Python包是否已安装
echo "检查依赖包..."
pip list | grep flask > /dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖包..."
    pip install -r requirements.txt
fi

echo "启动API服务..."
echo "服务地址: http://0.0.0.0:5000"
echo "使用 Ctrl+C 停止服务"
echo "========================================"

# 使用gunicorn启动Flask应用
# -w 4: 使用4个worker进程
# -b 0.0.0.0:5000: 绑定到所有网络接口的5000端口
# --timeout 120: 请求超时时间120秒
# --keep-alive 2: 保持连接2秒
# --max-requests 1000: 每个worker处理1000个请求后重启
exec gunicorn -w 4 -b 0.0.0.0:5000 app:app \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --access-logfile - \
    --error-logfile - 