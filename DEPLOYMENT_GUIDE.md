# XGBoost 糖尿病预测模型华为云部署指南

## 📋 目录
- [项目概述](#项目概述)
- [部署准备](#部署准备)
- [本地测试](#本地测试)
- [华为云ECS部署](#华为云ecs部署)
- [Docker容器化部署](#docker容器化部署)
- [华为云CCE部署](#华为云cce部署)
- [API使用说明](#api使用说明)
- [监控与维护](#监控与维护)
- [常见问题](#常见问题)

## 🎯 项目概述

本项目提供了一个基于XGBoost的糖尿病预测模型API服务，支持单个预测和批量预测功能。

### 📁 文件说明
```
├── app.py                    # Flask API服务主程序
├── xgboost_model.pkl        # 训练好的XGBoost模型文件
├── requirements.txt         # Python依赖包列表
├── start.sh                 # 启动脚本
├── Dockerfile              # Docker配置文件
├── test_api.py             # API测试脚本
└── DEPLOYMENT_GUIDE.md     # 部署指南(本文件)
```

### 🔧 API接口
- `GET /` - API信息和接口列表
- `GET /health` - 健康检查
- `POST /predict` - 单个预测
- `POST /predict_batch` - 批量预测

## 🚀 部署准备

### 系统要求
- Python 3.8+
- 内存: 2GB+
- 磁盘: 1GB+
- 网络: 公网IP或域名

### 必需文件检查
确保以下文件存在于项目目录中：
- ✅ `app.py`
- ✅ `xgboost_model.pkl` (重要: 337KB的模型文件)
- ✅ `requirements.txt`
- ✅ `start.sh`

## 🧪 本地测试

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动服务
```bash
python app.py
```

### 3. 测试API
```bash
python test_api.py
```

如果看到"✓ API主页访问成功"等成功信息，说明本地部署正常。

## 🌐 华为云ECS部署

### 步骤1: 购买ECS实例

1. 登录[华为云控制台](https://console.huaweicloud.com)
2. 进入"弹性云服务器 ECS"
3. 创建实例:
   - **规格**: 通用计算型 s6.large.2 (2核4GB) 或更高
   - **操作系统**: Ubuntu 20.04 LTS
   - **网络**: 选择VPC和子网
   - **安全组**: 开放5000端口
   - **存储**: 40GB系统盘

### 步骤2: 配置安全组

1. 进入"网络 > 安全组"
2. 找到ECS实例的安全组
3. 添加入方向规则:
   - 协议端口: TCP 5000
   - 源地址: 0.0.0.0/0 (或指定IP)

### 步骤3: 连接服务器

```bash
# Windows用户使用PuTTY或PowerShell
ssh ubuntu@<你的服务器公网IP>

# 或使用密钥文件
ssh -i your-key.pem ubuntu@<你的服务器公网IP>
```

### 步骤4: 环境配置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和工具
sudo apt install python3 python3-pip git wget curl -y

# 创建项目目录
mkdir ~/diabetes_prediction
cd ~/diabetes_prediction
```

### 步骤5: 上传文件

**方法1: 使用scp命令**
```bash
# 在本地执行，上传所有必要文件
scp app.py ubuntu@<服务器IP>:~/diabetes_prediction/
scp xgboost_model.pkl ubuntu@<服务器IP>:~/diabetes_prediction/
scp requirements.txt ubuntu@<服务器IP>:~/diabetes_prediction/
scp start.sh ubuntu@<服务器IP>:~/diabetes_prediction/
```

**方法2: 使用WinSCP或其他SFTP工具**
- 将以下文件上传到服务器的 `~/diabetes_prediction/` 目录:
  - `app.py`
  - `xgboost_model.pkl`
  - `requirements.txt`
  - `start.sh`

### 步骤6: 部署应用

```bash
cd ~/diabetes_prediction

# 安装Python依赖
pip3 install -r requirements.txt

# 给启动脚本执行权限
chmod +x start.sh

# 测试运行
python3 app.py
```

如果看到"API服务启动成功!"，按Ctrl+C停止，然后继续下一步。

### 步骤7: 生产环境部署

```bash
# 使用gunicorn启动
./start.sh
```

### 步骤8: 配置系统服务(可选)

创建systemd服务，让API开机自启动：

```bash
sudo nano /etc/systemd/system/diabetes-api.service
```

添加以下内容：
```ini
[Unit]
Description=Diabetes Prediction API
After=network.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/home/ubuntu/diabetes_prediction
Environment=PATH=/home/ubuntu/.local/bin
ExecStart=/home/ubuntu/.local/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app --timeout 120
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start diabetes-api
sudo systemctl enable diabetes-api
sudo systemctl status diabetes-api
```

### 步骤9: 测试部署

```bash
# 在服务器上测试
curl http://localhost:5000/health

# 在本地测试(替换为你的服务器IP)
python test_api.py http://<服务器IP>:5000
```

## 🐳 Docker容器化部署

### 本地构建和测试

```bash
# 构建Docker镜像
docker build -t diabetes-prediction-api .

# 运行容器
docker run -d -p 5000:5000 --name diabetes-api diabetes-prediction-api

# 测试
curl http://localhost:5000/health
```

### 华为云SWR部署

1. **登录华为云容器镜像服务**
   - 进入"容器镜像服务 SWR"
   - 创建组织(namespace)

2. **推送镜像**
```bash
# 登录华为云SWR
docker login swr.cn-north-4.myhuaweicloud.com

# 标记镜像
docker tag diabetes-prediction-api swr.cn-north-4.myhuaweicloud.com/<你的namespace>/diabetes-prediction-api:latest

# 推送镜像
docker push swr.cn-north-4.myhuaweicloud.com/<你的namespace>/diabetes-prediction-api:latest
```

3. **在ECS上运行**
```bash
# 拉取镜像
docker pull swr.cn-north-4.myhuaweicloud.com/<你的namespace>/diabetes-prediction-api:latest

# 运行容器
docker run -d -p 5000:5000 \
  --name diabetes-api \
  --restart always \
  swr.cn-north-4.myhuaweicloud.com/<你的namespace>/diabetes-prediction-api:latest
```

## ☸️ 华为云CCE部署

### 1. 创建CCE集群
1. 进入"云容器引擎 CCE"
2. 创建集群(选择标准集群)
3. 配置节点池

### 2. 部署应用

创建`k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diabetes-prediction-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: diabetes-prediction-api
  template:
    metadata:
      labels:
        app: diabetes-prediction-api
    spec:
      containers:
      - name: api
        image: swr.cn-north-4.myhuaweicloud.com/<你的namespace>/diabetes-prediction-api:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
apiVersion: v1
kind: Service
metadata:
  name: diabetes-prediction-service
spec:
  selector:
    app: diabetes-prediction-api
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

部署应用：
```bash
kubectl apply -f k8s-deployment.yaml
```

## 📚 API使用说明

### 单个预测示例

```bash
curl -X POST http://<服务器IP>:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "male",
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140,
    "smoking_history": "former",
    "hypertension": 1,
    "heart_disease": 0
  }'
```

### 批量预测示例

```bash
curl -X POST http://<服务器IP>:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "age": 35,
        "gender": "female",
        "bmi": 22.0,
        "HbA1c_level": 5.0,
        "blood_glucose_level": 100,
        "smoking_history": "never",
        "hypertension": 0,
        "heart_disease": 0
      },
      {
        "age": 65,
        "gender": "male",
        "bmi": 32.0,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 180,
        "smoking_history": "current",
        "hypertension": 1,
        "heart_disease": 1
      }
    ]
  }'
```

### 输入字段说明

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|---------|
| age | 数字 | 年龄(0-100) | 45 |
| gender | 字符串 | 性别 | "male", "female" |
| bmi | 数字 | 体重指数(10-60) | 28.5 |
| HbA1c_level | 数字 | 糖化血红蛋白水平 | 6.5 |
| blood_glucose_level | 数字 | 血糖水平 | 140 |
| smoking_history | 字符串 | 吸烟史 | "never", "former", "current", "not current", "ever", "no info" |
| hypertension | 数字 | 高血压(0或1) | 1 |
| heart_disease | 数字 | 心脏病(0或1) | 0 |

## 📊 监控与维护

### 查看服务状态
```bash
# systemd服务
sudo systemctl status diabetes-api

# Docker容器
docker ps
docker logs diabetes-api

# 进程监控
ps aux | grep gunicorn
```

### 查看日志
```bash
# systemd日志
sudo journalctl -u diabetes-api -f

# Docker日志
docker logs -f diabetes-api
```

### 性能监控
```bash
# 系统资源
htop
free -h
df -h

# 网络连接
netstat -tlnp | grep 5000
```

### 重启服务
```bash
# systemd服务
sudo systemctl restart diabetes-api

# Docker容器
docker restart diabetes-api
```

## ❓ 常见问题

### Q1: 模型加载失败
**症状**: API返回"模型未加载"错误

**解决方案**:
1. 检查`xgboost_model.pkl`文件是否存在
2. 确认文件大小约为337KB
3. 检查文件权限: `chmod 644 xgboost_model.pkl`

### Q2: 端口访问被拒绝
**症状**: 无法访问API接口

**解决方案**:
1. 检查安全组是否开放5000端口
2. 检查服务器防火墙: `sudo ufw status`
3. 确认服务正常运行: `netstat -tlnp | grep 5000`

### Q3: 内存不足
**症状**: 服务启动失败或频繁重启

**解决方案**:
1. 升级ECS实例规格
2. 减少gunicorn worker数量: 修改`start.sh`中的`-w 4`为`-w 2`
3. 增加swap空间

### Q4: 预测结果异常
**症状**: 预测结果始终相同或明显错误

**解决方案**:
1. 检查输入数据格式
2. 验证特征编码是否正确
3. 确认模型文件完整性

### Q5: 性能问题
**症状**: API响应缓慢

**解决方案**:
1. 增加gunicorn worker数量
2. 使用更高规格的ECS实例
3. 考虑使用负载均衡

## 🔒 安全建议

1. **网络安全**
   - 仅开放必要端口
   - 使用VPC私有网络
   - 配置防火墙规则

2. **访问控制**
   - 添加API认证机制
   - 限制API调用频率
   - 记录访问日志

3. **数据安全**
   - 对敏感数据加密
   - 定期备份模型文件
   - 监控异常访问

## 📞 技术支持

如遇到部署问题，请检查：
1. 所有文件是否完整上传
2. Python依赖是否正确安装
3. 网络和端口配置是否正确
4. 服务器资源是否充足

---

**部署成功后，你的XGBoost糖尿病预测API就可以为用户提供7x24小时的预测服务了！** 🎉 