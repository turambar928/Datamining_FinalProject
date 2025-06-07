import requests
import json
import os

# 设置环境变量，禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

def test_prediction():
    """简单的预测测试"""
    
    print("🔍 测试XGBoost糖尿病预测API")
    print("=" * 50)
    
    # 测试数据 - 使用正确的字段值格式
    test_data = {
        "age": 45,
        "gender": "Male",  # 注意首字母大写
        "bmi": 28.5,
        "HbA1c_level": 6.5,
        "blood_glucose_level": 140,
        "smoking_history": "former",  # 吸烟史值
        "hypertension": 1,
        "heart_disease": 0
    }
    
    try:
        # 1. 健康检查
        print("\n1. 健康检查...")
        response = requests.get("http://localhost:5000/health", proxies={'http': None, 'https': None})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务状态: {data.get('status')}")
            print(f"✅ 模型已加载: {data.get('model_loaded')}")
        
        # 2. 单个预测
        print("\n2. 单个预测测试...")
        response = requests.post(
            "http://localhost:5000/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            proxies={'http': None, 'https': None}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 预测成功!")
            print(f"   输入数据: 45岁男性，BMI={test_data['bmi']}, HbA1c={test_data['HbA1c_level']}")
            print(f"   预测结果: {result.get('prediction_text')}")
            print(f"   糖尿病概率: {result.get('probability', {}).get('diabetes', 0):.4f}")
            print(f"   非糖尿病概率: {result.get('probability', {}).get('non_diabetes', 0):.4f}")
            print(f"   置信度: {result.get('confidence', 0):.4f}")
        else:
            print(f"❌ 预测失败: {response.text}")
        
        # 3. 另一个测试案例
        print("\n3. 低风险案例测试...")
        low_risk_data = {
            "age": 25,
            "gender": "Female",  # 注意首字母大写
            "bmi": 22.0,
            "HbA1c_level": 5.0,
            "blood_glucose_level": 90,
            "smoking_history": "never",
            "hypertension": 0,
            "heart_disease": 0
        }
        
        response = requests.post(
            "http://localhost:5000/predict",
            json=low_risk_data,
            headers={'Content-Type': 'application/json'},
            proxies={'http': None, 'https': None}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 低风险预测成功!")
            print(f"   输入数据: 25岁女性，BMI={low_risk_data['bmi']}, HbA1c={low_risk_data['HbA1c_level']}")
            print(f"   预测结果: {result.get('prediction_text')}")
            print(f"   糖尿病概率: {result.get('probability', {}).get('diabetes', 0):.4f}")
            print(f"   置信度: {result.get('confidence', 0):.4f}")
        
        print("\n" + "=" * 50)
        print("🎉 测试完成！API功能正常，可以部署到华为云了！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    test_prediction() 