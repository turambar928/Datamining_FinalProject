from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 全局变量存储模型和预处理器
model = None
feature_stats = None

def load_model():
    """加载训练好的模型和特征统计信息"""
    global model, feature_stats
    
    try:
        import xgboost as xgb
        
        # 优先使用JSON格式加载，兼容性最好
        try:
            print("尝试使用JSON格式加载模型...")
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.load_model('xgboost_model.json')
            print("✅ 使用JSON格式成功加载XGBoost模型")
            
        except Exception as e1:
            print(f"JSON加载失败: {str(e1)}")
            print("尝试使用pickle格式加载...")
            
            # 如果JSON失败，尝试pickle格式
            try:
                model = joblib.load('xgboost_model.pkl')
                print("✅ 使用pickle格式成功加载XGBoost模型")
            except Exception as e2:
                print(f"pickle加载也失败: {str(e2)}")
                raise Exception(f"所有模型加载方式都失败了。JSON错误: {str(e1)}, Pickle错误: {str(e2)}")
        
        # 基于训练数据的准确特征统计信息（用于标准化）
        feature_stats = {
            'means': [41.886437, 0.07485, 0.03942, 27.31264, 5.527507, 138.058354, 0.41448, 2.180347, 1.663069],
            'stds': [22.517043, 0.263438, 0.194835, 6.59013, 1.070677, 40.708136, 0.493031, 1.889659, 1.170753],
            'feature_names': ['age', 'hypertension', 'heart_disease', 'bmi',
                            'HbA1c_level', 'blood_glucose_level',
                            'gender_encoded', 'smoking_history_encoded', 'age_group_encoded']
        }
        
        print("✅ 模型加载成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

def encode_categorical_features(data):
    """对分类特征进行编码 - 与训练时LabelEncoder完全一致"""
    
    # 性别编码 - LabelEncoder输出: Female->0, Male->1, Other->2
    gender_mapping = {
        'Female': 0, 'Male': 1, 'Other': 2,
        'female': 0, 'male': 1, 'other': 2  # 兼容小写输入
    }
    gender_value = data.get('gender', 'Female')
    data['gender_encoded'] = gender_mapping.get(gender_value, 0)
    
    # 吸烟史编码 - LabelEncoder输出: current->0, ever->2, former->3, never->4, 'No Info'->5, 'not current'->1
    smoking_mapping = {
        'current': 0,
        'not current': 1, 
        'ever': 2,
        'former': 3,
        'never': 4,
        'No Info': 5,
        'no info': 5  # 兼容小写输入
    }
    smoking_value = data.get('smoking_history', 'never')
    data['smoking_history_encoded'] = smoking_mapping.get(smoking_value, 4)  # 默认为never(4)
    
    # 年龄组编码 - LabelEncoder输出: elderly->0, middle_aged->1, senior->2, young->3
    age = data.get('age', 0)
    if age <= 30:
        age_group_encoded = 3  # young
    elif age <= 45:
        age_group_encoded = 1  # middle_aged  
    elif age <= 60:
        age_group_encoded = 2  # senior
    else:
        age_group_encoded = 0  # elderly
    
    data['age_group_encoded'] = age_group_encoded
    
    return data

def preprocess_input(data):
    """预处理输入数据"""
    try:
        # 数据清理
        processed_data = data.copy()
        processed_data['age'] = min(max(processed_data.get('age', 0), 0), 100)
        processed_data['bmi'] = min(max(processed_data.get('bmi', 0), 10), 60)
        
        # 编码分类特征
        processed_data = encode_categorical_features(processed_data)
        
        # 提取特征
        features = []
        for feature_name in feature_stats['feature_names']:
            features.append(processed_data.get(feature_name, 0))
        
        # 标准化
        features = np.array(features)
        means = np.array(feature_stats['means'])
        stds = np.array(feature_stats['stds'])
        
        features_scaled = (features - means) / stds
        
        return features_scaled.reshape(1, -1)
        
    except Exception as e:
        raise ValueError(f"数据预处理错误: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """主页"""
    return jsonify({
        "message": "XGBoost 糖尿病预测 API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "/": "API信息",
            "/health": "健康检查",
            "/predict": "单个预测 (POST)",
            "/predict_batch": "批量预测 (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file_exists": os.path.exists('xgboost_model.pkl')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """单个预测接口"""
    try:
        # 检查模型是否已加载
        if model is None:
            return jsonify({"error": "模型未加载"}), 500
            
        # 获取输入数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请提供JSON格式的输入数据"}), 400
        
        # 验证必需字段
        required_fields = ['age', 'gender', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                          'smoking_history', 'hypertension', 'heart_disease']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"缺少必需字段: {', '.join(missing_fields)}"}), 400
        
        # 预处理数据
        processed_data = preprocess_input(data)
        
        # 进行预测
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # 返回结果
        result = {
            "prediction": int(prediction),
            "prediction_text": "糖尿病" if prediction == 1 else "非糖尿病",
            "probability": {
                "non_diabetes": float(probability[0]),
                "diabetes": float(probability[1])
            },
            "confidence": float(max(probability)),
            "input_data": data
        }
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    try:
        if model is None:
            return jsonify({"error": "模型未加载"}), 500
            
        data = request.get_json()
        
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({"error": "请提供 'data' 字段，包含预测数据列表"}), 400
        
        results = []
        for i, item in enumerate(data['data']):
            try:
                processed_data = preprocess_input(item)
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0]
                
                result = {
                    "index": i,
                    "prediction": int(prediction),
                    "prediction_text": "糖尿病" if prediction == 1 else "非糖尿病",
                    "probability": {
                        "non_diabetes": float(probability[0]),
                        "diabetes": float(probability[1])
                    },
                    "confidence": float(max(probability))
                }
                results.append(result)
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "total_samples": len(data['data']),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": f"批量预测失败: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "接口不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误"}), 500

if __name__ == '__main__':
    # 启动时加载模型
    print("正在启动XGBoost糖尿病预测API服务...")
    if load_model():
        print("API服务启动成功!")
        print("访问 http://localhost:5000 查看API信息")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("模型加载失败，无法启动服务")
        print("请确保 xgboost_model.pkl 文件存在于当前目录") 