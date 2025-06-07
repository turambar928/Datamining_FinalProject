import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

print("=" * 60)
print("XGBoost 糖尿病预测模型")
print("=" * 60)

# 1. 数据加载与探索性分析
print("\n1. 加载数据集...")
data = pd.read_csv("diabetes_prediction_dataset.csv")
data = data.dropna()

print(f"数据集形状: {data.shape}")
print(f"糖尿病阳性比例: {data['diabetes'].mean():.2%}")
print(data.info())
print(data.head())

print(f"\n目标变量分布:")
print(data['diabetes'].value_counts())

# 2. 数据预处理
print("\n2. 数据预处理...")
data['age'] = data['age'].clip(upper=100)
data['bmi'] = data['bmi'].clip(10, 60)

data['age_group'] = pd.cut(data['age'],
                           bins=[0, 30, 45, 60, 100],
                           labels=['young', 'middle_aged', 'senior', 'elderly'])

label_encoders = {}
categorical_features = ['gender', 'smoking_history', 'age_group']
for feature in categorical_features:
    le = LabelEncoder()
    data[feature + '_encoded'] = le.fit_transform(data[feature])
    label_encoders[feature] = le

feature_columns = ['age', 'hypertension', 'heart_disease', 'bmi',
                   'HbA1c_level', 'blood_glucose_level',
                   'gender_encoded', 'smoking_history_encoded', 'age_group_encoded']
X = data[feature_columns]
y = data['diabetes']

print(f"特征列: {feature_columns}")
print(f"特征矩阵形状: {X.shape}")

# 3. 数据标准化
print("\n3. 数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# 4. 处理类别不平衡
print("\n4. 处理类别不平衡...")
print(f"原始分布: {y.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"SMOTE后分布: {pd.Series(y_resampled).value_counts().to_dict()}")

# 5. 划分训练集和测试集
print("\n5. 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"训练集大小: {X_train.shape[0]}，测试集大小: {X_test.shape[0]}")

# 6. 使用随机搜索优化XGBoost模型
print("\n6. 使用随机搜索优化XGBoost模型...")

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
    n_jobs=-1
)

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 7),
    'learning_rate': uniform(0.05, 0.15),  # 0.05 ~ 0.2
    'subsample': uniform(0.7, 0.3),        # 0.7 ~ 1.0
    'colsample_bytree': uniform(0.7, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"最佳参数: {random_search.best_params_}")
print(f"最佳交叉验证F1分数: {random_search.best_score_:.4f}")
best_xgb_model = random_search.best_estimator_

# 7. 模型评估
print("\n7. 模型评估...")
y_pred = best_xgb_model.predict(X_test)
y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

# 8. 可视化结果
print("\n8. 可视化结果...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['预测非糖尿病', '预测糖尿病'],
            yticklabels=['实际非糖尿病', '实际糖尿病'])
axes[0, 0].set_title('混淆矩阵')

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('假阳性率')
axes[0, 1].set_ylabel('真正率')
axes[0, 1].set_title('ROC 曲线')
axes[0, 1].legend(loc='lower right')

# 特征重要性
xgb.plot_importance(best_xgb_model, ax=axes[1, 0], importance_type='gain', show_values=False)
axes[1, 0].set_title("特征重要性（Gain）")

# 空图
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 9. 模型保存
print("\n9. 保存模型...")
joblib.dump(best_xgb_model, "xgboost_model.pkl")
best_xgb_model.save_model("xgboost_model.json")
print("模型已保存为 xgboost_model.pkl 和 xgboost_model.json，请在左侧文件栏中点击刷新查看。")


