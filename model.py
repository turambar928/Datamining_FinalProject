import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import Precision, Recall
# 设置支持中文字体，例如使用SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子确保可复现性
np.random.seed(42)
tf.random.set_seed(42)

# 1. 数据加载与探索性分析
# ----------------------------------------------------
print("加载数据集...")
# 这里使用您提供的示例数据
data = pd.read_csv("diabetes_prediction_dataset.csv")
# 清理最后一行空数据
data = data.dropna()

# 显示数据集信息
print(f"数据集形状: {data.shape}")
print(f"糖尿病阳性比例: {data['diabetes'].mean():.2%}")
print("\n前5行数据:")
print(data.head())

# 可视化目标变量分布
plt.figure(figsize=(8, 5))
sns.countplot(x='diabetes', data=data)
plt.title('糖尿病分布 (0=阴性, 1=阳性)')
plt.savefig('diabetes_distribution.png', dpi=300)
plt.close()

# 2. 数据预处理
# ----------------------------------------------------
print("\n数据预处理...")

# 处理异常值
data['age'] = data['age'].clip(upper=100)  # 年龄上限设为100岁
data['bmi'] = data['bmi'].clip(10, 60)    # BMI在合理范围内

# 特征工程
data['age_group'] = pd.cut(data['age'], 
                         bins=[0, 30, 45, 60, 100],
                         labels=['<30', '30-45', '45-60', '>60'])

# 定义特征和目标
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# 分类特征和数值特征
categorical_features = ['gender', 'smoking_history', 'age_group']
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 
                     'HbA1c_level', 'blood_glucose_level']

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)
print(f"预处理后特征形状: {X_processed.shape}")

# 3. 处理类别不平衡
# ----------------------------------------------------
print("\n处理类别不平衡...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_processed, y)
print(f"重采样后数据形状: {X_res.shape}")
print(f"重采样后阳性比例: {y_res.mean():.2%}")

# 4. 划分训练集和测试集
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 5. 构建神经网络模型
# ----------------------------------------------------
print("\n构建神经网络模型...")

# 修复的自定义F1-score指标函数
def f1_metric(y_true, y_pred):
    # 确保数据类型一致
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

# 创建模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), f1_metric]
)

model.summary()

# 6. 训练模型
# ----------------------------------------------------
print("\n训练模型...")
early_stop = EarlyStopping(
    monitor='val_f1_metric', 
    patience=15, 
    verbose=1,
    mode='max',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_f1_metric',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
    mode='max'
)

# 确保数据类型一致
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32').values
y_test = y_test.astype('float32').values

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 7. 模型评估
# ----------------------------------------------------
print("\n模型评估...")
# 预测测试集
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# 计算F1-score
f1 = f1_score(y_test, y_pred)
print(f"\n测试集F1-score: {f1:.4f}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

# AUC-ROC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测非糖尿病', '预测糖尿病'],
            yticklabels=['实际非糖尿病', '实际糖尿病'])
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# 8. 训练过程可视化
# ----------------------------------------------------
plt.figure(figsize=(12, 10))

# 准确率
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('训练和验证准确率')
plt.legend()

# 损失
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('训练和验证损失')
plt.legend()

# F1-score
plt.subplot(2, 2, 3)
plt.plot(history.history['f1_metric'], label='训练F1')
plt.plot(history.history['val_f1_metric'], label='验证F1')
plt.title('训练和验证F1-score')
plt.legend()

# 精确率-召回率
plt.subplot(2, 2, 4)
plt.plot(history.history['precision'], label='训练精确率')
plt.plot(history.history['val_precision'], label='验证精确率')
plt.plot(history.history['recall'], label='训练召回率')
plt.plot(history.history['val_recall'], label='验证召回率')
plt.title('精确率和召回率')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.close()

# # 9. 特征重要性分析（使用Permutation Importance）
# # ----------------------------------------------------
# print("\n特征重要性分析...")
# from sklearn.inspection import permutation_importance

# # 使用未重采样的测试集进行特征重要性分析
# X_test_orig = preprocessor.transform(data.drop('diabetes', axis=1))
# y_test_orig = data['diabetes']

# # 创建模型包装器
# def predict_proba_wrapper(X):
#     return model.predict(X.astype('float32'))

# # 计算排列重要性
# result = permutation_importance(
#     predict_proba_wrapper, 
#     X_test_orig, 
#     y_test_orig, 
#     n_repeats=10, 
#     random_state=42, 
#     scoring='f1'
# )

# # 获取特征名称
# num_features = numerical_features
# cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
# all_features = np.concatenate([num_features, cat_features])

# # 排序特征重要性
# sorted_idx = result.importances_mean.argsort()[::-1]
# top_features = min(15, len(all_features))  # 显示前15个重要特征

# plt.figure(figsize=(12, 8))
# plt.barh(range(top_features), result.importances_mean[sorted_idx][:top_features][::-1], 
#          xerr=result.importances_std[sorted_idx][:top_features][::-1], align='center')
# plt.yticks(range(top_features), all_features[sorted_idx][:top_features][::-1])
# plt.title("特征重要性 (Permutation Importance)")
# plt.xlabel("F1-score减少量")
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300)
# plt.close()

# 10. 模型保存
# ----------------------------------------------------
model.save('diabetes_prediction_model.h5')
print("模型已保存为 'diabetes_prediction_model.h5'")

print("\n所有分析完成！结果已保存为图像文件。")