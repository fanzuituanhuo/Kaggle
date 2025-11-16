import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\文件资料\Downloads\kaggle\titanic\train.csv")
kaggle_testing = pd.read_csv(r"D:\文件资料\Downloads\kaggle\titanic\test.csv")

def age_group(age):
    """
    根据年龄段生存率特征进行分桶:
    0-10: 很高 (儿童优先上船) -> 0
    10-15: 中等偏高 -> 1
    15-28: 最低 (典型成年男性) -> 2
    28-50: 稍高 -> 3
    50-70: 更高 (富人多) -> 4
    70+: 略低 -> 5
    """
    if pd.isna(age):
        return None
    if age < 10:
        return 0
    elif age < 15:
        return 1
    elif age < 28:
        return 2
    elif age < 50:
        return 3
    elif age < 70:
        return 4
    else:
        return 5

def extract_title(name: str) -> str:
    if pd.isna(name): return 'Rare'
    title = name.split(',')[1].split('.')[0].strip()
    mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
               'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
               'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
               'Jonkheer': 'Rare', 'Dona': 'Rare'}
    return mapping.get(title, title if title in {'Mr', 'Mrs', 'Miss', 'Master'} else 'Rare')

def fill_missing_age(data, title_medians, age_median, log_prefix=""):
    """按Title填充缺失年龄"""
    age_mask = data['Age'].isna()
    if age_mask.any():
        data.loc[age_mask, 'Age'] = data.loc[age_mask, 'Title'].map(title_medians).fillna(age_median)
        if log_prefix:
            print(f"{log_prefix}: 按Title填充{age_mask.sum()}个Age缺失值")

def fill_missing_fare(data, fare_median, log_prefix=""):
    """填充缺失票价"""
    fare_mask = data['Fare'].isna()
    if fare_mask.any():
        data.loc[fare_mask, 'Fare'] = fare_median
        if log_prefix:
            print(f"{log_prefix}: 填充{fare_mask.sum()}个Fare缺失值")

def preprocess_data(df, scaler, title_medians, age_median, fare_median, fit_scaler=False, log_prefix=""):
    """数据预处理：特征选择、编码、填充、标准化"""
    # 选择特征并复制
    data = df[['Pclass', 'Sex', 'Fare', 'Age', 'Name']].copy()

    # 性别编码
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # 提取称谓
    data['Title'] = data['Name'].apply(extract_title)

    # 填充缺失值
    fill_missing_age(data, title_medians, age_median, log_prefix)
    fill_missing_fare(data, fare_median, log_prefix)

    # 年龄分桶
    data['Age_Group'] = data['Age'].apply(age_group)

    # 票价标准化
    if fit_scaler:
        scaler = scaler or StandardScaler()
        data['Fare_scaled'] = scaler.fit_transform(data[['Fare']])
    else:
        data['Fare_scaled'] = scaler.transform(data[['Fare']])

    return data[['Pclass', 'Sex', 'Fare_scaled', 'Age_Group']], scaler

def preprocess_features(df_train, df_test=None, scaler=None, age_median=None, fare_median=None, title_medians=None):
    """特征预处理主函数"""
    # 计算统计量（仅在首次调用时）
    if age_median is None:
        age_median = df_train['Age'].median()
    if fare_median is None:
        fare_median = df_train['Fare'].median()
    if title_medians is None:
        temp_df = df_train.copy()
        temp_df['Title'] = temp_df['Name'].apply(extract_title)
        title_medians = temp_df.groupby('Title')['Age'].median().to_dict()

    # 处理训练集
    X_train, scaler = preprocess_data(df_train, scaler, title_medians, age_median, fare_median, fit_scaler=True)

    # 处理测试集（如果提供）
    if df_test is not None:
        X_test, _ = preprocess_data(df_test, scaler, title_medians, age_median, fare_median, fit_scaler=False, log_prefix="测试集")
        return X_train, X_test, scaler, age_median, fare_median, title_medians

    return X_train, scaler, age_median, fare_median, title_medians

# ========== 数据划分与预处理 ==========
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Survived'])
X_train, X_test, scaler, age_median, fare_median, title_medians = preprocess_features(train_set, test_set)
y_train, y_test = train_set['Survived'], test_set['Survived']

# ========== 网格搜索最优参数 ==========
print("="*60 + "\n开始网格搜索...\n" + "="*60)
param_grid = {
    'C': [0.1, 1, 10, 20, 50, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print(f"\n最优参数: {grid_search.best_params_}")
print(f"最优交叉验证得分: {grid_search.best_score_:.4f}\n" + "="*60)

# ========== 评估模型性能 ==========
model = grid_search.best_estimator_
acc_train = model.score(X_train, y_train)
acc_test = model.score(X_test, y_test)
print(f'\nTraining: {acc_train:.4f}, Test: {acc_test:.4f}, Gap: {acc_train-acc_test:.4f}')
if acc_train - acc_test > 0.1:
    print('⚠️ 可能过拟合')

# ========== 训练最终集成模型 ==========
print(f"\n使用最优参数训练最终模型...")
X_train_full, scaler_full, age_median_full, fare_median_full, title_medians_full = preprocess_features(df)
y_full = df['Survived']

model_final = VotingClassifier(
    estimators=[
        ('svc', SVC(**grid_search.best_params_)),
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(random_state=42))
    ],
    voting='hard'
)
model_final.fit(X_train_full, y_full)
print("✅ 集成模型训练完成！")

# ========== 预测Kaggle测试集 ==========
_, X_test_kaggle, _, _, _, _ = preprocess_features(
    df, kaggle_testing, scaler_full, age_median_full, fare_median_full, title_medians_full
)
predictions = model_final.predict(X_test_kaggle)

# ========== 投票详情 ==========
vote_df = pd.DataFrame({'PassengerId': kaggle_testing['PassengerId']})
for name, estimator in model_final.named_estimators_.items():
    vote_df[f'{name}_vote'] = estimator.predict(X_test_kaggle)
vote_df['final_vote'] = predictions
print("\n投票示例（前5行）:")
print(vote_df.head())

# ========== 保存提交文件 ==========
submission = pd.DataFrame({
    'PassengerId': kaggle_testing['PassengerId'],
    'Survived': predictions
})
output_path = r"D:\文件资料\Downloads\kaggle\submission.csv"
submission.to_csv(output_path, index=False)

print(f"\n✅ 提交文件已生成: {output_path}")
print(f"📊 统计: 总数={len(submission)}, 存活={predictions.sum()}, 死亡={len(predictions)-predictions.sum()}")
print("\n前10行预览:")
print(submission.head(10))
