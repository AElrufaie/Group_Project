import os
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import matplotlib.pyplot as plt

# ———— 特征列表定义 ————
numerical = ["age_days_outcome", "los_at_shelter"]
categorical = [
    "age_group_intake", "is_fixed", "breed_type",
    "color_group", "intake_condition_group", "month_of_outcome"
]
FEATURES = numerical + categorical
TARGET = 'outcome_group'

# 输出图像目录
GRAPH_DIR = r'C:\Users\Kevin\Documents\GitHub\Group_Project_IS2\graph'
# 确保目录存在
os.makedirs(GRAPH_DIR, exist_ok=True)

# ———— 1. 加载并随机抽样数据 ————
def load_and_sample(path: str, sample_size: int = 10000, random_state: int = 42):
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df

# ———— 2. 编码特征并返回稀疏矩阵与列名 ————
def encode_features(df: pd.DataFrame, numerical: list, categorical: list):
    # 数值特征
    X_num = df[numerical].fillna(0).values
    X_num_sparse = sparse.csr_matrix(X_num)
    # 类别特征稀疏 One-Hot，drop first
    X_cat = df[categorical].fillna('NA')
    encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')
    X_cat_sparse = encoder.fit_transform(X_cat)
    # 合并稀疏矩阵
    X_sparse = sparse.hstack([X_num_sparse, X_cat_sparse], format='csr')
    # 列名
    cat_cols = encoder.get_feature_names_out(categorical)
    columns = list(numerical) + list(cat_cols)
    return X_sparse, columns

# ———— 3. 初始化 SHAP 解释器 ————
def get_shap_explainer(model, X_train: pd.DataFrame, background_size: int = 100):
    background = shap.sample(X_train, background_size)
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.KernelExplainer(model.predict, background)

# ———— 4. 计算 SHAP 值 ————
def get_shap_values(explainer, X: pd.DataFrame):
    return explainer.shap_values(X)

# ———— 5. 特征重要性汇总 ————
def get_shap_importance_df(shap_values, X: pd.DataFrame):
    if isinstance(shap_values, list):
        abs_vals = sum([np.abs(vals) for vals in shap_values])
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        abs_vals = np.mean(np.abs(shap_values), axis=2)
    else:
        abs_vals = np.abs(shap_values)
    abs_df = pd.DataFrame(abs_vals, columns=X.columns)
    importance = abs_df.mean().reset_index()
    importance.columns = ['feature', 'mean_abs_shap']
    return importance.sort_values('mean_abs_shap', ascending=False)

# ———— 6. 绘图函数 ————
def plot_shap_summary(shap_values, X: pd.DataFrame, show: bool = True):
    plot_vals = shap_values
    if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        plot_vals = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    shap.summary_plot(plot_vals, X, show=show)


def plot_shap_dependence(feature: str, shap_values, X: pd.DataFrame, show: bool = True):
    plot_vals = shap_values
    if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        plot_vals = shap_values[:, :, 0]
    shap.dependence_plot(feature, plot_vals, X, show=show)

# ———— 7. 主流程 ————
def main():
    # 1) 加载并抽样
    path = r'C:\Users\Kevin\Documents\GitHub\Group_Project_IS2\data\animal_df_clean.csv'
    df = load_and_sample(path, sample_size=10000)

    # 2) 编码特征
    X_sparse, columns = encode_features(df, numerical, categorical)
    X = pd.DataFrame(X_sparse.toarray(), columns=columns)
    y = df[TARGET]

    # 3) 训练模型
    params = {
        'n_estimators': 127,
        'max_depth': 13,
        'min_samples_split': 5,
        'min_samples_leaf': 4
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X, y)

    # 4) SHAP 分析
    explainer = get_shap_explainer(model, X, background_size=100)
    shap_values = get_shap_values(explainer, X)
    importance_df = get_shap_importance_df(shap_values, X)

    # 5) 输出及可视化
    print('Global feature importance:')
    print(importance_df)

    # 保存 Summary Plot
    plot_shap_summary(shap_values, X, show=False)
    summary_path = os.path.join(GRAPH_DIR, 'shap_summary_randomforest in stacking.png')
    plt.savefig(summary_path, bbox_inches='tight')
    plt.clf()

    # 保存 Dependence Plot
    top_feat = importance_df.iloc[0]['feature']
    plot_shap_dependence(top_feat, shap_values, X, show=False)
    dep_path = os.path.join(GRAPH_DIR, f'shap_dependence_{top_feat}_randomforest in stacking.png')
    plt.savefig(dep_path, bbox_inches='tight')
    plt.clf()

    print(f'Saved summary plot to {summary_path}')
    print(f'Saved dependence plot to {dep_path}')

if __name__ == '__main__':
    main()
