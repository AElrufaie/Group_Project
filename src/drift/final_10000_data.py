import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# ---------- Split by Index (Last N as Current) ----------

def split_by_index(df: pd.DataFrame, curr_size: int = 10000):
    """
    Split DataFrame so that the last `curr_size` rows are the current set,
    and the preceding rows are the reference set.
    """
    if curr_size >= len(df):
        raise ValueError("curr_size must be smaller than total rows")
    ref = df.iloc[:-curr_size].reset_index(drop=True)
    curr = df.iloc[-curr_size:].reset_index(drop=True)
    # label them
    ref['__drift_label__'] = 0
    curr['__drift_label__'] = 1
    combined = pd.concat([ref, curr], ignore_index=True)
    return ref, curr, combined

# ---------- Model-based Drift Detection ----------

def detect_model_drift(
    combined: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    threshold: float = 0.55,
    test_size: float = 0.3,
    random_state: int = 42
) -> dict:
    # 数值特征矩阵
    X_num = combined[numerical_features].fillna(0).values
    # 类别特征独热编码
    X_cat = combined[categorical_features].fillna('NA')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_enc = encoder.fit_transform(X_cat)
    # 合并两部分
    X = np.hstack([X_num, X_cat_enc])
    y = combined['__drift_label__'].values

    # 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 随机森林分类
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # 用 AUC 判断
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
        metric_val = roc_auc_score(y_test, y_prob)
        metric_name = 'roc_auc'
    else:
        y_pred = clf.predict(X_test)
        metric_val = accuracy_score(y_test, y_pred)
        metric_name = 'accuracy'

    drift_flag = metric_val > threshold
    return {
        'metric_name': metric_name,
        'metric_value': metric_val,
        'threshold': threshold,
        'drift': drift_flag
    }

# ---------- Main: 按索引检测最后10000条的漂移 ----------

def main():
    # 1) 读数据
    df = pd.read_csv(
        r"C:\Users\Kevin\Desktop\MLE Project - Fatih\Group_Project_IS2\data\animal_df_clean.csv"
    )
    # 2) 定义特征
    numerical = ["age_days_outcome", "los_at_shelter"]
    categorical = [
        "age_group_intake", "is_fixed", "breed_type",
        "color_group", "intake_condition_group", "month_of_outcome"
    ]
    # 3) 按索引划分
    ref, curr, combined = split_by_index(df, curr_size=10000)
    print(f"Reference rows: {len(ref)}, Current rows: {len(curr)}")

    # 4) 检测漂移
    drift_info = detect_model_drift(
        combined,
        numerical_features=numerical,
        categorical_features=categorical,
        threshold=0.65
    )

    # 5) 输出结果
    print("=== Model-based Drift Detection on Last 10000 Rows ===")
    print(f"Metric ({drift_info['metric_name']}): {drift_info['metric_value']:.4f}")
    print(f"Threshold: {drift_info['threshold']}")
    print("Drift detected!" if drift_info['drift'] else "No drift detected.")

if __name__ == "__main__":
    main()
