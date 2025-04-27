import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Plot ROC Curve for a single model
def plot_roc_curve(model, X_test, y_test, class_mapping, model_name="Model"):
    y_score = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    fig = px.line()
    for i in range(n_classes):
        class_label = class_mapping.get(i, f"Class {i}")
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_scatter(x=fpr, y=tpr, name=f"ROC curve of {class_label} (AUC = {roc_auc:.2f})")

    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(
        title_text=f"ROC Curves for {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )
    fig.show()

# Plot Feature Importances for Tree-based models
def plot_feature_importances(model, X_train, model_name="Model"):
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"Model {model_name} does not have feature_importances_. Skipping.")
        return

    feature_names = X_train.columns
    importance_df = (pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                     .sort_values(by='Importance', ascending=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importances for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# Plot ROC Curves for multiple models together
def plot_combined_roc_curves(models, X_test, y_test, class_mapping):
    fig = go.Figure()
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    for model_name, model in models.items():
        y_probs = model.predict_proba(X_test)
        for i in range(n_classes):
            class_label = class_mapping.get(i, f"Class {i}")
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
            roc_auc_score = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'{model_name} - {class_label} (AUC = {roc_auc_score:.2f})'
            ))

    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(
        title_text="Combined ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )
    fig.show()

# Plot SHAP Summary Plot
def shap_summary_plot(model, X_train, model_name="Model"):
    print(f"Calculating SHAP values for {model_name}...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=True)
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")