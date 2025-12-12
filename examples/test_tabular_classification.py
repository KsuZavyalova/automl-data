import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from automl_data import AutoForge, ProfilerAdapter

df = sns.load_dataset("titanic")
df = df.drop(columns=["alive", "deck"])
target = "survived"

from automl_data.core.container import DataContainer
container = DataContainer(data=df)
# Анализ сырых данных БЕЗ обработки
profiler = ProfilerAdapter()
result = profiler.fit_transform(container)
profiler.save_report("raw_data_analysis.html")

processed = AutoForge(target=target, task="classification")
processed_df = processed.fit_transform(df)

X_train_cl, X_test_cl, y_train_cl, y_test_cl = processed_df.get_splits()

model_cl = LogisticRegression(max_iter=100)
model_cl.fit(X_train_cl, y_train_cl)

cl_pred = model_cl.predict(X_test_cl)
cl_proba = model_cl.predict_proba(X_test_cl)[:, 1]

cl_acc = accuracy_score(y_test_cl, cl_pred)
cl_f1 = f1_score(y_test_cl, cl_pred)
cl_auc = roc_auc_score(y_test_cl, cl_proba)

print("\n=== METRICS===")
print("Accuracy:", round(cl_acc, 4))
print("F1:", round(cl_f1, 4))
print("ROC-AUC:", round(cl_auc, 4))

processed_df.save_report('titanic_report.html')