import seaborn as sns
from automl_data.core.forge import AutoForge

# Diamonds — лучший датасет для тестирования
df = sns.load_dataset("diamonds")

target = "price"

forge = AutoForge(
    target=target,
    task="regression",
    impute_strategy="auto",
    encode_strategy="auto",
    scaling="auto",
    outlier_method="iqr",
    balance=False,
    verbose=True
)

result = forge.fit_transform(df)

print("\n=== RESULT SUMMARY ===")
print(result.summary())
print("\nProcessed DataFrame shape:", result.shape)

X_train, X_test, y_train, y_test = result.get_splits()
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

result.save_report("diamonds_report.html")
print("\nHTML report saved to diamonds_report.html")
