# examples/titanic_basic.py
"""
Базовый пример использования AutoForge на датасете Titanic.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from automl_data.core.forge import AutoForge
import seaborn as sns

df = sns.load_dataset("titanic")
df = df.drop(columns=["alive", "deck"])
target = "Survived"
"""
# 1. Загружаем данные
print("📥 Загрузка Titanic dataset...")
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame.copy()
"""
# Переименуем целевую переменную
df = df.rename(columns={'survived': 'Survived'})

print(f"Исходные данные: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(f"Пропуски: {df.isnull().sum().sum()}")

# 2. Создаем AutoForge
print("\n🔧 Создание AutoForge...")
forge = AutoForge(
    target=target,
    task="auto", 
    balance=True,
    verbose=True
)

# 3. Обрабатываем данные (fit + transform в одном вызове)
print("\n🔄 Обработка данных...")
result = forge.fit_transform(df)

# 4. Анализируем результат
print("\n" + "="*50)
print("📊 РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
print("="*50)
print(f"Обработано строк: {result.shape[0]:,}")
print(f"Колонок после обработки: {result.shape[1]}")
print(f"Качество данных: {result.quality_score:.0%}")
print(f"Выполнено шагов: {result.steps}")

# 5. Получаем train/test сплиты
print("\n📈 Train/Test сплиты:")
X_train, X_test, y_train, y_test = result.get_splits()
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# 6. Сохраняем отчет
print("\n💾 Сохранение отчета...")
result.save_report("titanic_report.html")
print("✅ Отчет сохранен: titanic_report.html")

# 7. Показываем пример обработанных данных
print("\n📋 Пример обработанных данных:")
print(X_train.head())

# 8. Можно обучить простую модель
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n🤖 Обучение модели...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Точность модели: {accuracy:.3f}")

# 9. Анализ важности признаков
print("\n🎯 Важные признаки:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(16).to_string())