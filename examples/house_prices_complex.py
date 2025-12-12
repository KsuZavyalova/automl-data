# examples/house_prices_complex.py
"""
Пример обработки сложного датасета House Prices.
С Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from automl_data.core.forge import AutoForge
from automl_data.core.config import TabularConfig

def load_house_prices_data():
    """
    Загрузка и предварительный анализ датасета House Prices.
    Если файл не найден, скачиваем с Kaggle API.
    """
    import os
    
    train_path = '/Users/kseniazavyalova/Downloads/house-prices-advanced-regression-techniques/train.csv'
    
    if os.path.exists(train_path):
        print(f"📂 Загрузка из {train_path}")
        df = pd.read_csv(train_path)

    return df


def analyze_dataset_complexity(df, target_col='SalePrice'):
    """Анализ сложности датасета"""
    print("\n" + "="*70)
    print("🔍 АНАЛИЗ СЛОЖНОСТИ ДАТАСЕТА")
    print("="*70)
    
    print(f"Размер: {df.shape[0]:,} строк × {df.shape[1]} колонок")
    print(f"Целевая переменная: {target_col}")
    
    # Типы данных
    dtypes = df.dtypes.value_counts()
    print(f"\n📊 Типы данных:")
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count} колонок")
    
    # Пропуски
    missing_total = df.isnull().sum().sum()
    missing_cols = (df.isnull().sum() > 0).sum()
    missing_percent = (missing_total / (df.shape[0] * df.shape[1])) * 100
    
    print(f"\n❌ Пропуски:")
    print(f"  Всего пропусков: {missing_total:,}")
    print(f"  Колонок с пропусками: {missing_cols}")
    print(f"  Процент пропусков: {missing_percent:.1f}%")
    
    # Самые проблемные колонки
    missing_by_col = df.isnull().sum().sort_values(ascending=False)
    print(f"\n🔥 Топ-10 колонок с пропусками:")
    for col, count in missing_by_col.head(10).items():
        percent = (count / len(df)) * 100
        print(f"  {col:25} {count:4} ({percent:5.1f}%)")
    
    # Категориальные переменные
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"\n🎭 Категориальные переменные: {len(categorical_cols)}")
    
    # Анализ кардинальности
    print(f"\n📈 Кардинальность категориальных переменных:")
    for col in categorical_cols[:5]:
        n_unique = df[col].nunique()
        print(f"  {col:25} {n_unique:3} уникальных значений")
    
    # Распределение целевой переменной
    print(f"\n🎯 Распределение целевой переменной ({target_col}):")
    target = df[target_col]
    print(f"  Тип: {target.dtype}")
    print(f"  Минимум: {target.min():,.0f}")
    print(f"  Максимум: {target.max():,.0f}")
    print(f"  Медиана: {target.median():,.0f}")
    print(f"  Среднее: {target.mean():,.0f}")
    print(f"  Стандартное отклонение: {target.std():,.0f}")
    
    # Проверка на выбросы в целевой
    from scipy import stats
    skewness = target.skew()
    kurtosis = target.kurtosis()
    print(f"  Асимметрия: {skewness:.2f} {'⚠️ (сильная асимметрия)' if abs(skewness) > 1 else ''}")
    print(f"  Эксцесс: {kurtosis:.2f}")
    
    # Проверка мультиколлинеарности (быстрая)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = (corr_matrix > 0.8).sum().sum() - len(numeric_cols)  # Исключаем диагональ
        print(f"\n🔗 Потенциальная мультиколлинеарность:")
        print(f"  Корреляций > 0.8: {high_corr}")
    
    return df

def test_auto_forge_complex(df, target_col='SalePrice'):
    """Тестирование AutoForge на сложном датасете"""
    print("\n" + "="*70)
    print("🚀 ТЕСТИРОВАНИЕ AUTOFORGE НА СЛОЖНОМ ДАТАСЕТЕ")
    print("="*70)
    
    # 1. Базовый тест с автонастройками
    print("\n1. 📊 Базовый AutoForge (полностью автоматический):")
    forge_basic = AutoForge(
        target=target_col,
        task='auto',  # Явно указываем регрессию
        balance=True,  # Для регрессии балансировка не нужна
        verbose=True
    )
    
    try:
        result_basic = forge_basic.fit_transform(df)
        
        print(f"\n   ✅ Результат базовой обработки:")
        print(f"      • Исходный размер: {df.shape}")
        print(f"      • Финальный размер: {result_basic.shape}")
        print(f"      • Качество данных: {result_basic.quality_score:.0%}")
        print(f"      • Шаги обработки: {result_basic.steps}")
        print(f"      • Время выполнения: {result_basic.execution_time:.2f} сек")
        
        # Проверяем, что target не потерялся
        assert result_basic.y is not None, "❌ Target потерян!"
        assert len(result_basic.y) == len(result_basic.data), "❌ Target не синхронизирован!"
        
        # Проверяем пропуски
        missing_after = result_basic.data.isnull().sum().sum()
        print(f"      • Пропусков после обработки: {missing_after}")
        
    except Exception as e:
        print(f"   ❌ Ошибка в базовом режиме: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Продвинутый тест с кастомной конфигурацией
    print("\n" + "="*70)
    print("2. ⚙️ Продвинутый AutoForge (кастомная конфигурация):")
    
    # Создаем кастомную конфигурацию для сложных данных
    tabular_config = TabularConfig(
        impute_strategy='iterative',  # Итеративная импьютация для сложных пропусков
        scaling='robust',  # RobustScaler из-за выбросов
        encode_strategy='target',  # Target encoding для высокой кардинальности
        max_onehot_cardinality=20,  # Повышаем порог для one-hot
        outlier_method='isolation_forest',  # Изоляционный лес для сложных выбросов
        outlier_action='clip'  # Отсечение вместо удаления
    )
    
    forge_advanced = AutoForge(
        target=target_col,
        task='regression',
        tabular_config=tabular_config,
        test_size=0.2,
        random_state=42,
        verbose=True
    )
    
    try:
        result_advanced = forge_advanced.fit_transform(df)
        
        print(f"\n   ✅ Результат продвинутой обработки:")
        print(f"      • Финальный размер: {result_advanced.shape}")
        print(f"      • Качество данных: {result_advanced.quality_score:.0%}")
        print(f"      • Шаги обработки: {result_advanced.steps}")
        
        # Детальный анализ
        print(f"\n   🔍 Детальный анализ:")
        
        # Анализ типов признаков после обработки
        X = result_advanced.X
        numeric_count = X.select_dtypes(include=[np.number]).shape[1]
        print(f"      • Числовых признаков: {numeric_count}")
        
        # Проверка распределения целевой переменной
        if result_advanced.y is not None:
            y = result_advanced.y
            print(f"      • Target: min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.0f}")
            
            # Проверка логарифмирования (часто нужно для цен)
            skew_before = df[target_col].skew()
            skew_after = y.skew()
            print(f"      • Асимметрия target: до={skew_before:.2f}, после={skew_after:.2f}")
        
        # Анализ рекомендаций
        print(f"\n   💡 Рекомендации AutoForge:")
        for i, rec in enumerate(result_advanced.recommendations[:5]):
            print(f"      {i+1}. {rec.get('type', 'info')}: {rec}")
        
        # Сохраняем отчет
        result_advanced.save_report("house_prices_report.html")
        print(f"\n   💾 Отчет сохранен: house_prices_report.html")
        
        # Сохраняем обработанные данные
        result_advanced.data.to_csv("house_prices_processed.csv", index=False)
        print(f"   💾 Данные сохранены: house_prices_processed.csv")
        
    except Exception as e:
        print(f"   ❌ Ошибка в продвинутом режиме: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Тестирование на модели регрессии
    print("\n" + "="*70)
    print("3. 🤖 ТЕСТИРОВАНИЕ НА МОДЕЛИ РЕГРЕССИИ")
    print("="*70)
    
    if 'result_advanced' in locals() and result_advanced.y is not None:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import matplotlib.pyplot as plt
        
        # Получаем данные
        X, y = result_advanced.X, result_advanced.y
        
        # Разделяем
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n   📈 Размеры данных для модели:")
        print(f"      • X_train: {X_train.shape}")
        print(f"      • X_test: {X_test.shape}")
        print(f"      • y_train: {y_train.shape}")
        print(f"      • y_test: {y_test.shape}")
        
        # Обучаем модель
        print(f"\n   🔧 Обучение RandomForestRegressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n   📊 Метрики регрессии:")
        print(f"      • MSE:  {mse:,.0f}")
        print(f"      • RMSE: {rmse:,.0f}")
        print(f"      • MAE:  {mae:,.0f}")
        print(f"      • R²:   {r2:.3f}")
        
        # Анализ важности признаков
        print(f"\n   🎯 Топ-10 важных признаков:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in enumerate(feature_importance.head(10).itertuples()):
            print(f"      {i+1:2}. {row.feature:30} {row.importance:.4f}")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Сравнение предсказаний
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Фактические значения')
        axes[0, 0].set_ylabel('Предсказанные значения')
        axes[0, 0].set_title(f'Предсказания vs Факт (R²={r2:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ошибки предсказаний
        errors = y_pred - y_test
        axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Ошибка предсказания')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].set_title('Распределение ошибок')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Важность признаков
        top_features = feature_importance.head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'].values)
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'].values)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('Важность')
        axes[1, 0].set_title('Топ-15 важных признаков')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Распределение целевой переменной
        axes[1, 1].hist(y, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Цена (SalePrice)')
        axes[1, 1].set_ylabel('Частота')
        axes[1, 1].set_title('Распределение целевой переменной')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Анализ регрессионной модели на House Prices', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('house_prices_regression_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n   📊 Визуализация сохранена: house_prices_regression_analysis.png")




def main():
    """Основная функция"""
    print("="*70)
    print("🏠 АВТОМАТИЧЕСКАЯ ОБРАБОТКА СЛОЖНОГО ДАТАСЕТА HOUSE PRICES")
    print("="*70)
    
    # 1. Загрузка данных
    df = load_house_prices_data()
    
    # 2. Анализ сложности
    df = analyze_dataset_complexity(df)
    
    # 3. Тестирование AutoForge
    test_auto_forge_complex(df)
    
    print("\n" + "="*70)
    print("✅ ЗАВЕРШЕНО! Библиотека успешно обработала сложный датасет.")
    print("="*70)

if __name__ == "__main__":
    main()