# examples/benchmark_text_pipeline.py
"""
Бенчмарк текстового пайплайна.

Сравнивает качество классификации на:
1. Сырых данных
2. Данных с минимальной предобработкой (для трансформеров)
3. Данных с полной предобработкой (для классических методов)
4. Данных с аугментацией и балансировкой

Датасеты:
- IMDB Reviews (sentiment analysis)
- SMS Spam Collection
- AG News (topic classification)
"""

import sys
import os
import time
import warnings
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_data import AutoForge

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-eng')


# ============================================================================
# ЗАГРУЗКА ДАТАСЕТОВ
# ============================================================================


def load_sms_spam() -> pd.DataFrame:
    """
    Загрузка SMS Spam датасета.
    """
    try:
        # Попробуем загрузить через URL
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        
        print("📥 Loading SMS Spam dataset...")
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Class distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Failed to load SMS Spam: {e}")
        return _create_fallback_spam()


def load_ag_news_sample(n_samples: int = 2000) -> pd.DataFrame:
    """
    Загрузка AG News датасета.
    """
    try:
        from datasets import load_dataset
        
        print("📥 Loading AG News dataset...")
        dataset = load_dataset("ag_news", split="train")
        
        df = pd.DataFrame({
            'text': dataset['text'][:n_samples],
            'label': dataset['label'][:n_samples]
        })
        
        # Преобразуем метки
        label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        df['label'] = df['label'].map(label_map)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Class distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except ImportError:
        print("⚠️  datasets library not found, using fallback data")
        return _create_fallback_news()


def _create_fallback_imdb() -> pd.DataFrame:
    """Создание тестовых данных если датасет недоступен."""
    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "I loved every minute of it. Highly recommend to everyone.",
        "One of the best films I've ever seen. Masterpiece!",
        "Brilliant performances by all actors. A must watch.",
        "An incredible journey from start to finish. Amazing!",
        "Perfect movie for the whole family. Very entertaining.",
        "The cinematography was stunning. Beautiful visuals.",
        "I was blown away by the plot twists. Excellent writing.",
        "This film exceeded all my expectations. Wonderful!",
        "A heartwarming story that will make you smile.",
    ] * 50
    
    negative_reviews = [
        "Terrible movie. Complete waste of time and money.",
        "I couldn't even finish watching it. So boring.",
        "The worst film I've seen this year. Awful acting.",
        "Don't waste your time on this garbage.",
        "Extremely disappointing. Expected much better.",
        "Poor storyline and terrible execution.",
        "I want my two hours back. Horrible experience.",
        "The acting was so bad it was painful to watch.",
        "Nothing good about this movie. Total disaster.",
        "Avoid at all costs. Worst movie ever made.",
    ] * 50
    
    df = pd.DataFrame({
        'text': positive_reviews + negative_reviews,
        'label': ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _create_fallback_spam() -> pd.DataFrame:
    """Создание тестовых данных для спама."""
    ham_messages = [
        "Hey, are you coming to the party tonight?",
        "Can we meet tomorrow for lunch?",
        "Thanks for sending me the report.",
        "I'll call you back in 5 minutes.",
        "Don't forget to pick up milk on your way home.",
        "The meeting has been rescheduled to 3 PM.",
        "Happy birthday! Hope you have a great day!",
        "Did you watch the game last night?",
        "Let me know when you arrive at the station.",
        "I finished the project, please review it.",
    ] * 80
    
    spam_messages = [
        "CONGRATULATIONS! You've won $1,000,000! Click here to claim!",
        "FREE iPhone 15! Reply YES to claim your prize NOW!",
        "URGENT: Your account has been compromised. Click link immediately!",
        "You have been selected for a special offer! Act now!",
        "WIN a free trip to Hawaii! Text WIN to 12345!",
        "Your loan has been approved! Call now for instant cash!",
        "Hot singles in your area waiting to meet you!",
        "Make money from home! $5000 per week guaranteed!",
        "Limited time offer! 90% discount on luxury watches!",
        "You're our lucky winner! Claim your prize before midnight!",
    ] * 20
    
    df = pd.DataFrame({
        'text': ham_messages + spam_messages,
        'label': ['ham'] * len(ham_messages) + ['spam'] * len(spam_messages)
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _create_fallback_news() -> pd.DataFrame:
    """Создание тестовых данных для новостей."""
    world_news = [
        "UN Security Council meets to discuss global peace initiatives.",
        "World leaders gather at G20 summit to address climate change.",
        "International trade agreements signed between major economies.",
    ] * 100
    
    sports_news = [
        "Championship finals draw record-breaking viewership numbers.",
        "Star player signs multi-million dollar contract extension.",
        "Olympic committee announces new sports for upcoming games.",
    ] * 100
    
    business_news = [
        "Stock markets reach all-time highs amid economic recovery.",
        "Major tech company announces quarterly earnings beat expectations.",
        "Central bank considers interest rate adjustment policy.",
    ] * 100
    
    tech_news = [
        "New AI breakthrough promises revolutionary applications.",
        "Smartphone manufacturer unveils next-generation device.",
        "Cybersecurity firm discovers critical vulnerability in software.",
    ] * 100
    
    df = pd.DataFrame({
        'text': world_news + sports_news + business_news + tech_news,
        'label': (['World'] * len(world_news) + 
                  ['Sports'] * len(sports_news) + 
                  ['Business'] * len(business_news) + 
                  ['Sci/Tech'] * len(tech_news))
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================================
# ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================================

def train_and_evaluate(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    vectorizer_params: Dict[str, Any] = None,
    model_name: str = "LogisticRegression"
) -> Dict[str, Any]:
    """
    Обучение и оценка модели.
    """
    vectorizer_params = vectorizer_params or {}
    
    # Векторизация
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        **vectorizer_params
    )
    
    X_train_vec = vectorizer.fit_transform(X_train.astype(str))
    X_test_vec = vectorizer.transform(X_test.astype(str))
    
    # Выбор модели
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "NaiveBayes":
        model = MultinomialNB()
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Обучение
    start_time = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    
    # Предсказание
    y_pred = model.predict(X_test_vec)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'train_time': train_time,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train_vec.shape[1],
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred)
    }


def run_experiment(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    dataset_name: str,
    test_size: float = 0.2
) -> pd.DataFrame:
    """
    Запуск полного эксперимента на датасете.
    """
    print("\n" + "=" * 80)
    print(f"🔬 ЭКСПЕРИМЕНТ: {dataset_name}")
    print("=" * 80)
    
    results = []
    
    # Фиксированный сплит для честного сравнения
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_column],
        random_state=42
    )
    
    X_test = test_df[text_column]
    y_test = test_df[target_column]
    
    print(f"\n📊 Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"📊 Classes: {df[target_column].nunique()}")
    print(f"📊 Class distribution (train): {train_df[target_column].value_counts().to_dict()}")
    
    # =========================================================================
    # 1. СЫРЫЕ ДАННЫЕ (без обработки)
    # =========================================================================
    print("\n" + "-" * 40)
    print("1️⃣  СЫРЫЕ ДАННЫЕ (raw)")
    print("-" * 40)
    
    metrics_raw = train_and_evaluate(
        train_df[text_column],
        X_test,
        train_df[target_column],
        y_test
    )
    
    print(f"   Accuracy:    {metrics_raw['accuracy']:.4f}")
    print(f"   F1 (macro):  {metrics_raw['f1_macro']:.4f}")
    print(f"   F1 (weighted): {metrics_raw['f1_weighted']:.4f}")
    print(f"   Train time:  {metrics_raw['train_time']:.2f}s")
    
    results.append({
        'experiment': 'Raw Data',
        'preprocessing': 'None',
        'augmentation': 'None',
        **{k: v for k, v in metrics_raw.items() if k not in ['y_pred', 'classification_report']}
    })
    
    # =========================================================================
    # 2. МИНИМАЛЬНАЯ ПРЕДОБРАБОТКА (для трансформеров)
    # =========================================================================
    print("\n" + "-" * 40)
    print("2️⃣  МИНИМАЛЬНАЯ ПРЕДОБРАБОТКА (minimal)")
    print("-" * 40)
    
    try:
        forge_minimal = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="minimal",
            text_augment=False,
            balance=False,
            verbose=False
        )
        
        result_minimal = forge_minimal.fit_transform(train_df.copy())
        processed_train_minimal = result_minimal.data
        
        # Применяем ту же предобработку к тесту (только препроцессинг)
        forge_test = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="minimal",
            text_augment=False,
            balance=False,
            verbose=False
        )
        result_test = forge_test.fit_transform(test_df.copy())
        processed_test = result_test.data
        
        metrics_minimal = train_and_evaluate(
            processed_train_minimal[text_column],
            processed_test[text_column],
            processed_train_minimal[target_column],
            processed_test[target_column]
        )
        
        print(f"   Accuracy:    {metrics_minimal['accuracy']:.4f}")
        print(f"   F1 (macro):  {metrics_minimal['f1_macro']:.4f}")
        print(f"   F1 (weighted): {metrics_minimal['f1_weighted']:.4f}")
        print(f"   Train size:  {metrics_minimal['train_size']}")
        
        results.append({
            'experiment': 'Minimal Preprocessing',
            'preprocessing': 'minimal',
            'augmentation': 'None',
            **{k: v for k, v in metrics_minimal.items() if k not in ['y_pred', 'classification_report']}
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # =========================================================================
    # 3. ПОЛНАЯ ПРЕДОБРАБОТКА (для классических методов)
    # =========================================================================
    print("\n" + "-" * 40)
    print("3️⃣  ПОЛНАЯ ПРЕДОБРАБОТКА (full)")
    print("-" * 40)
    
    try:
        forge_full = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="full",
            text_augment=False,
            balance=False,
            verbose=False
        )
        
        result_full = forge_full.fit_transform(train_df.copy())
        processed_train_full = result_full.data
        
        # Тест с полной предобработкой
        forge_test_full = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="full",
            text_augment=False,
            balance=False,
            verbose=False
        )
        result_test_full = forge_test_full.fit_transform(test_df.copy())
        processed_test_full = result_test_full.data
        
        metrics_full = train_and_evaluate(
            processed_train_full[text_column],
            processed_test_full[text_column],
            processed_train_full[target_column],
            processed_test_full[target_column]
        )
        
        print(f"   Accuracy:    {metrics_full['accuracy']:.4f}")
        print(f"   F1 (macro):  {metrics_full['f1_macro']:.4f}")
        print(f"   F1 (weighted): {metrics_full['f1_weighted']:.4f}")
        print(f"   Train size:  {metrics_full['train_size']}")
        
        results.append({
            'experiment': 'Full Preprocessing',
            'preprocessing': 'full',
            'augmentation': 'None',
            **{k: v for k, v in metrics_full.items() if k not in ['y_pred', 'classification_report']}
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # =========================================================================
    # 4. ПОЛНАЯ ПРЕДОБРАБОТКА + EDA АУГМЕНТАЦИЯ
    # =========================================================================
    print("\n" + "-" * 40)
    print("4️⃣  ПОЛНАЯ ПРЕДОБРАБОТКА + EDA АУГМЕНТАЦИЯ")
    print("-" * 40)
    
    try:
        forge_aug = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="full",
            text_augment=True,
            text_augment_factor=1.5,  # Увеличиваем в 1.5 раза
            text_augment_methods=["eda"],
            text_balance_classes=False,
            balance=False,
            verbose=False
        )
        
        result_aug = forge_aug.fit_transform(train_df.copy())
        processed_train_aug = result_aug.data
        
        metrics_aug = train_and_evaluate(
            processed_train_aug[text_column],
            processed_test_full[text_column],  # Тест без аугментации!
            processed_train_aug[target_column],
            processed_test_full[target_column]
        )
        
        print(f"   Accuracy:    {metrics_aug['accuracy']:.4f}")
        print(f"   F1 (macro):  {metrics_aug['f1_macro']:.4f}")
        print(f"   F1 (weighted): {metrics_aug['f1_weighted']:.4f}")
        print(f"   Train size:  {metrics_aug['train_size']} (augmented)")
        
        results.append({
            'experiment': 'Full + EDA Augmentation',
            'preprocessing': 'full',
            'augmentation': 'EDA (1.5x)',
            **{k: v for k, v in metrics_aug.items() if k not in ['y_pred', 'classification_report']}
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # =========================================================================
    # 5. ПОЛНАЯ ПРЕДОБРАБОТКА + БАЛАНСИРОВКА КЛАССОВ
    # =========================================================================
    print("\n" + "-" * 40)
    print("5️⃣  ПОЛНАЯ ПРЕДОБРАБОТКА + БАЛАНСИРОВКА")
    print("-" * 40)
    
    try:
        forge_balanced = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="full",
            text_augment=True,
            text_balance_classes=True,  # Балансировка!
            text_augment_methods=["eda", "synonym_wordnet"],
            balance=True,
            verbose=False
        )
        
        result_balanced = forge_balanced.fit_transform(train_df.copy())
        processed_train_balanced = result_balanced.data
        
        metrics_balanced = train_and_evaluate(
            processed_train_balanced[text_column],
            processed_test_full[text_column],
            processed_train_balanced[target_column],
            processed_test_full[target_column]
        )
        
        print(f"   Accuracy:    {metrics_balanced['accuracy']:.4f}")
        print(f"   F1 (macro):  {metrics_balanced['f1_macro']:.4f}")
        print(f"   F1 (weighted): {metrics_balanced['f1_weighted']:.4f}")
        print(f"   Train size:  {metrics_balanced['train_size']} (balanced)")
        print(f"   New class distribution: {processed_train_balanced[target_column].value_counts().to_dict()}")
        
        results.append({
            'experiment': 'Full + Class Balancing',
            'preprocessing': 'full',
            'augmentation': 'Balanced',
            **{k: v for k, v in metrics_balanced.items() if k not in ['y_pred', 'classification_report']}
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # =========================================================================
    # 6. ВСЕ МЕТОДЫ АУГМЕНТАЦИИ
    # =========================================================================
    print("\n" + "-" * 40)
    print("6️⃣  ПОЛНАЯ ПРЕДОБРАБОТКА + ВСЕ АУГМЕНТАЦИИ")
    print("-" * 40)
    
    try:
        forge_all = AutoForge(
            target=target_column,
            text_column=text_column,
            text_preprocessing_level="full",
            text_augment=True,
            text_augment_factor=2.0,
            text_augment_methods=["eda", "synonym_wordnet", "pronoun_to_noun"],
            text_balance_classes=False,
            balance=False,
            verbose=False
        )
        
        result_all = forge_all.fit_transform(train_df.copy())
        processed_train_all = result_all.data
        
        metrics_all = train_and_evaluate(
            processed_train_all[text_column],
            processed_test_full[text_column],
            processed_train_all[target_column],
            processed_test_full[target_column]
        )
        
        print(f"   Accuracy:    {metrics_all['accuracy']:.4f}")
        print(f"   F1 (macro):  {metrics_all['f1_macro']:.4f}")
        print(f"   F1 (weighted): {metrics_all['f1_weighted']:.4f}")
        print(f"   Train size:  {metrics_all['train_size']} (2x augmented)")
        
        results.append({
            'experiment': 'Full + All Augmentations (2x)',
            'preprocessing': 'full',
            'augmentation': 'All methods (2x)',
            **{k: v for k, v in metrics_all.items() if k not in ['y_pred', 'classification_report']}
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Создаём DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    return results_df


def print_summary(results_df: pd.DataFrame, dataset_name: str):
    """Вывод сводки результатов."""
    print("\n" + "=" * 80)
    print(f"📊 СВОДКА РЕЗУЛЬТАТОВ: {dataset_name}")
    print("=" * 80)
    
    # Сортируем по F1 macro
    results_df = results_df.sort_values('f1_macro', ascending=False)
    
    print("\n🏆 Рейтинг по F1 (macro):\n")
    for i, row in results_df.iterrows():
        emoji = "🥇" if results_df.index.get_loc(i) == 0 else "🥈" if results_df.index.get_loc(i) == 1 else "🥉" if results_df.index.get_loc(i) == 2 else "  "
        print(f"{emoji} {row['experiment']:<35} | "
              f"Acc: {row['accuracy']:.4f} | "
              f"F1: {row['f1_macro']:.4f} | "
              f"Size: {row['train_size']}")
    
    print("\n" + "-" * 80)
    
    # Улучшение относительно baseline
    baseline = results_df[results_df['experiment'] == 'Raw Data'].iloc[0]
    
    print("\n📈 Улучшение относительно сырых данных:\n")
    for i, row in results_df.iterrows():
        if row['experiment'] == 'Raw Data':
            continue
        
        acc_diff = (row['accuracy'] - baseline['accuracy']) * 100
        f1_diff = (row['f1_macro'] - baseline['f1_macro']) * 100
        
        acc_arrow = "↑" if acc_diff > 0 else "↓" if acc_diff < 0 else "="
        f1_arrow = "↑" if f1_diff > 0 else "↓" if f1_diff < 0 else "="
        
        print(f"   {row['experiment']:<35} | "
              f"Acc: {acc_arrow}{abs(acc_diff):+.2f}% | "
              f"F1: {f1_arrow}{abs(f1_diff):+.2f}%")


def main():
    """Главная функция."""
    print("=" * 80)
    print("🔬 БЕНЧМАРК ТЕКСТОВОГО ПАЙПЛАЙНА")
    print("=" * 80)
    print("\nСравнение качества классификации на разных уровнях обработки данных")
    
    all_results = {}
    
    # =========================================================================
    # ТЕСТ 1: SMS Spam Detection (бинарная классификация, дисбаланс)
    # =========================================================================
    try:
        df_spam = load_sms_spam()
        results_spam = run_experiment(
            df_spam,
            text_column='text',
            target_column='label',
            dataset_name='SMS Spam Detection'
        )
        all_results['SMS Spam'] = results_spam
        print_summary(results_spam, 'SMS Spam Detection')
    except Exception as e:
        print(f"❌ SMS Spam experiment failed: {e}")
    
    # =========================================================================
    # ТЕСТ 3: AG News (мультиклассовая классификация)
    # =========================================================================
    try:
        df_news = load_ag_news_sample(n_samples=1000)
        results_news = run_experiment(
            df_news,
            text_column='text',
            target_column='label',
            dataset_name='AG News Classification'
        )
        all_results['AG News'] = results_news
        print_summary(results_news, 'AG News Classification')
    except Exception as e:
        print(f"❌ AG News experiment failed: {e}")
    
    # =========================================================================
    # ОБЩАЯ СВОДКА
    # =========================================================================
    print("\n" + "=" * 80)
    print("🏁 ОБЩАЯ СВОДКА ПО ВСЕМ ДАТАСЕТАМ")
    print("=" * 80)
    
    for dataset_name, results in all_results.items():
        best = results.sort_values('f1_macro', ascending=False).iloc[0]
        baseline = results[results['experiment'] == 'Raw Data'].iloc[0]
        
        improvement = (best['f1_macro'] - baseline['f1_macro']) * 100
        
        print(f"\n📊 {dataset_name}:")
        print(f"   Лучший метод: {best['experiment']}")
        print(f"   F1 (macro): {best['f1_macro']:.4f}")
        print(f"   Улучшение: +{improvement:.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ БЕНЧМАРК ЗАВЕРШЁН")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()