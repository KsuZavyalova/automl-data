# AutoML Data Processing Framework 

**Turn 2 weeks of data science work into 2 lines of code!**

AutoML Data Processing Framework is an intelligent, production-ready library for automated data preparation for machine learning. It analyzes your data and automatically applies optimal preprocessing strategies.

## 🎯 Features

### 🤖 Intelligent Automation
- **Auto-detection** of data types (tabular, text, images)
- **Auto-selection** of optimal preprocessing strategies
- **Auto-balancing** of imbalanced datasets
- **Quality scoring** with actionable recommendations

### 🏗️ Production Ready
- **Full error handling** with graceful degradation
- **Comprehensive logging** and monitoring
- **Reproducible pipelines** with code generation
- **Type hints** and full mypy support

### 🔧 Flexible & Extensible
- **Unified interface** for all data types
- **Modular adapter system** - easy to extend
- **Deep configuration** with sensible defaults
- **Batch and streaming** support

## 🚀 Quick Start

### Basic Usage
```python
from automl_data import AutoForge
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Auto-process everything!
forge = AutoForge(target="price")
result = forge.fit_transform(df)

# Ready for ML!
X_train, X_test, y_train, y_test = result.get_splits()
print(f"Data quality: {result.quality_score:.0%}")

# Generate report
result.save_report("analysis.html")
```

### Advanced Usage
```python
# Text processing
forge = AutoForge(
    target="sentiment",
    text_column="review",
    balance=True,
    text_preprocessing_level="full"
)

# Image processing  
forge = AutoForge(
    target="class",
    image_column="path",
    image_dir="images/",
    augment=True
)
```

### Development installation
```bash
git clone https://github.com/yourusername/automl-data.git
cd automl-data
pip install -e ".[dev]"
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ --cov=automl_data --cov-report=html
```
