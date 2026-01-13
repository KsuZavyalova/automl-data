# examples/check_env.py
from automl_data.utils.dependencies import DependencyManager, print_dependency_status

def main():
    print("🔍 Scanning environment dependencies...")
    
    # Инициализация менеджера (он синглтон)
    manager = DependencyManager()
    
    # Печать отчета
    manager.print_status()
    
    # Программная проверка
    missing = manager.get_missing()
    
    if not missing:
        print("\n✨ All systems go! Ready for full AutoForge experience.")
    else:
        print(f"\n⚠️ Found {len(missing)} missing optional packages.")
        print("Basic functionality will work, but some adapters might be disabled.")

if __name__ == "__main__":
    main()