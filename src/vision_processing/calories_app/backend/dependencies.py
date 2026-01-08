from .core.ai_engine import NutrientScannerMultiView

# Singleton instance
try:
    scanner = NutrientScannerMultiView()
except Exception as e:
    print(f"❌ Erreur Init AI: {e}")
    scanner = None

def get_scanner():
    return scanner
