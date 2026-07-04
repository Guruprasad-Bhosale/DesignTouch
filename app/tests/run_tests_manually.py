import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.tests.test_systems import (
    test_config_service,
    test_storage_service,
    test_tracking_smoothing_and_debounce,
    test_feature_extractor,
    test_language_processor,
    test_classifier_heuristics
)

class TempPath:
    def __init__(self, path):
        self.path = path
    def __truediv__(self, other):
        return os.path.join(self.path, other)

if __name__ == "__main__":
    print("--- RUNNING CORE SYSTEMS TESTS MANUALLY ---")
    
    # Setup clean temp folder
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_test_run")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    tp = TempPath(temp_dir)
    
    try:
        print("1. Running test_config_service...")
        test_config_service(tp)
        print("[PASS] test_config_service")
        
        print("2. Running test_storage_service...")
        test_storage_service(tp)
        print("[PASS] test_storage_service")
        
        print("3. Running test_tracking_smoothing_and_debounce...")
        test_tracking_smoothing_and_debounce()
        print("[PASS] test_tracking_smoothing_and_debounce")
        
        print("4. Running test_feature_extractor...")
        test_feature_extractor()
        print("[PASS] test_feature_extractor")
        
        print("5. Running test_language_processor...")
        test_language_processor()
        print("[PASS] test_language_processor")
        
        print("6. Running test_classifier_heuristics...")
        test_classifier_heuristics()
        print("[PASS] test_classifier_heuristics")
        
        print("\n=== ALL CORE SYSTEMS TESTS PASSED SUCCESSFULLY ===")
        
        # Clean up temp
        shutil.rmtree(temp_dir)
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[FAIL] Test run encountered error: {e}")
        traceback.print_exc()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        sys.exit(1)
