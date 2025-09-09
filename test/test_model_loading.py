#!/usr/bin/env python3
"""
Test script to debug model loading issues.
This helps identify problems with model serialization/deserialization.
"""

import os
import sys
import tempfile
import joblib
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

def test_model_compatibility():
    """Test if training and serving models are compatible"""
    print("=== Testing Model Compatibility ===")
    
    try:
        # Test imports that server uses
        from ml_pipelines_kfp.iris_xgboost.models.instance import Instance
        from ml_pipelines_kfp.iris_xgboost.models.prediction import Prediction
        print("✓ Model classes imported successfully")
        
        # Test creating instances
        test_instance = Instance(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2
        )
        print("✓ Instance model works")
        
        # Test prediction model
        test_prediction = Prediction(
            class_=0,
            class_probabilities=[0.8, 0.1, 0.1]
        )
        print("✓ Prediction model works")
        
        return True
        
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_joblib_versions():
    """Test joblib version compatibility"""
    print("\n=== Testing Joblib Compatibility ===")
    
    try:
        import joblib
        import sklearn
        import numpy
        import pandas
        
        print(f"joblib version: {joblib.__version__}")
        print(f"sklearn version: {sklearn.__version__}")
        print(f"numpy version: {numpy.__version__}")
        print(f"pandas version: {pandas.__version__}")
        
        # Create test model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        model = RandomForestClassifier(random_state=42)
        model.fit(iris.data, iris.target)
        
        # Test serialization/deserialization
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            # Save model
            joblib.dump(model, model_path)
            print("✓ Model saved successfully")
            
            # Load model
            loaded_model = joblib.load(model_path)
            print("✓ Model loaded successfully")
            
            # Test prediction
            prediction = loaded_model.predict(iris.data[:1])
            probabilities = loaded_model.predict_proba(iris.data[:1])
            
            print(f"✓ Prediction: {prediction[0]}")
            print(f"✓ Probabilities shape: {probabilities.shape}")
            
            return True
            
    except Exception as e:
        print(f"✗ Joblib compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_fsspec_gcs():
    """Test fsspec and gcsfs for model loading from GCS"""
    print("\n=== Testing GCS Model Loading ===")
    
    try:
        import fsspec
        import gcsfs
        
        print(f"fsspec version: {fsspec.__version__}")
        print(f"gcsfs version: {gcsfs.__version__}")
        
        # Test fsspec.open with local file (simulating GCS)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.joblib"
            
            # Create and save test model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import load_iris
            
            iris = load_iris()
            model = RandomForestClassifier(random_state=42)
            model.fit(iris.data, iris.target)
            joblib.dump(model, model_path)
            
            # Test loading with fsspec (like server does)
            with fsspec.open(str(model_path), "rb") as f:
                loaded_model = joblib.load(f)
            
            print("✓ Model loaded via fsspec successfully")
            
            # Test prediction
            prediction = loaded_model.predict(iris.data[:1])
            print(f"✓ Prediction after fsspec load: {prediction[0]}")
            
            return True
            
    except Exception as e:
        print(f"✗ GCS loading test failed: {e}")
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test environment variable handling"""
    print("\n=== Testing Environment Variables ===")
    
    # Test AIP_STORAGE_URI handling
    test_uri = "gs://test-bucket/model-path"
    os.environ["AIP_STORAGE_URI"] = test_uri
    
    try:
        from ml_pipelines_kfp.iris_xgboost.constants import MODEL_FILENAME
        
        expected_model_uri = f"{test_uri}/{MODEL_FILENAME}"
        print(f"✓ Model URI would be: {expected_model_uri}")
        
        # Test if the server would construct the right path
        model_uri = f'{os.environ["AIP_STORAGE_URI"]}/{MODEL_FILENAME}'
        print(f"✓ Server would use: {model_uri}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        traceback.print_exc()
        return False

def analyze_pipeline_model():
    """Analyze a real model from pipeline if available"""
    print("\n=== Analyzing Pipeline Model ===")
    
    # Look for pipeline-generated models
    possible_paths = [
        "model.joblib",
        "pipeline_outputs/model.joblib",
        "artifacts/model.joblib"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                print(f"Found model at: {path}")
                model = joblib.load(path)
                
                print(f"Model type: {type(model)}")
                print(f"Model attributes: {dir(model)}")
                
                if hasattr(model, 'feature_names_in_'):
                    print(f"Expected features: {model.feature_names_in_}")
                
                if hasattr(model, 'classes_'):
                    print(f"Model classes: {model.classes_}")
                
                # Test with sample data
                import numpy as np
                sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
                
                prediction = model.predict(sample_data)
                probabilities = model.predict_proba(sample_data)
                
                print(f"✓ Sample prediction: {prediction[0]}")
                print(f"✓ Sample probabilities: {probabilities[0]}")
                
                return True
                
            except Exception as e:
                print(f"✗ Error analyzing model at {path}: {e}")
                continue
    
    print("No pipeline models found to analyze")
    return False

if __name__ == "__main__":
    print("Model Loading Debug Tests")
    print("=" * 40)
    
    tests = [
        test_model_compatibility,
        test_joblib_versions, 
        test_fsspec_gcs,
        test_environment_variables,
        analyze_pipeline_model
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test.__name__}")
    
    if all(results):
        print("\n✓ All tests passed! Model loading should work.")
    else:
        print("\n✗ Some tests failed. Fix issues before deployment.")