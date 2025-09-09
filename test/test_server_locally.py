#!/usr/bin/env python3
"""
Local testing script to debug FastAPI server startup issues.
This simulates the Vertex AI deployment environment.
"""

import os
import sys
import tempfile
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, "src")

def create_test_model():
    """Create a test model file to simulate what would be stored in GCS"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model

def test_server_startup():
    """Test server startup with a mock model"""
    print("=== Testing FastAPI Server Startup ===")
    
    # Create temporary model file
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.joblib"
        
        # Create and save test model
        test_model = create_test_model()
        joblib.dump(test_model, model_path)
        
        # Set environment variable that server expects
        os.environ["AIP_STORAGE_URI"] = str(temp_dir)
        
        try:
            # Import the server module
            from ml_pipelines_kfp.iris_xgboost.server import build_app, init_model
            
            print("✓ Server module imported successfully")
            
            # Test model initialization
            try:
                init_model()
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"✗ Model loading failed: {e}")
                return False
            
            # Test app creation
            try:
                app = build_app()
                print("✓ FastAPI app created successfully")
            except Exception as e:
                print(f"✗ App creation failed: {e}")
                return False
            
            return True
            
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    print("\n=== Testing Prediction Endpoint ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.joblib"
        test_model = create_test_model()
        joblib.dump(test_model, model_path)
        
        os.environ["AIP_STORAGE_URI"] = str(temp_dir)
        
        try:
            from ml_pipelines_kfp.iris_xgboost.server import app, init_model
            from ml_pipelines_kfp.iris_xgboost.models.instance import Instance
            
            # Initialize model
            init_model()
            
            # Create test instance
            test_instance = Instance(
                sepal_length=5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2
            )
            
            # Test prediction logic
            from ml_pipelines_kfp.iris_xgboost.server import MODELS
            
            if "best_model" in MODELS:
                model = MODELS["best_model"]
                df = pd.DataFrame([test_instance.model_dump()])
                prediction = model.predict(df)
                probabilities = model.predict_proba(df)
                
                print(f"✓ Prediction: {prediction[0]}")
                print(f"✓ Probabilities: {probabilities[0]}")
                return True
            else:
                print("✗ Model not found in MODELS dictionary")
                return False
                
        except Exception as e:
            print(f"✗ Prediction test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_with_docker():
    """Instructions for testing with Docker"""
    print("\n=== Docker Testing Instructions ===")
    print("""
To test the exact deployment environment:

1. Build the Docker image:
   docker build -t ml-pipelines-test .

2. Create a test model file:
   mkdir -p /tmp/test-model
   # Copy a real model.joblib from your pipeline outputs to /tmp/test-model/

3. Run the container with model volume:
   docker run -p 8080:8080 \\
     -e AIP_STORAGE_URI=/app/model \\
     -v /tmp/test-model:/app/model \\
     ml-pipelines-test

4. Test the endpoints:
   curl http://localhost:8080/health/live
   
   curl -X POST http://localhost:8080/predict \\
     -H "Content-Type: application/json" \\
     -d '{
       "instances": [{
         "sepal_length": 5.1,
         "sepal_width": 3.5, 
         "petal_length": 1.4,
         "petal_width": 0.2
       }]
     }'
""")

if __name__ == "__main__":
    print("FastAPI Server Local Testing")
    print("=" * 40)
    
    # Test 1: Server startup
    startup_success = test_server_startup()
    
    # Test 2: Prediction endpoint  
    if startup_success:
        prediction_success = test_prediction_endpoint()
    
    # Test 3: Docker instructions
    test_with_docker()
    
    print("\n" + "=" * 40)
    if startup_success:
        print("✓ Local testing completed successfully")
        print("If the server works locally but fails in Vertex AI,")
        print("the issue is likely with the Docker environment or model loading.")
    else:
        print("✗ Local testing failed")
        print("Fix the issues above before deploying to Vertex AI.")