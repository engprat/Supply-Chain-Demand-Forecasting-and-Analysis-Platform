#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Add root path for src import resolution
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the validation pipeline
try:
    from src.pipelines.data_validation_pipeline import DataValidationPipeline
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📋 Check your PYTHONPATH or src structure.")
    sys.exit(1)

# Initialize FastAPI
app = FastAPI()

# Set up logging
logger = logging.getLogger(__name__)

# Route for testing
@app.get("/")
def read_root():
    return {"message": "Hello, Welcome to Supply Chain Forecasting!"}

# Route to trigger data validation pipeline
@app.get("/run-validation")
def run_validation_pipeline():
    """
    Runs the data validation pipeline and returns the result.
    """
    try:
        print("🚀 Starting Unified Orders Data Validation Pipeline")
        print("=" * 60)

        # Instantiate and run the pipeline
        pipeline = DataValidationPipeline(config_path="configs/config.yaml")
        success = pipeline.run_pipeline()

        if success:
            return JSONResponse(content={"message": "🎉 Pipeline completed successfully!"}, status_code=200)
        else:
            return JSONResponse(content={"message": "❌ Pipeline finished with issues. Check logs."}, status_code=500)

    except KeyboardInterrupt:
        return JSONResponse(content={"message": "⚠️  Interrupted by user."}, status_code=400)

    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        return JSONResponse(content={"message": f"❌ Critical error occurred: {e}"}, status_code=500)


if __name__ == "__main__":
    # Run FastAPI with uvicorn
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
