#!/usr/bin/env python3
"""
serve_model.py

FastAPI server for serving spaCy text classification models.

Usage:
    python serve_model.py --model ./output/model-best --port 8000

Endpoints:
    POST /classify         - Classify single text
    POST /classify/batch   - Classify multiple texts
    GET  /health           - Health check
    GET  /info             - Model information
"""

import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

try:
    import spacy
except ImportError:
    print("Error: spaCy not installed. Run: pip install spacy")
    sys.exit(1)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("classifier-api")


# Global model reference
nlp = None
model_info = {}


# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Text to classify")


class BatchInput(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100, description="List of texts")


class ClassificationResult(BaseModel):
    category: str
    confidence: float
    scores: dict[str, float]
    processing_time_ms: float


class BatchResult(BaseModel):
    results: list[dict]
    total_processing_time_ms: float
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_path: str
    pipeline: list[str]
    categories: list[str]
    spacy_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global nlp, model_info
    
    model_path = os.environ.get("MODEL_PATH", "./model")
    
    logger.info(f"Loading model from {model_path}...")
    start = time.time()
    
    try:
        nlp = spacy.load(model_path)
        load_time = time.time() - start
        
        # Get categories from textcat component
        categories = []
        for name in nlp.pipe_names:
            if "textcat" in name:
                pipe = nlp.get_pipe(name)
                categories = list(pipe.labels)
                break
        
        model_info = {
            "model_path": str(model_path),
            "pipeline": nlp.pipe_names,
            "categories": categories,
            "spacy_version": spacy.__version__
        }
        
        logger.info(f"Model loaded in {load_time:.2f}s")
        logger.info(f"Categories: {categories}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Text Classifier API",
    description="Classify text using spaCy TextCategorizer",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if nlp is not None else "unhealthy",
        model_loaded=nlp is not None
    )


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(**model_info)


@app.post("/classify", response_model=ClassificationResult)
async def classify_text(input: TextInput):
    """Classify a single text."""
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        with nlp.memory_zone():
            doc = nlp(input.text)
            category = max(doc.cats, key=doc.cats.get)
            confidence = doc.cats[category]
            scores = {k: round(v, 4) for k, v in doc.cats.items()}
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    processing_time = (time.time() - start) * 1000
    
    logger.info(f"Classified: {category} ({confidence:.2%}) in {processing_time:.1f}ms")
    
    return ClassificationResult(
        category=category,
        confidence=round(confidence, 4),
        scores=scores,
        processing_time_ms=round(processing_time, 2)
    )


@app.post("/classify/batch", response_model=BatchResult)
async def classify_batch(input: BatchInput):
    """Classify multiple texts in a batch."""
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    results = []
    
    try:
        with nlp.memory_zone():
            for doc in nlp.pipe(input.texts, batch_size=50):
                category = max(doc.cats, key=doc.cats.get)
                results.append({
                    "text": doc.text[:100],  # Truncate for response
                    "category": category,
                    "confidence": round(doc.cats[category], 4),
                    "scores": {k: round(v, 4) for k, v in doc.cats.items()}
                })
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    processing_time = (time.time() - start) * 1000
    
    logger.info(f"Batch classified {len(results)} texts in {processing_time:.1f}ms")
    
    return BatchResult(
        results=results,
        total_processing_time_ms=round(processing_time, 2),
        count=len(results)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Serve spaCy text classification model via REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic server
    python serve_model.py --model ./output/model-best

    # Custom port
    python serve_model.py --model ./output/model-best --port 8080

    # Production settings
    python serve_model.py --model ./output/model-best --host 0.0.0.0 --port 8000 --workers 4

API Endpoints:
    POST /classify         - Classify single text
    POST /classify/batch   - Classify multiple texts (up to 100)
    GET  /health           - Health check
    GET  /info             - Model information
    GET  /docs             - Interactive API documentation
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--host", "-H",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model not found: {args.model}")
    
    # Set model path for lifespan
    os.environ["MODEL_PATH"] = str(model_path)
    
    print(f"\n🚀 Starting Text Classifier API")
    print(f"   Model: {args.model}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print()
    
    uvicorn.run(
        "serve_model:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
