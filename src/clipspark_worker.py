import os
import logging
import traceback
from pathlib import Path
from celery import Celery
from src.clipspark_engine import ClipSparkPipeline, PipelineConfig, setup_logging

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("clipspark_worker", broker=redis_url, backend=redis_url)

logger = setup_logging("INFO")

@app.task(bind=True, name="clipspark.process_video")
def process_video_task(self, video_path: str, output_dir: str):
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting..."})
        
        # Configure Pipeline to use the SHARED output directory
        config = PipelineConfig()
        
        # CRITICAL: Tell engine where the shared root is
        # If output_dir is /app/shared/output, base_dir is /app/shared
        config.base_dir = Path(output_dir).parent 
        
        pipeline = ClipSparkPipeline(config, logger)
        result = pipeline.run(video_path)
        
        return result

    except Exception as e:
        logger.error(f"Task failed: {e}")
        self.update_state(state='FAILURE', meta=str(e))
        raise e