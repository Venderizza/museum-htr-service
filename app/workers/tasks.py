import logging
import time
from pathlib import Path
from uuid import UUID

import dramatiq
import httpx
import time

from app.core.config import get_settings
from app.db.models import OCRAttemptLog
from app.db.repositories.jobs import JobRepository
from app.db.repositories.results import ResultRepository
from app.db.session import SessionLocal
from app.ocr.pipeline.service import OCRPipelineService
from app.services.image_storage import ImageStorageService
from app.workers.broker import redis_broker  # noqa: F401

logger = logging.getLogger(__name__)
settings = get_settings()


@dramatiq.actor(max_retries=3)
def process_ocr_job(job_id: str) -> None:
    started = time.perf_counter()
    db = SessionLocal()
    image_id = None
    try:
        jobs = JobRepository(db)
        results = ResultRepository(db)
        job = jobs.get(UUID(job_id))
        if job is None:
            logger.error("Job not found", extra={"job_id": job_id})
            return

        image = job.image
        image_id = image.id
        jobs.set_status(job, status="processing", progress=10)
        db.commit()

        update_ocr_page_status.send(str(image.id), "processing")

        if not image.storage_path:
            raise RuntimeError("Image has no storage path")

        jobs.set_status(job, status="preprocessing", progress=20)
        db.commit()

        pipeline = OCRPipelineService()

        jobs.set_status(job, status="recognizing", progress=60)
        db.commit()

        pipeline_result = pipeline.run(Path(image.storage_path))

        storage = ImageStorageService()
        processed_path = storage.mark_processed(Path(image.storage_path))
        image.storage_path = str(processed_path)

        result = results.create(
            image_id=image.id,
            job_id=job.id,
            title=image.title,
            raw_text=pipeline_result.raw_text,
            normalized_text=pipeline_result.normalized_text,
            body=pipeline_result.body,
            confidence=pipeline_result.confidence,
            segmenter=pipeline_result.segmenter,
            segmentation_model=pipeline_result.segmentation_model,
            recognizer=pipeline_result.recognizer,
            recognition_model=pipeline_result.recognition_model,
            recognition_model_version=pipeline_result.recognition_model_version,
            preprocessing_config=pipeline_result.preprocessing_config,
            recognition_config=pipeline_result.recognition_config,
            result_json=pipeline_result.result_json,
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        jobs.set_status(job, status="completed", progress=100, processing_time_ms=elapsed_ms)
        db.add(
            OCRAttemptLog(
                image_id=image.id,
                job_id=job.id,
                stage="pipeline",
                status="completed",
                message="OCR pipeline completed",
                payload={"result_id": str(result.id), "processing_time_ms": elapsed_ms},
            )
        )
        db.commit()

        post_ocr_result.send(str(image.id), str(result.normalized_text))

    except Exception as exc:
        logger.exception("OCR job failed", extra={"job_id": job_id})
        db.rollback()
        job = JobRepository(db).get(UUID(job_id))
        if job is not None:
            JobRepository(db).set_status(
                job,
                status="failed",
                progress=100,
                error_code="OCR_FAILED",
                error_message=str(exc),
            )
            db.commit()
        if image_id is not None:
            update_ocr_page_status.send(str(image_id), "failed")
        raise
    finally:
        db.close()


@dramatiq.actor(max_retries=3)
def update_ocr_page_status(image_id: str, status: str) -> None:
    resp = httpx.post(f"{settings.api_admin_panel_url}/{image_id}/status", data={"status": status})
    resp.raise_for_status()
    logger.info(f"status send for image: {image_id}")


@dramatiq.actor(max_retries=3)
def post_ocr_result(image_id: str, text: str) -> None:
    resp = httpx.post(f"{settings.api_admin_panel_url}/{image_id}/result", data={"text": text})
    resp.raise_for_status()
    logger.info(f"result send for image: {image_id}")
