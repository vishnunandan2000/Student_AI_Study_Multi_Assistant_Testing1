import json
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.core import studio_db
from backend.modules.document_processor import (
    run_document_processing_generator,
    cancel_document_job
)

app = FastAPI(
    title="Azure Enterprise Knowledge Intelligence Platform (AEKIP) API",
    version="2.0.0",
    description="Backend API for Document Processing, AI Analysis, Embeddings & Real-Time Job Progress Tracking."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Request & Response Models
# =========================================================================

class DocumentEditRequest(BaseModel):
    updated_text: str = Field(..., description="Full updated text content of the document")
    filename: Optional[str] = Field("Document.txt", description="Document filename")
    change_summary: Optional[str] = Field("", description="Optional user note explaining changes")
    task_type: Optional[str] = Field(
        "Edit & Reindex Existing Document (Versioned Update)",
        description="Downstream analysis or indexing task"
    )

class DocumentReindexRequest(BaseModel):
    task_type: Optional[str] = Field(
        "Comprehensive Document Intelligence",
        description="Reindexing operation task type"
    )


# =========================================================================
# Background Job Execution Runner
# =========================================================================

def _run_document_pipeline_task(job_id: str, doc_id: str, text: str, filename: str, task_type: str):
    """Executes document workflow in background worker thread and updates SQLite state."""
    gen = run_document_processing_generator(
        file_obj=None,
        pasted_text=text,
        task_type=task_type,
        existing_doc_id=doc_id,
        job_id=job_id
    )
    for state in gen:
        if state["is_complete"] or state["is_error"] or state["is_cancelled"]:
            break


# =========================================================================
# Core API Routes
# =========================================================================

@app.get("/")
def read_root():
    return {
        "platform": "Azure Enterprise Knowledge Intelligence Platform (AEKIP)",
        "status": "operational",
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}


# =========================================================================
# Document Management & Reindexing Endpoints
# =========================================================================

@app.get("/api/documents")
def list_documents(limit: int = 50):
    """Lists tracked enterprise documents with their current version and indexed status."""
    return {"documents": studio_db.list_document_records(limit=limit)}

@app.get("/api/documents/{document_id}")
def get_document_details(document_id: str):
    """Retrieves document record metadata and historical version list."""
    rec = studio_db.get_document_record(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    versions = studio_db.get_document_versions(document_id)
    return {"document": rec, "versions": versions}

@app.post("/api/documents/{document_id}/edit")
def edit_document(document_id: str, req: DocumentEditRequest, background_tasks: BackgroundTasks):
    """
    Submits an edited version of an existing document.
    Triggers change validation, semantic chunking, and embedding regeneration in background.
    """
    job_id = f"job_edit_{uuid.uuid4().hex[:8]}"
    filename = req.filename or f"{document_id}.txt"

    background_tasks.add_task(
        _run_document_pipeline_task,
        job_id=job_id,
        doc_id=document_id,
        text=req.updated_text,
        filename=filename,
        task_type=req.task_type or "Edit & Reindex Existing Document (Versioned Update)"
    )

    return {
        "message": f"Document '{document_id}' edit accepted for background processing.",
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",
        "poll_url": f"/api/jobs/{job_id}",
        "events_url": f"/api/jobs/{job_id}/events"
    }

@app.post("/api/documents/{document_id}/reindex")
def reindex_document(document_id: str, req: DocumentReindexRequest, background_tasks: BackgroundTasks):
    """
    Triggers re-chunking, embedding regeneration, and search index refresh for an existing document.
    """
    rec = studio_db.get_document_record(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    versions = studio_db.get_document_versions(document_id)
    latest_text = versions[-1]["content_text"] if versions else ""
    if not latest_text:
        raise HTTPException(status_code=400, detail="Document has no content to reindex.")

    job_id = f"job_reindex_{uuid.uuid4().hex[:8]}"
    filename = rec.get("filename", "Document.txt")

    background_tasks.add_task(
        _run_document_pipeline_task,
        job_id=job_id,
        doc_id=document_id,
        text=latest_text,
        filename=filename,
        task_type=req.task_type or "Comprehensive Document Intelligence"
    )

    return {
        "message": f"Reindex scheduled for document '{document_id}'.",
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",
        "poll_url": f"/api/jobs/{job_id}",
        "events_url": f"/api/jobs/{job_id}/events"
    }


# =========================================================================
# Real-Time Job Progress & Status Endpoints
# =========================================================================

@app.get("/api/jobs")
def list_jobs(limit: int = 30):
    """Lists recent document processing jobs with status and progress percentage."""
    return {"jobs": studio_db.list_document_jobs(limit=limit)}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Retrieves the real-time processing status, current stage, and progress (0-100%) for a job.
    """
    job = studio_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    """Safely cancels an active document processing job."""
    job = studio_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    success, msg = cancel_document_job(job_id)
    return {"job_id": job_id, "cancelled": success, "message": msg}

@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    """
    Server-Sent Events (SSE) stream providing real-time progress updates for a document job.
    """
    async def event_generator():
        last_progress = -1.0
        last_stage = ""
        for _ in range(120): # Poll up to 60 seconds
            job = studio_db.get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            curr_pct = job.get("progress", 0.0)
            curr_stage = job.get("current_stage", "")
            curr_status = job.get("status", "")

            if curr_pct != last_progress or curr_stage != last_stage:
                last_progress = curr_pct
                last_stage = curr_stage
                yield f"data: {json.dumps(job)}\n\n"

            if curr_status in ["completed", "failed", "cancelled"]:
                # Send terminal event
                yield f"data: {json.dumps(job)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

