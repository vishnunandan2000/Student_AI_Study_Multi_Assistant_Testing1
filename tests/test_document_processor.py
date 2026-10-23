"""
Automated Unit and Integration Tests for Document Processing Progress Bar & Real-Time Status Monitor
Verifies:
1. Stage progression and percentage advancement across all 8 stages (0% -> 100%).
2. Status transitions (Uploading, Extracting, Analyzing, Generating embeddings, Indexing, Validating, Completed).
3. Text extraction, normalization, and semantic chunking with overlap.
4. Embedding metadata generation and SQLite search index persistence.
5. Document editing, version increment (v1 -> v2), and diff calculation.
6. Safe job cancellation and failed stage reporting.
7. Live status monitor card rendering with stage checklist.
8. FastAPI endpoints: /api/documents, /api/documents/{id}/edit, /api/jobs/{id}, /api/jobs/{id}/cancel.
"""

import os
import sys
import json
import uuid
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.document_processor import (
    clean_and_normalize_text,
    chunk_document_text,
    simulate_generate_embeddings,
    detect_document_diff,
    render_document_status_card,
    run_document_processing_generator,
    cancel_document_job,
    DOCUMENT_STAGES
)
from backend.core import studio_db
from backend.main import app


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.mock_resp = MagicMock()
        self.mock_resp.content = "Executive Summary: Validated document content."
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = self.mock_resp

        self.llm_patcher = patch("backend.modules.document_processor.get_llm", return_value=self.mock_llm)
        self.llm_patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.llm_patcher.stop()

    def test_clean_and_normalize_text(self):
        raw = "  Heading 1   \n\n\n\nParagraph with    extra     spaces.\n\n\nConclusion.  "
        cleaned = clean_and_normalize_text(raw)
        self.assertNotIn("    ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertTrue(cleaned.startswith("Heading 1"))
        self.assertTrue(cleaned.endswith("Conclusion."))

    def test_semantic_chunking_with_overlap(self):
        sample_text = (
            "Section 1: Introduction to Corporate Policy.\n"
            "This document establishes enterprise remote work guidelines and equipment usage.\n\n"
            "Section 2: Information Security Protocols.\n"
            "All employees must connect via corporate VPN and enable multi-factor authentication.\n\n"
            "Section 3: Compliance and Annual Reviews.\n"
            "Compliance audits occur quarterly to ensure security standards are strictly enforced.\n"
        )
        chunks = chunk_document_text(sample_text, chunk_size=120, chunk_overlap=30)
        self.assertGreaterEqual(len(chunks), 2, "Expected multiple semantic chunks")
        for idx, c in enumerate(chunks):
            self.assertIn("chunk_id", c)
            self.assertIn("text", c)
            self.assertGreater(len(c["text"]), 0)
            self.assertEqual(c["chunk_index"], idx)

    def test_embedding_generation_metadata(self):
        chunks = [
            {"chunk_id": "c1", "chunk_index": 0, "text": "Remote work security policy and VPN guidelines."},
            {"chunk_id": "c2", "chunk_index": 1, "text": "Quarterly audit procedures and incident response."}
        ]
        embedded = simulate_generate_embeddings(chunks)
        self.assertEqual(len(embedded), 2)
        for c in embedded:
            self.assertTrue(c["has_embedding"])
            self.assertEqual(c["embedding_dim"], 384)
            self.assertTrue(len(c["vector_sample"]) > 0)

    def test_document_diff_detection(self):
        v1_text = "Line 1: Remote work is permitted on Fridays.\nLine 2: Use laptop provided by IT."
        v2_text = "Line 1: Remote work is permitted on all weekdays.\nLine 2: Use laptop provided by IT.\nLine 3: VPN is mandatory."
        diff = detect_document_diff(v1_text, v2_text)

        self.assertTrue(diff["is_substantive_change"])
        self.assertEqual(diff["lines_added_count"], 2)
        self.assertEqual(diff["lines_removed_count"], 1)
        self.assertGreater(diff["char_difference"], 0)

    def test_status_card_rendering_states(self):
        # Queued / Pending state
        html_pending = render_document_status_card(
            current_stage_idx=0,
            total_stages=8,
            stage_name="Ready for Document Processing",
            pct=0.0,
            elapsed_sec=0.0,
            status_label="pending",
            filename="Policy_2026.pdf",
            version_num=1
        )
        self.assertIn("0.0%", html_pending)
        self.assertIn("Policy_2026.pdf", html_pending)
        self.assertIn("PENDING", html_pending)

        # Mid-processing state
        html_mid = render_document_status_card(
            current_stage_idx=5,
            total_stages=8,
            stage_name="Generating vector embeddings for chunks",
            pct=60.0,
            elapsed_sec=14.0,
            status_label="generating embeddings",
            filename="Policy_2026.pdf",
            version_num=2,
            estimated_remaining_sec=9.0
        )
        self.assertIn("60.0%", html_mid)
        self.assertIn("Stage 5 of 8", html_mid)
        self.assertIn("v2", html_mid)
        self.assertIn("00:14", html_mid)
        self.assertIn("00:09", html_mid)

        # Completed state
        html_done = render_document_status_card(
            current_stage_idx=8,
            total_stages=8,
            stage_name="Finalizing AI analysis",
            pct=100.0,
            elapsed_sec=28.0,
            status_label="completed",
            filename="Policy_2026.pdf",
            version_num=2
        )
        self.assertIn("100.0%", html_done)
        self.assertIn("COMPLETED", html_done)

    def test_8_stage_generator_workflow_success(self):
        doc_id = f"test_doc_{uuid.uuid4().hex[:8]}"
        job_id = f"job_success_{uuid.uuid4().hex[:8]}"
        doc_text = "Enterprise AI Architecture and Model Governance Policy.\nGuidelines for deploying LLM microservices."
        gen = run_document_processing_generator(
            file_obj=None,
            pasted_text=doc_text,
            task_type="Comprehensive Document Intelligence",
            existing_doc_id=doc_id,
            job_id=job_id
        )

        observed_stages = []
        observed_progress = []
        final_state = None

        for state in gen:
            observed_stages.append(state["stage_index"])
            observed_progress.append(state["progress_percent"])
            final_state = state

        # Verify all 8 stages were hit in order
        self.assertEqual(len(set(observed_stages)), 8)
        self.assertEqual(sorted(list(set(observed_stages))), [1, 2, 3, 4, 5, 6, 7, 8])

        # Verify progress is monotonic
        for i in range(len(observed_progress) - 1):
            self.assertLessEqual(observed_progress[i], observed_progress[i + 1])

        # Verify final state reached 100% and completed
        self.assertTrue(final_state["is_complete"])
        self.assertEqual(final_state["progress_percent"], 100.0)
        self.assertEqual(final_state["status_label"], "completed")

        # Verify database record was persisted
        doc_rec = studio_db.get_document_record(doc_id)
        self.assertIsNotNone(doc_rec)
        self.assertEqual(doc_rec["document_id"], doc_id)
        self.assertGreater(doc_rec["chunk_count"], 0)

    def test_document_versioning_increment(self):
        doc_id = f"test_doc_ver_{uuid.uuid4().hex[:8]}"
        job_id_v1 = f"job_v1_{uuid.uuid4().hex[:8]}"
        job_id_v2 = f"job_v2_{uuid.uuid4().hex[:8]}"

        # Version 1 upload
        v1_text = "Initial Policy Version 1: Core security rules."
        for _ in run_document_processing_generator(
            file_obj=None,
            pasted_text=v1_text,
            task_type="Comprehensive Document Intelligence",
            existing_doc_id=doc_id,
            job_id=job_id_v1
        ):
            pass

        rec_v1 = studio_db.get_document_record(doc_id)
        self.assertEqual(rec_v1["current_version"], 1)

        # Version 2 edit
        v2_text = "Updated Policy Version 2: Core security rules plus cloud infrastructure guidelines."
        for _ in run_document_processing_generator(
            file_obj=None,
            pasted_text=v2_text,
            task_type="Edit & Reindex Existing Document (Versioned Update)",
            existing_doc_id=doc_id,
            job_id=job_id_v2
        ):
            pass

        rec_v2 = studio_db.get_document_record(doc_id)
        self.assertEqual(rec_v2["current_version"], 2)

        versions = studio_db.get_document_versions(doc_id)
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version_number"], 1)
        self.assertEqual(versions[1]["version_number"], 2)
        self.assertIn("Diff", versions[1]["change_summary"])

    def test_cancellation_stops_execution(self):
        doc_text = "Document for cancellation testing."
        doc_id = f"test_doc_cancel_{uuid.uuid4().hex[:8]}"
        job_id = f"job_cancel_{uuid.uuid4().hex[:8]}"

        gen = run_document_processing_generator(
            file_obj=None,
            pasted_text=doc_text,
            existing_doc_id=doc_id,
            job_id=job_id
        )

        # Start generator, get initial state, then request cancellation
        initial_state = next(gen)
        self.assertFalse(initial_state["is_cancelled"])

        cancel_document_job(job_id)

        final_state = None
        for state in gen:
            final_state = state
            if state["is_cancelled"]:
                break

        self.assertTrue(final_state["is_cancelled"])
        self.assertEqual(final_state["status_label"], "cancelled")

    def test_fastapi_endpoints(self):
        # 1. Health check
        res_h = self.client.get("/health")
        self.assertEqual(res_h.status_code, 200)

        # 2. Document listing
        res_docs = self.client.get("/api/documents")
        self.assertEqual(res_docs.status_code, 200)
        self.assertIn("documents", res_docs.json())

        # 3. Document edit submission endpoint
        edit_payload = {
            "updated_text": "Updated corporate travel policy guidelines for Q4.",
            "filename": "Travel_Policy_Q4.pdf",
            "change_summary": "Added per-diem limits",
            "task_type": "Comprehensive Document Intelligence"
        }
        res_edit = self.client.post("/api/documents/doc_travel/edit", json=edit_payload)
        self.assertEqual(res_edit.status_code, 200)
        edit_data = res_edit.json()
        self.assertIn("job_id", edit_data)
        self.assertEqual(edit_data["document_id"], "doc_travel")

        # 4. Job status polling endpoint
        job_id = edit_data["job_id"]
        res_job = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(res_job.status_code, 200)
        job_status = res_job.json()
        self.assertEqual(job_status["job_id"], job_id)

        # 5. Job cancellation endpoint
        res_cancel = self.client.post(f"/api/jobs/{job_id}/cancel")
        self.assertEqual(res_cancel.status_code, 200)
        self.assertTrue(res_cancel.json()["cancelled"])


if __name__ == "__main__":
    unittest.main()
