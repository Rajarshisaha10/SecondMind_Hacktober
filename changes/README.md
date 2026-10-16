# SecondMind Changes & Backup Directory

> **Backup Context for Teammates & Antigravity IDE**  
> This directory (`changes/`) contains a complete, self-contained backup and reference implementation for the **three major features** added to SecondMind:
> 1. **Feature 1: Local Visual Object Retrieval & Reference Image Embeddings** (Commit `2c7d6ac`)
> 2. **Feature 2: Accurate Observation Timestamps & Temporal Reasoning** (Commit `b885984`)
> 3. **Feature 3: Reality Checksum: Discrepancy & Conflict Verification** (Current feature)

---

## 🚀 Quick Start for Teammates & Antigravity

If your branch or code encounters issues, you can:
1. **Drop-in Replace:** The directory layout inside `changes/` mirrors the repository root 1-to-1 (`changes/backend/...`, `changes/frontend/...`). You can copy files directly over the originals to restore fully verified functionality.
2. **Review Annotated Diffs:** Every modified file in this directory contains explicit inline block comments:
   - `### [NEW - Feature Name]` — highlights freshly introduced functions, routes, and schemas.
   - `### [MODIFIED - Feature Name]` — explains what was originally there in commit `584d7b5`, why it changed, and how it behaves now.
3. **Antigravity Context Acceleration:** Pointing Antigravity to `changes/README.md` or any file inside `changes/` gives the model immediate end-to-end understanding of all feature pipelines without having to analyze raw git diffs.

---

## 📁 File Manifest & Overview

### 1. Brand New Files (14 files)
These files did not exist in the initial repository (`584d7b5`) and were created to implement the features:

| File Path | Feature | Purpose |
| :--- | :--- | :--- |
| `backend/services/reality_checksum_service.py` | Feature 3 | **Reality Checksum Service**: detects conflicts between new observations and stored memories, verifies discrepancies, tracks relocations, supersedes outdated rows, and manages verified reality states. |
| `backend/tests/test_reality_checksum.py` | Feature 3 | **Pytest Test Suite**: 8 unit & integration tests covering location changes, stationary verification, ambiguous discrepancies, temporal anomalies, and manual resolution. |
| `backend/time_utils.py` | Feature 2 | Reusable deterministic time formatting, timezone handling (`Asia/Kolkata`), relative time calculation, and natural language location/time phrase helpers. |
| `backend/stores/object_store.py` | Feature 1 | SQLite store managing registered objects, reference image paths, 576-dim MobileNetV3 visual embeddings (BLOB storage), and visual match audit logging. |
| `backend/visual/__init__.py` | Feature 1 | Module exports for visual encoder, matcher, proposals, and service. |
| `backend/visual/encoder.py` | Feature 1 | MobileNetV3-Small deep feature extractor producing L2-normalized 576-dim embeddings with LRU caching and OpenCV histogram fallback. |
| `backend/visual/proposals.py` | Feature 1 | Object proposal generator using edge/contour saliency, selective search, and sliding window crops for instance detection in camera frames. |
| `backend/visual/matcher.py` | Feature 1 & 2 | Multi-candidate cosine similarity matcher, multi-reference max-pooling, spatial bounding box prediction, local feature (SIFT/ORB) verification, and timestamp attachment. |
| `backend/visual/service.py` | Feature 1, 2, 3 | High-level orchestrator for object registration, reference embedding persistence, frame search, observation logging, timestamp propagation, and Reality Checksum diagnostics. |
| `backend/evaluation.py` | Feature 1 | Evaluation benchmark suite measuring visual retrieval precision, recall, top-1 accuracy, and latency under lighting/occlusion variations. |
| `backend/tests/test_object_endpoints.py` | Feature 1 | Pytest suite validating object registration, reference upload/delete, and retrieval REST endpoints. |
| `backend/tests/test_visual_encoder.py` | Feature 1 | Pytest unit tests for MobileNetV3 embedding extraction, normalization, device selection, and fallback. |
| `backend/tests/test_visual_retrieval.py` | Feature 1 | Pytest integration tests for visual matching, similarity thresholds, ambiguity detection, and local keypoint verification. |
| `backend/tests/test_timestamp_output.py` | Feature 2 | 20 comprehensive unit and integration tests covering human formatting, timezone conversions, MemoryAgent responses, singular/plural grammar, and API headers. |

---

### 2. Modified Existing Files (11 files)
Each of these files contains inline annotations explaining the additions and modifications:

| File Path | What Changed |
| :--- | :--- |
| `backend/main.py` | Initialized `ObjectStore`, `VisualRetrievalService`, and `RealityChecksumService`; mounted 9 visual retrieval endpoints and 4 Reality Checksum endpoints (`/api/reality-checksum/conflicts`, `/states`, `/verify`, `/summary`); attached timestamp headers (`X-Observed-At`, `X-Formatted-Time`, `X-Response-Generated-At`). |
| `backend/stores/memory_store.py` | Added safe SQLite column migrations (`observed_at`, `created_at`, `checksum_status`, `superseded_by`, etc.); automatically invokes `RealityChecksumService.process_observation()`; marks previous observations as `superseded` upon verified relocation. |
| `backend/agents/memory_agent.py` | Injects `RealityChecksumService`; incorporates verified object relocation into answer reasoning (`"Reality Checksum: Verified relocation from '{prev_loc}' to '{new_loc}'"`) and answer text when asking about moved items; handles singular/plural grammar; deterministic templates across 5 tiers. |
| `backend/schemas.py` | Added `RealityConflictVerifyRequest`; added `observed_at`, `formatted_time`, and `response_generated_at` to `AgentResponse` and `AskResponse`; added `ObjectRegisterRequest` and `VisualSearchRequest` DTOs. |
| `backend/config.py` | Added default `TIMEZONE = "Asia/Kolkata"` and visual retrieval settings (`VISUAL_ENCODER_MODEL`, thresholds, image limits, directory paths). |
| `backend/services/ask_service.py` | Passed `observed_at`, `formatted_time`, and `response_generated_at` from `AgentResponse` to `AskResponse` in `process_ask_text()` and `process_ask_audio()`. |
| `backend/agents/orchestrator.py` | Added deterministic rule-based keyword intent fallback in `classify_intent()` when LLM is cold, slow, or offline in local test environments. |
| `backend/perception.py` | Captures exact camera frame timestamp at `cap.read()` in configured timezone; sends frame to local visual search service before Ollama; attaches `observed_at` to observations. |
| `requirements.txt` | Added `torch>=2.0.0`, `torchvision>=0.15.0`, `psutil>=5.9.0` for visual feature extraction and diagnostics. |
| `frontend/index.html` | Added "Object Registry & Vision" tab and "Reality Checksum" tab with verified states table, conflict history table, and resolution actions. |
| `frontend/app.js` | Added client controllers for object registration, visual search tester, and Reality Checksum (`loadRealityChecksum`, `resolveDiscrepancyModal`). |

---

## 🛡️ Feature 3 Architecture: Reality Checksum

```
Incoming Observation (Perception or Visual Search)
                      │
                      ▼
        [RealityChecksumService.process_observation()]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[Location Matched]          [Location Mismatched]
        │                           │
  Checksum Valid            Check Evidence & Timestamp
  (Stationary Confirmed)            │
                            ├─► New Ts < Prev Ts ──► [TEMPORAL ANOMALY] (Reject out-of-order)
                            │
                            ├─► Weak vs Strong   ──► [AMBIGUOUS DISCREPANCY] (Flag, keep prior)
                            │
                            └─► Confirmed Move   ──► [LOCATION CHANGE]
                                                           │
                                                           ├─► Supersede old row in MemoryStore
                                                           ├─► Update reality_object_states
                                                           └─► Log in reality_conflicts
```

### Conflict Types & Resolution
1. **`LOCATION_CHANGE`**: Object moved from `Location A` to `Location B`. The previous observation in `MemoryStore` is marked `superseded`, and the new verified location becomes current.
2. **`STATIONARY_CONFIRMED`**: Object observed at the same location. Extends recency without conflict.
3. **`AMBIGUOUS_DISCREPANCY`**: A weak or ambiguous sighting contradicts a verified strong prior memory. It is flagged as `FLAGGED_UNVERIFIED` without corrupting the confirmed location until verified.
4. **`TEMPORAL_ANOMALY`**: An observation arriving with an older timestamp than current verified state is recorded historically as `REJECTED` from overwriting newer state.

---

## 🧪 Verification & Test Results

Run the entire test suite from the repository root:
```bash
pytest backend/tests
```

**Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0
collected 75 items

backend\tests\test_ask_endpoint.py ...                                   [  4%]
backend\tests\test_confidence.py ....                                    [  9%]
backend\tests\test_dashboard_api.py ....                                 [ 14%]
backend\tests\test_demo_scenario.py .                                    [ 16%]
backend\tests\test_health.py ..                                          [ 18%]
backend\tests\test_object_endpoints.py ......                            [ 26%]
backend\tests\test_reality_checksum.py ........                          [ 37%]
backend\tests\test_reminder_agent.py ..                                  [ 40%]
backend\tests\test_reminder_endpoints.py .                               [ 41%]
backend\tests\test_routing.py ..                                         [ 44%]
backend\tests\test_session_context.py ...                                [ 48%]
backend\tests\test_stores.py ....                                        [ 53%]
backend\tests\test_timestamp_output.py ....................              [ 80%]
backend\tests\test_visual_encoder.py ........                            [ 90%]
backend\tests\test_visual_retrieval.py .......                           [100%]

======================= 75 passed in 100.86s (0:01:40) ========================
```
