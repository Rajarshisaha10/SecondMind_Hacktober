# ===========================================================================
# [CHANGES BACKUP - INCLUDES FEATURE 1, 2, AND FEATURE 3: REALITY CHECKSUM]
# Target File: backend/main.py
# ===========================================================================

"""Main FastAPI application entrypoint for Second Mind Track A backend.

Mounts thin API endpoints, static dashboard assets, and caregiver setup interfaces.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import settings
from backend.constants import Confidence, EventType, Intent, ReminderType
from backend.llm_service import LLMService
from backend.logger import logger
from backend.prompt_manager import PromptManager
from backend.schemas import (
    AskResponse,
    HealthResponse,
    ReminderAckResponse,
    ReminderCheckResponse,
)
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore
from backend.stores.schedule_store import ScheduleStore
from backend.stores.session_store import SessionStore
from backend.stt import STTWrapper
from backend.tts import TTSWrapper

from backend.agents.emergency_agent import EmergencyAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.qa_agent import QAAgent
from backend.agents.reminder_agent import ReminderAgent
from backend.agents.schedule_agent import ScheduleAgent

from backend.services.ask_service import AskService
from backend.services.health_service import HealthService
from backend.services.reminder_service import ReminderService
from backend.stores.object_store import ObjectStore
from backend.visual.service import VisualRetrievalService
from backend.services.reality_checksum_service import RealityChecksumService
from backend.schemas import ObjectRegisterRequest, VisualSearchRequest, RealityConflictVerifyRequest


# ---------------------------------------------------------------------------
# Global Component Initialization (Dependency Injection Container)
# ---------------------------------------------------------------------------
memory_store = MemoryStore()
reality_checksum_service = RealityChecksumService(memory_store=memory_store)
memory_store.reality_checksum = reality_checksum_service
memory_store.seed_demo_data()

object_store = ObjectStore()
visual_service = VisualRetrievalService(object_store=object_store, memory_store=memory_store)

profile_store = ProfileStore()
schedule_store = ScheduleStore()
session_store = SessionStore()


prompt_mgr = PromptManager()
llm_service = LLMService()

stt_wrapper = STTWrapper()
tts_wrapper = TTSWrapper()

memory_agent = MemoryAgent(
    memory_store=memory_store,
    profile_store=profile_store,
    llm=llm_service,
    prompt_mgr=prompt_mgr,
    reality_checksum=reality_checksum_service,
)
schedule_agent = ScheduleAgent(schedule_store, llm_service, prompt_mgr)
qa_agent = QAAgent(profile_store, llm_service, prompt_mgr)
emergency_agent = EmergencyAgent(profile_store, memory_store, llm_service, prompt_mgr)

orchestrator_agent = OrchestratorAgent(
    llm=llm_service,
    prompt_mgr=prompt_mgr,
    session_store=session_store,
    memory_agent=memory_agent,
    schedule_agent=schedule_agent,
    qa_agent=qa_agent,
    emergency_agent=emergency_agent,
)

reminder_agent = ReminderAgent(
    profile_store=profile_store,
    schedule_store=schedule_store,
    memory_store=memory_store,
    tts=tts_wrapper,
)

health_service = HealthService()
ask_service = AskService(orchestrator=orchestrator_agent, stt=stt_wrapper, tts=tts_wrapper)
reminder_service = ReminderService(reminder_agent=reminder_agent)


# ---------------------------------------------------------------------------
# FastAPI Lifespan (Startup/Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to start and stop background timer threads."""
    logger.info("FastAPI: Starting background reminder thread...")
    reminder_service.start_agent()
    yield
    logger.info("FastAPI: Stopping background reminder thread...")
    reminder_service.stop_agent()


app = FastAPI(
    title="Second Mind Backend API & Caregiver Dashboard",
    description="Track A Backend API and Caregiver Setup Dashboard for dementia ambient assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount frontend static directory
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Mount synthesized audio cache directory for device and browser playback
audio_dir = Path(settings.AUDIO_OUTPUT_DIR)
audio_dir.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")


class AskTextRequest(BaseModel):
    """Fallback JSON text request schema for testing /ask without audio."""

    transcript: str
    device_id: Optional[str] = settings.DEFAULT_DEVICE_ID


# ---------------------------------------------------------------------------
# Caregiver Setup Dashboard Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Dashboard"])
@app.get("/dashboard", tags=["Dashboard"])
def get_dashboard():
    """Serve the Caregiver Setup Dashboard web interface."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(content={"message": "Dashboard UI loaded", "status": "ok"})


@app.get("/api/profile", tags=["Caregiver Setup API"])
def get_api_profile() -> Dict[str, Any]:
    """Get current caregiver patient profile setup."""
    return profile_store.get_full_profile()


@app.post("/api/profile", tags=["Caregiver Setup API"])
def post_api_profile(profile_data: Dict[str, Any]) -> Dict[str, str]:
    """Update and persist caregiver patient profile setup."""
    profile_store.save_profile(profile_data)
    return {"status": "saved", "message": "Patient profile updated successfully"}


@app.get("/api/schedule", tags=["Caregiver Setup API"])
def get_api_schedule() -> Dict[str, Any]:
    """Get current caregiver schedule setup."""
    return schedule_store.get_full_schedule()


@app.post("/api/schedule", tags=["Caregiver Setup API"])
def post_api_schedule(schedule_data: Dict[str, Any]) -> Dict[str, str]:
    """Update and persist caregiver schedule setup."""
    schedule_store.save_schedule(schedule_data)
    return {"status": "saved", "message": "Schedule setup updated successfully"}


@app.get("/api/observations", tags=["Caregiver Setup API"])
def get_api_observations() -> List[Dict[str, Any]]:
    """Get recent room observations recorded in MemoryStore."""
    return memory_store.query_all(limit=50)


@app.post("/observations", tags=["Perception Integration"])
@app.post("/api/observations", tags=["Perception Integration"])
def post_observations(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Receive structured room observations from Track B perception loop and save to MemoryStore."""
    added_count = 0
    for obs in observations:
        obj_name = obs.get("object", "")
        loc = obs.get("location", "")
        act = obs.get("action", "")
        evt_type = obs.get("event_type", EventType.OBJECT_OBSERVATION)
        raw_desc = obs.get("raw_description", "")
        ts = obs.get("observed_at") or obs.get("timestamp")
        obs_at = obs.get("observed_at") or ts

        if obj_name and loc:
            memory_store.add_observation(
                object_name=obj_name,
                location=loc,
                action=act,
                event_type=evt_type,
                raw_description=raw_desc,
                timestamp=ts,
                observed_at=obs_at,
            )
            added_count += 1

    logger.info("Main: Saved %d perception observation(s) from Track B", added_count)
    return {"status": "saved", "added_count": added_count}


# ---------------------------------------------------------------------------
# Object Registry & Visual Search Routes
# ---------------------------------------------------------------------------

@app.get("/api/objects", tags=["Object Registry"])
def get_objects() -> List[Dict[str, Any]]:
    """List all registered objects and reference counts."""
    return visual_service.object_store.list_objects()


@app.post("/api/objects", tags=["Object Registry"])
async def create_object(
    request: Request,
    name: Optional[str] = Form(None),
    category: Optional[str] = Form("personal"),
    description: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Register an important object with an optional reference image (supports JSON or multipart form)."""
    content_type = request.headers.get("content-type", "")
    img_bytes = None
    orig_filename = None

    if "application/json" in content_type:
        try:
            data = await request.json()
            name_val = data.get("name")
            cat_val = data.get("category", "personal")
            desc_val = data.get("description", "")
            b64_val = data.get("image_base64")
            if not name_val:
                raise HTTPException(status_code=400, detail="Object 'name' is required.")
            if b64_val:
                import base64
                if "," in b64_val:
                    b64_val = b64_val.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_val)
                orig_filename = f"{name_val}.jpg"
            return visual_service.register_object(
                name=name_val,
                category=cat_val,
                description=desc_val,
                image_bytes=img_bytes,
                filename=orig_filename,
            )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Failed to parse object registration: {e}")

    # Multipart form
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Object 'name' is required.")

    if file is not None:
        img_bytes = await file.read()
        orig_filename = file.filename

    try:
        return visual_service.register_object(
            name=name.strip(),
            category=category or "personal",
            description=description or "",
            image_bytes=img_bytes,
            filename=orig_filename,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/objects/{object_id}", tags=["Object Registry"])
def get_object(object_id: str) -> Dict[str, Any]:
    """Get details of a registered object and its reference images."""
    obj = visual_service.object_store.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found.")
    return obj


@app.delete("/api/objects/{object_id}", tags=["Object Registry"])
def delete_object(object_id: str) -> Dict[str, Any]:
    """Delete a registered object and all its associated reference images."""
    deleted = visual_service.delete_object(object_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Object not found.")
    return {"status": "deleted", "object_id": object_id}


@app.post("/api/objects/{object_id}/references", tags=["Object Registry"])
async def add_reference(object_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and attach a new reference image to an existing registered object."""
    img_bytes = await file.read()
    try:
        ref = visual_service.add_reference_to_object(
            object_id=object_id, image_bytes=img_bytes, filename=file.filename
        )
        return ref
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error adding reference image: %s", e)
        raise HTTPException(status_code=500, detail="Internal error processing reference image.")


@app.delete("/api/objects/{object_id}/references/{reference_id}", tags=["Object Registry"])
def delete_reference(object_id: str, reference_id: str) -> Dict[str, Any]:
    """Delete an individual reference image for an object."""
    deleted = visual_service.delete_reference(reference_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Reference image not found.")
    return {"status": "deleted", "reference_id": reference_id}


@app.get("/api/objects/{object_id}/references/{reference_id}/image", tags=["Object Registry"])
def get_reference_image(object_id: str, reference_id: str):
    """Retrieve the stored reference image file."""
    obj = visual_service.object_store.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found.")
    ref = next((r for r in obj.get("references", []) if r["id"] == reference_id), None)
    if not ref or not ref.get("image_path"):
        raise HTTPException(status_code=404, detail="Reference image not found.")
    p = Path(ref["image_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
    return FileResponse(str(p))


@app.post("/api/visual-search", tags=["Visual Retrieval"])
async def post_visual_search(
    request: Request,
    file: Optional[UploadFile] = File(None),
    target_object_id: Optional[str] = Form(None),
    save_observation: bool = Form(True),
    frame_timestamp: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Search incoming camera frame or uploaded image against registered reference objects."""
    content_type = request.headers.get("content-type", "")
    img_bytes = None
    target_id = target_object_id
    save_obs = save_observation
    frame_ts = frame_timestamp

    if "application/json" in content_type:
        try:
            data = await request.json()
            b64_val = data.get("image_base64")
            target_id = data.get("target_object_id", target_id)
            save_obs = data.get("save_observation", True)
            frame_ts = data.get("frame_timestamp") or data.get("observed_at") or frame_ts
            if not b64_val:
                raise HTTPException(status_code=400, detail="Missing image_base64 in JSON request.")
            import base64
            if "," in b64_val:
                b64_val = b64_val.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_val)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Invalid JSON request: {e}")
    elif file is not None:
        img_bytes = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Must provide image file or image_base64 payload.")

    try:
        bgr_frame = visual_service.validate_image_bytes(img_bytes)
        result = visual_service.search_frame(
            frame=bgr_frame,
            target_object_id=target_id,
            save_observation=save_obs,
            frame_timestamp=frame_ts,
        )
        return result.to_dict()
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Visual search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Visual search failed: {e}")


@app.get("/api/visual-search/recent", tags=["Visual Retrieval"])
def get_recent_visual_searches(limit: int = 50, object_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve recent visual match logs."""
    return visual_service.object_store.get_recent_visual_matches(limit=limit, object_id=object_id)


@app.get("/api/visual-search/diagnostics", tags=["Visual Retrieval"])
def get_visual_diagnostics() -> Dict[str, Any]:
    """Get active visual retrieval pipeline configuration and diagnostics."""
    return visual_service.get_diagnostics()


# ---------------------------------------------------------------------------
# Reality Checksum & Memory Verification Routes
# ---------------------------------------------------------------------------

@app.get("/api/reality-checksum/conflicts", tags=["Reality Checksum"])
def get_reality_conflicts(
    limit: int = 50,
    object_name: Optional[str] = None,
    active_only: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve detected memory discrepancies, relocations, and conflicts."""
    return reality_checksum_service.get_conflicts(
        limit=limit, object_name=object_name, active_only=active_only
    )


@app.get("/api/reality-checksum/states", tags=["Reality Checksum"])
def get_reality_states() -> List[Dict[str, Any]]:
    """Retrieve verified current reality state for all monitored objects."""
    return reality_checksum_service.get_all_states()


@app.post("/api/reality-checksum/verify", tags=["Reality Checksum"])
def verify_reality_discrepancy(payload: RealityConflictVerifyRequest) -> Dict[str, Any]:
    """Manually verify or resolve a flagged discrepancy or relocation."""
    try:
        return reality_checksum_service.resolve_discrepancy(
            conflict_id=payload.conflict_id,
            verified_location=payload.verified_location,
            verified_by=payload.verified_by,
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.exception("Failed to resolve reality conflict: %s", e)
        raise HTTPException(status_code=500, detail="Internal error resolving conflict.")


@app.get("/api/reality-checksum/summary", tags=["Reality Checksum"])
def get_reality_summary() -> Dict[str, Any]:
    """Get metrics summary for reality verification and memory discrepancies."""
    return reality_checksum_service.get_summary()


# ---------------------------------------------------------------------------
# Core Track A Backend API Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health() -> HealthResponse:
    """Perform liveness and subsystem readiness check for demo verification."""
    return health_service.check_health()



@app.post("/ask", tags=["Active Care Loop"])
async def post_ask(
    request: Request,
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    transcript: Optional[str] = Form(None),
    x_device_id: Optional[str] = Header(None, alias="X-Device-ID"),
    x_emergency: Optional[str] = Header(None, alias="X-Emergency"),
    return_json: bool = False,
):
    """Process incoming user question (WAV audio file, JSON transcript text, or Form data).

    Returns WAV audio file response with debug headers:
    - X-Transcript: Transcribed text
    - X-Intent: Classified intent category
    - X-Confidence: Computed confidence level
    - X-Reasoning: Natural language reasoning trace
    - X-Answer-Text: Natural language spoken response
    """
    device_id = x_device_id or settings.DEFAULT_DEVICE_ID
    upload_file = file or audio

    ask_resp: AskResponse
    audio_path: str

    content_type = request.headers.get("content-type", "")

    if x_emergency == "true" or x_emergency == "True":
        ask_resp, audio_path = ask_service.process_ask_text("I need help, emergency call!", device_id=device_id)
    elif "application/json" in content_type:
        try:
            data = await request.json()
            input_text = data.get("transcript", "")
            dev_id = data.get("device_id", device_id)
            if not input_text:
                raise HTTPException(status_code=400, detail="Missing transcript in JSON body.")
            ask_resp, audio_path = ask_service.process_ask_text(input_text, device_id=dev_id)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
    elif transcript is not None and len(transcript.strip()) > 0:
        ask_resp, audio_path = ask_service.process_ask_text(transcript, device_id=device_id)
    elif upload_file is not None:
        audio_bytes = await upload_file.read()
        logger.info(
            "Main: Received audio upload '%s' (%d bytes, content_type='%s')",
            upload_file.filename,
            len(audio_bytes),
            upload_file.content_type,
        )
        ask_resp, audio_path = ask_service.process_ask_audio(audio_bytes, device_id=device_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide audio file upload, form transcript, or JSON transcript body.",
        )

    def _sanitize_header(val: str) -> str:
        if not val:
            return ""
        return (
            val.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    ask_resp.transcript = _sanitize_header(ask_resp.transcript)
    ask_resp.reasoning = _sanitize_header(ask_resp.reasoning)
    ask_resp.answer_text = _sanitize_header(ask_resp.answer_text)

    headers = {
        "X-Transcript": ask_resp.transcript,
        "X-Intent": ask_resp.intent.value,
        "X-Confidence": ask_resp.confidence.value,
        "X-Reasoning": ask_resp.reasoning,
        "X-Answer-Text": ask_resp.answer_text,
    }
    if ask_resp.observed_at:
        headers["X-Observed-At"] = ask_resp.observed_at
    if ask_resp.formatted_time:
        headers["X-Formatted-Time"] = _sanitize_header(ask_resp.formatted_time)
    if ask_resp.response_generated_at:
        headers["X-Response-Generated-At"] = ask_resp.response_generated_at

    if return_json:
        return JSONResponse(content=ask_resp.model_dump(mode="json"), headers=headers)

    p = Path(audio_path)
    if not p.exists():
        p = Path(tts_wrapper.synthesize_to_file(ask_resp.answer_text))

    headers["Content-Length"] = str(p.stat().st_size)

    return FileResponse(
        path=str(p),
        media_type="audio/wav",
        headers=headers,
    )


@app.get("/reminder/check", response_model=ReminderCheckResponse, tags=["Proactive Care Loop"])
def check_reminder() -> ReminderCheckResponse:
    """Poll for pending proactive reminders."""
    return reminder_service.check_reminder()


@app.post("/reminder/ack", response_model=ReminderAckResponse, tags=["Proactive Care Loop"])
def acknowledge_reminder() -> ReminderAckResponse:
    """Acknowledge and clear currently pending proactive reminder."""
    return reminder_service.acknowledge_reminder()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
