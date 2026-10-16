"""Integration test suite for Object Registry and Visual Search REST API endpoints."""

import base64
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app, visual_service

client = TestClient(app)


def create_encoded_jpeg(bgr_color=(0, 0, 255)) -> bytes:
    """Create in-memory JPEG bytes for testing."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (108, 108), bgr_color, -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_post_objects_json() -> None:
    """Verify POST /api/objects with JSON payload registers an object."""
    jpeg_bytes = create_encoded_jpeg(bgr_color=(0, 255, 0))
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")

    payload = {
        "name": "House Keys",
        "category": "keys",
        "description": "Keychain with green tag",
        "image_base64": b64_str,
    }
    response = client.post("/api/objects", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "House Keys"
    assert data["category"] == "keys"
    assert "id" in data
    assert len(data.get("references", [])) == 1

    # Clean up
    client.delete(f"/api/objects/{data['id']}")


def test_post_objects_multipart() -> None:
    """Verify POST /api/objects with multipart form-data file upload."""
    jpeg_bytes = create_encoded_jpeg(bgr_color=(255, 0, 0))
    file_io = io.BytesIO(jpeg_bytes)

    files = {"file": ("reading_glasses.jpg", file_io, "image/jpeg")}
    form_data = {
        "name": "Reading Glasses",
        "category": "glasses",
        "description": "Black rectangular frame",
    }
    response = client.post("/api/objects", data=form_data, files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Reading Glasses"
    obj_id = data["id"]

    # Verify listing includes the new object
    list_resp = client.get("/api/objects")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert any(item["id"] == obj_id for item in items)

    # Verify getting single object details
    get_resp = client.get(f"/api/objects/{obj_id}")
    assert get_resp.status_code == 200
    details = get_resp.json()
    assert len(details["references"]) == 1
    ref_id = details["references"][0]["id"]

    # Verify reference image download endpoint
    img_resp = client.get(f"/api/objects/{obj_id}/references/{ref_id}/image")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] in ["image/jpeg", "image/jpg"]
    assert len(img_resp.content) > 0

    # Clean up
    del_resp = client.delete(f"/api/objects/{obj_id}")
    assert del_resp.status_code == 200


def test_add_and_delete_reference_image() -> None:
    """Verify adding additional reference to an object and deleting it."""
    # Register object without image
    response = client.post(
        "/api/objects",
        json={"name": "Wallet", "category": "personal", "description": "Leather wallet"},
    )
    assert response.status_code == 200
    obj_id = response.json()["id"]

    # Upload first reference
    jpeg_1 = create_encoded_jpeg(bgr_color=(50, 50, 50))
    files = {"file": ("wallet_front.jpg", io.BytesIO(jpeg_1), "image/jpeg")}
    add_resp = client.post(f"/api/objects/{obj_id}/references", files=files)
    assert add_resp.status_code == 200
    ref_id = add_resp.json()["id"]

    # Check reference count is 1
    details = client.get(f"/api/objects/{obj_id}").json()
    assert len(details["references"]) == 1

    # Delete reference
    del_ref_resp = client.delete(f"/api/objects/{obj_id}/references/{ref_id}")
    assert del_ref_resp.status_code == 200
    details2 = client.get(f"/api/objects/{obj_id}").json()
    assert len(details2["references"]) == 0

    # Clean up object
    client.delete(f"/api/objects/{obj_id}")


def test_visual_search_endpoint_multipart() -> None:
    """Verify POST /api/visual-search with frame upload."""
    # First register an object
    target_img = create_encoded_jpeg(bgr_color=(0, 255, 255))
    reg_resp = client.post(
        "/api/objects",
        json={"name": "Highlighter", "category": "personal", "image_base64": base64.b64encode(target_img).decode("utf-8")},
    )
    obj_id = reg_resp.json()["id"]

    # Construct frame containing the target
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    sub = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(sub, (10, 10), (90, 90), (0, 255, 255), -1)
    frame[50:150, 50:150] = sub
    _, frame_buf = cv2.imencode(".jpg", frame)

    files = {"file": ("camera_frame.jpg", io.BytesIO(frame_buf.tobytes()), "image/jpeg")}
    search_resp = client.post("/api/visual-search", files=files, data={"save_observation": "true"})
    assert search_resp.status_code == 200
    res_data = search_resp.json()
    assert "status" in res_data
    assert "latency_ms" in res_data
    assert "diagnostics" in res_data

    # Check recent matches endpoint
    recent_resp = client.get("/api/visual-search/recent")
    assert recent_resp.status_code == 200
    assert isinstance(recent_resp.json(), list)

    # Clean up
    client.delete(f"/api/objects/{obj_id}")


def test_visual_search_endpoint_invalid_inputs() -> None:
    """Verify POST /api/visual-search handles corrupt or oversized files with HTTP 400."""
    # Missing both file and base64
    resp_empty = client.post("/api/visual-search", json={})
    assert resp_empty.status_code == 400

    # Corrupted non-image file
    corrupt_io = io.BytesIO(b"NOT_A_VALID_JPEG_HEADER")
    resp_corrupt = client.post(
        "/api/visual-search",
        files={"file": ("corrupt.jpg", corrupt_io, "image/jpeg")},
    )
    assert resp_corrupt.status_code == 400

    # Tiny image (< 32x32)
    tiny_img = np.zeros((16, 16, 3), dtype=np.uint8)
    _, tiny_buf = cv2.imencode(".jpg", tiny_img)
    resp_tiny = client.post(
        "/api/visual-search",
        files={"file": ("tiny.jpg", io.BytesIO(tiny_buf.tobytes()), "image/jpeg")},
    )
    assert resp_tiny.status_code == 400


def test_visual_diagnostics_endpoint() -> None:
    """Verify GET /api/visual-search/diagnostics returns pipeline configuration."""
    resp = client.get("/api/visual-search/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    assert "encoder_model" in data
    assert "encoder_dimension" in data
    assert "cached_references_count" in data
    assert "strong_threshold" in data
