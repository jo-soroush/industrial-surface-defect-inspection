"""Unified inspection routes for the future /inspect/image endpoint."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.app.schemas.inspection import ImageInspectionResponse
from src.inspection_ai.inspection_pipeline import MAX_UPLOAD_BYTES, inspect_image


router = APIRouter()


@router.post("/inspect/image", response_model=ImageInspectionResponse)
async def inspect_uploaded_image(
    file: UploadFile | None = File(default=None),
) -> ImageInspectionResponse:
    """Inspect one uploaded image with the unified governed pipeline."""
    request_id = str(uuid4())

    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing image upload. Use multipart field 'file'.",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image. Received content type: {content_type!r}.",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty or missing.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte limit.",
        )

    try:
        return inspect_image(
            image_bytes=payload,
            filename=file.filename,
            content_type=content_type,
            request_id=request_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unified inspection failed unexpectedly.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unified inspection failed unexpectedly.",
        ) from None
