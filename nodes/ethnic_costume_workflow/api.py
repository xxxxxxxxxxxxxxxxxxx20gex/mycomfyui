import asyncio
import base64
import json
import logging
import os
from io import BytesIO

import aiohttp
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

from comfy import utils as comfy_utils
from comfy_api.latest import IO
from comfy_api_nodes.apis.openai import (
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
MENTALOUT_TASK_API_BASE = "https://image.mentalout.top"
DEFAULT_TEXT_MODEL = "gpt-4o-mini"
IMAGE_MODEL = "gpt-image-2"
TEXT_MODEL_RETRY_STATUSES = {429, 500, 502, 503, 504}
TEXT_MODEL_MAX_ATTEMPTS = 4
PACKY_COMPAT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
}

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), ".env"))


def _openai_base_url(base_url: str | None = None) -> str:
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_DEFAULT_BASE_URL).strip()
    base_url = base_url.rstrip("/")
    if "mentalout.top" in base_url.lower() and "/v1" not in base_url.lower():
        base_url = f"{base_url}/v1"
    return base_url


def _openai_api_url(path: str, base_url: str | None = None) -> str:
    base_url = _openai_base_url(base_url)
    normalized_path = path.strip("/")
    if base_url.endswith(f"/{normalized_path}"):
        return base_url
    return f"{base_url}/{normalized_path}"


def _chat_base_url(base_url: str | None = None) -> str:
    base_url = (base_url or os.environ.get("BASE_URL") or OPENAI_DEFAULT_BASE_URL).strip().rstrip("/")
    return base_url


def _chat_api_url(path: str, base_url: str | None = None) -> str:
    base_url = _chat_base_url(base_url)
    normalized_path = path.strip("/")
    if base_url.endswith(f"/{normalized_path}"):
        return base_url
    return f"{base_url}/{normalized_path}"


def _uses_mentalout_batch_api(base_url: str | None = None) -> bool:
    mode = os.environ.get("OPENAI_IMAGE_API_MODE", "").strip().lower()
    if mode in {"mentalout_batch", "batch", "task"}:
        return True
    if mode in {"sync", "openai", "images"}:
        return False
    return "chunfeng.mentalout.top" in _openai_base_url(base_url).lower()


def _mentalout_api_origin(base_url: str | None = None) -> str:
    base_url = _openai_base_url(base_url)
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url.rstrip("/")


def _mentalout_task_api_base() -> str:
    return (
        os.environ.get("MENTALOUT_TASK_API_BASE")
        or os.environ.get("OPENAI_IMAGE_TASK_API_BASE")
        or MENTALOUT_TASK_API_BASE
    ).strip().rstrip("/")


def _require_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")
    return api_key


def _require_chat_api_key() -> str:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is not set in .env")
    return api_key


def _text_model() -> str:
    return (
        os.environ.get("TEXT_MODEL")
        or os.environ.get("MODEL")
        or os.environ.get("OPENAI_TEXT_MODEL")
        or DEFAULT_TEXT_MODEL
    ).strip()


def _short_response_text(text: str, limit: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


async def _download_url_to_bytesio(url: str, dest: BytesIO, timeout: int | None = None) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), trust_env=True) as session:
        async with session.get(url, headers=PACKY_COMPAT_HEADERS) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"Image download failed ({resp.status}): {_short_response_text(text)}")
            dest.write(await resp.read())
            dest.seek(0)


def _downscale_image_tensor(image: torch.Tensor, total_pixels: int = 2048 * 2048) -> torch.Tensor:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    batch, height, width, channels = image.shape
    pixels = height * width
    if pixels <= total_pixels:
        return image

    scale = (total_pixels / pixels) ** 0.5
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))
    nchw = image.movedim(-1, 1)
    resized = torch.nn.functional.interpolate(nchw, size=(new_height, new_width), mode="bilinear", align_corners=False)
    return resized.movedim(1, -1)


def _parse_openai_image_response(
    response_text: str,
    status: int,
    content_type: str | None,
    operation: str,
) -> OpenAIImageGenerationResponse:
    try:
        raw_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        preview = _short_response_text(response_text) or "<empty response body>"
        raise RuntimeError(
            f"{operation} returned a non-JSON response "
            f"(status={status}, content_type={content_type or 'unknown'}): {preview}"
        ) from exc

    if status >= 400:
        error = raw_data.get("error", raw_data) if isinstance(raw_data, dict) else raw_data
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise RuntimeError(f"{operation} failed ({status}): {message}")

    try:
        response_data = OpenAIImageGenerationResponse.model_validate(raw_data)
        if response_data.data:
            return response_data
    except Exception as exc:
        preview = _short_response_text(response_text) or "<empty response body>"
        raise RuntimeError(
            f"{operation} returned JSON in an unsupported format "
            f"(status={status}, content_type={content_type or 'unknown'}): {preview}"
        ) from exc

    preview = _short_response_text(response_text) or "<empty response body>"
    raise RuntimeError(
        f"{operation} returned JSON without image data "
        f"(status={status}, content_type={content_type or 'unknown'}): {preview}"
    )


def _extract_chat_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


async def call_text_model(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": _text_model(),
        "messages": messages,
    }
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        last_error = ""
        for attempt in range(1, TEXT_MODEL_MAX_ATTEMPTS + 1):
            async with session.post(
                _chat_api_url("chat/completions"),
                headers={
                    "Authorization": f"Bearer {_require_chat_api_key()}",
                    "Content-Type": "application/json",
                    **PACKY_COMPAT_HEADERS,
                },
                json=payload,
            ) as resp:
                response_text = await resp.text()
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    preview = _short_response_text(response_text) or "<empty response body>"
                    raise RuntimeError(
                        f"Prompt optimization returned a non-JSON response "
                        f"(status={resp.status}, content_type={resp.headers.get('Content-Type') or 'unknown'}): {preview}"
                    ) from exc

                if resp.status >= 400:
                    error = data.get("error", data) if isinstance(data, dict) else data
                    message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                    last_error = f"Prompt optimization failed ({resp.status}): {message}"
                    if resp.status in TEXT_MODEL_RETRY_STATUSES and attempt < TEXT_MODEL_MAX_ATTEMPTS:
                        retry_after = resp.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdecimal() else min(2 ** attempt, 12)
                        logging.warning(
                            "Prompt optimization retry %s/%s after status %s: %s",
                            attempt + 1,
                            TEXT_MODEL_MAX_ATTEMPTS,
                            resp.status,
                            message,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(last_error)

                text = _extract_chat_text(data)
                if not text:
                    raise RuntimeError(f"Prompt optimization returned empty text: {_short_response_text(response_text)}")
                return text

        raise RuntimeError(last_error or "Prompt optimization failed after retries")


def _resolve_mentalout_image_url(task_api_base: str, image_url: str) -> str:
    image_url = (image_url or "").strip()
    if not image_url:
        return ""
    if image_url.startswith(("http://", "https://", "data:image/")):
        return image_url
    return f"{task_api_base.rstrip('/')}/{image_url.lstrip('/')}"


def _mentalout_batch_status(batch: dict) -> str:
    status = str(batch.get("status") or "").lower()
    if status in {"succeeded", "failed", "canceled"}:
        return status

    tasks = batch.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return status or "running"

    total = len(tasks)
    succeeded = sum(1 for task in tasks if str(task.get("status") or "").lower() == "succeeded")
    failed = sum(1 for task in tasks if str(task.get("status") or "").lower() == "failed")
    canceled = sum(1 for task in tasks if str(task.get("status") or "").lower() == "canceled")
    if succeeded + failed + canceled >= total:
        if canceled:
            return "canceled"
        if failed:
            return "failed"
        if succeeded:
            return "succeeded"
    return status or "running"


def _mentalout_batch_error(batch: dict) -> str:
    errors = []
    for task in batch.get("tasks") or []:
        for key in ("errorMessage", "debugMessage", "error"):
            value = task.get(key)
            if value:
                errors.append(str(value))
                break
    return "; ".join(errors) or str(batch.get("error") or "后台任务失败")


def _mentalout_batch_images(batch: dict, task_api_base: str) -> list[dict[str, str]]:
    items = []
    tasks = batch.get("tasks") if isinstance(batch, dict) else None
    if isinstance(tasks, list):
        for task in sorted(tasks, key=lambda item: item.get("index") or 0):
            if str(task.get("status") or "").lower() != "succeeded":
                continue
            image_url = _resolve_mentalout_image_url(task_api_base, str(task.get("imageUrl") or ""))
            if image_url:
                items.append({"url": image_url})

    if not items:
        for image_url in batch.get("imageUrls") or []:
            image_url = _resolve_mentalout_image_url(task_api_base, str(image_url or ""))
            if image_url:
                items.append({"url": image_url})

    return items


async def _mentalout_post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    operation: str,
) -> dict:
    async with session.post(
        url,
        headers={"Content-Type": "application/json", **PACKY_COMPAT_HEADERS},
        json=payload,
    ) as resp:
        response_text = await resp.text()
        try:
            data = json.loads(response_text) if response_text else {}
        except json.JSONDecodeError as exc:
            preview = _short_response_text(response_text) or "<empty response body>"
            raise RuntimeError(
                f"{operation} returned a non-JSON response "
                f"(status={resp.status}, content_type={resp.headers.get('Content-Type') or 'unknown'}): {preview}"
            ) from exc
        if resp.status >= 400:
            message = data.get("error") or data.get("message") or str(data)
            raise RuntimeError(f"{operation} failed ({resp.status}): {message}")
        return data


async def _mentalout_submit_batch(
    session: aiohttp.ClientSession,
    task_api_base: str,
    payload: dict,
    files: list[tuple[str, tuple[str, BytesIO, str]]] | None = None,
) -> dict:
    url = f"{task_api_base}/api/batches"
    if not files:
        return await _mentalout_post_json(session, url, payload, "Image task submission")

    form = aiohttp.FormData()
    form.add_field("payload", json.dumps(payload, ensure_ascii=False))
    for _field_name, (filename, file_obj, content_type) in files:
        file_obj.seek(0)
        form.add_field("image", file_obj, filename=filename, content_type=content_type)

    async with session.post(url, data=form, headers=PACKY_COMPAT_HEADERS) as resp:
        response_text = await resp.text()
        try:
            data = json.loads(response_text) if response_text else {}
        except json.JSONDecodeError as exc:
            preview = _short_response_text(response_text) or "<empty response body>"
            raise RuntimeError(
                f"Image task submission returned a non-JSON response "
                f"(status={resp.status}, content_type={resp.headers.get('Content-Type') or 'unknown'}): {preview}"
            ) from exc
        if resp.status >= 400:
            message = data.get("error") or data.get("message") or str(data)
            raise RuntimeError(f"Image task submission failed ({resp.status}): {message}")
        return data


async def _mentalout_fetch_batch_statuses(
    session: aiohttp.ClientSession,
    task_api_base: str,
    batch_ids: list[str],
) -> list[dict]:
    data = await _mentalout_post_json(
        session,
        f"{task_api_base}/api/batches/status",
        {"ids": batch_ids},
        "Image task status query",
    )
    items = data.get("items")
    return items if isinstance(items, list) else []


async def _generate_image_with_mentalout_batch(
    data: OpenAIImageGenerationRequest | OpenAIImageEditRequest,
    files: list[tuple[str, tuple[str, BytesIO, str]]] | None = None,
    base_url: str | None = None,
    node_cls: type[IO.ComfyNode] | None = None,
) -> OpenAIImageGenerationResponse:
    task_api_base = _mentalout_task_api_base()
    payload = {
        "apiBaseUrl": _mentalout_api_origin(base_url),
        "apiKey": _require_openai_api_key(),
        "prompt": data.prompt,
        "style": "",
        "size": data.size or "auto",
        "count": data.n or 1,
        "tuning": {
            "quality": data.quality or "",
            "outputFormat": data.output_format or "",
            "outputCompression": data.output_compression
            if isinstance(data.output_compression, int)
            else None,
        },
    }

    timeout = aiohttp.ClientTimeout(total=3600, sock_connect=30, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        reference_files = [file_info for file_info in files or [] if file_info[0] == "image"]
        progress_bar = None
        if node_cls is not None:
            try:
                progress_bar = comfy_utils.ProgressBar(100, node_id=node_cls.hidden.unique_id)
                progress_bar.update_absolute(1)
            except Exception:
                progress_bar = None

        batch = await _mentalout_submit_batch(session, task_api_base, payload, reference_files)
        batch_id = str(batch.get("id") or "")
        if not batch_id:
            raise RuntimeError(f"Image task submission returned no batch id: {batch}")
        logging.info(
            "Ethnic costume image task submitted: batch_id=%s count=%s references=%s",
            batch_id,
            payload["count"],
            len(reference_files),
        )

        last_logged_status = ""
        for attempt in range(1800):
            status = _mentalout_batch_status(batch)
            image_items = _mentalout_batch_images(batch, task_api_base)
            if node_cls is not None:
                try:
                    if progress_bar is not None:
                        progress_bar.update_absolute(100 if status == "succeeded" else min(95, 5 + attempt))
                except Exception:
                    pass

            if status != last_logged_status or image_items:
                logging.info(
                    "Ethnic costume image task status: batch_id=%s status=%s images=%s",
                    batch_id,
                    status,
                    len(image_items),
                )
                last_logged_status = status
            if status == "succeeded" and image_items:
                return OpenAIImageGenerationResponse.model_validate({"data": image_items})
            if status in {"failed", "canceled"}:
                raise RuntimeError(f"Image task {status}: {_mentalout_batch_error(batch)}")

            await asyncio.sleep(2)
            batches = await _mentalout_fetch_batch_statuses(session, task_api_base, [batch_id])
            if batches:
                batch = batches[0]

    raise RuntimeError(f"Image task timed out before completion: {batch_id}")


async def generate_image(
    prompt: str,
    *,
    quality: str = "medium",
    size: str = "1024x1536",
    background: str = "opaque",
    n: int = 1,
    node_cls: type[IO.ComfyNode] | None = None,
) -> OpenAIImageGenerationResponse:
    data = OpenAIImageGenerationRequest(
        model=IMAGE_MODEL,
        prompt=prompt,
        quality=quality,
        background=background,
        n=n,
        size=size,
        moderation="low",
    )
    if _uses_mentalout_batch_api():
        return await _generate_image_with_mentalout_batch(data, node_cls=node_cls)

    payload = data.model_dump(exclude_none=True)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.post(
            _openai_api_url("images/generations"),
            headers={
                "Authorization": f"Bearer {_require_openai_api_key()}",
                "Content-Type": "application/json",
                **PACKY_COMPAT_HEADERS,
            },
            json=payload,
        ) as resp:
            response_text = await resp.text()
            return _parse_openai_image_response(
                response_text,
                resp.status,
                resp.headers.get("Content-Type"),
                "Image generation",
            )


async def edit_image(
    prompt: str,
    files: list[tuple[str, tuple[str, BytesIO, str]]],
    *,
    quality: str = "medium",
    size: str = "1024x1536",
    background: str = "opaque",
    n: int = 1,
    node_cls: type[IO.ComfyNode] | None = None,
) -> OpenAIImageGenerationResponse:
    data = OpenAIImageEditRequest(
        model=IMAGE_MODEL,
        prompt=prompt,
        quality=quality,
        background=background,
        n=n,
        size=size,
        moderation="low",
    )
    if _uses_mentalout_batch_api():
        return await _generate_image_with_mentalout_batch(data, files, node_cls=node_cls)

    form = aiohttp.FormData()
    for key, value in data.model_dump(exclude_none=True).items():
        form.add_field(key, str(value))

    for field_name, (filename, file_obj, content_type) in files:
        file_obj.seek(0)
        form.add_field(field_name, file_obj, filename=filename, content_type=content_type)

    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.post(
            _openai_api_url("images/edits"),
            headers={"Authorization": f"Bearer {_require_openai_api_key()}", **PACKY_COMPAT_HEADERS},
            data=form,
        ) as resp:
            response_text = await resp.text()
            return _parse_openai_image_response(
                response_text,
                resp.status,
                resp.headers.get("Content-Type"),
                "Image editing",
            )


async def response_to_tensor(response: OpenAIImageGenerationResponse, timeout: int | None = None) -> torch.Tensor:
    data = response.data
    if not data:
        raise ValueError("No images returned from image endpoint")

    tensors = []
    for item in data:
        if item.b64_json:
            img_io = BytesIO(base64.b64decode(item.b64_json))
        elif item.url:
            img_io = BytesIO()
            await _download_url_to_bytesio(item.url, img_io, timeout=timeout)
        else:
            raise ValueError("Invalid image payload: neither URL nor base64 data present")

        pil_img = Image.open(img_io).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))

    return torch.stack(tensors, dim=0)


def image_tensor_to_edit_files(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> list[tuple[str, tuple[str, BytesIO, str]]]:
    image_tensors: list[torch.Tensor] = []
    if len(image.shape) == 4:
        image_tensors.extend(image[i : i + 1] for i in range(image.shape[0]))
    else:
        image_tensors.append(image.unsqueeze(0))

    files: list[tuple[str, tuple[str, BytesIO, str]]] = []
    for i, single_image in enumerate(image_tensors):
        scaled_image = _downscale_image_tensor(single_image, total_pixels=2048 * 2048).squeeze()
        image_np = (scaled_image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        field_name = "image" if len(image_tensors) == 1 else "image[]"
        files.append((field_name, (f"image_{i}.png", img_byte_arr, "image/png")))

    if mask is not None:
        if len(image_tensors) != 1:
            raise Exception("Cannot use a mask with multiple images")
        ref_image = image_tensors[0]
        image_height, image_width = ref_image.shape[1:-1]
        mask_2d = mask.squeeze().cpu()

        if tuple(mask_2d.shape) == (64, 64) and torch.count_nonzero(mask_2d).item() == 0:
            return files

        if tuple(mask_2d.shape) != (image_height, image_width):
            logging.warning(
                "Ignoring mask with mismatched size: mask=%s image=%sx%s",
                tuple(mask_2d.shape),
                image_width,
                image_height,
            )
            return files

        height, width = mask_2d.shape
        rgba_mask = torch.zeros(height, width, 4, device="cpu")
        rgba_mask[:, :, 3] = 1 - mask_2d
        scaled_mask = _downscale_image_tensor(rgba_mask.unsqueeze(0), total_pixels=2048 * 2048).squeeze()
        mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_np)
        mask_img_byte_arr = BytesIO()
        mask_img.save(mask_img_byte_arr, format="PNG")
        mask_img_byte_arr.seek(0)
        files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

    return files
