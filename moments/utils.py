import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse
from pathlib import Path

import jwt
import PIL
from flask import current_app, flash, redirect, request, url_for
from typing import Tuple
from jwt.exceptions import InvalidTokenError
from PIL import Image


def generate_token(user, operation, expiration=3600, **kwargs):
    payload = {
        'id': user.id,
        'operation': operation.value,
        'exp': datetime.now(timezone.utc) + timedelta(seconds=expiration)
    }
    payload.update(**kwargs)
    return jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')


def parse_token(user, token, operation):
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
    except InvalidTokenError:
        return {}

    if operation.value != payload.get('operation') or user.id != payload.get('id'):
        return {}
    return payload


def rename_image(old_filename):
    ext = Path(old_filename).suffix
    new_filename = uuid.uuid4().hex + ext
    return new_filename


def resize_image(image, filename, base_width):
    ext = Path(filename).suffix
    img = Image.open(image)
    if img.size[0] <= base_width:
        return filename
    w_percent = base_width / float(img.size[0])
    h_size = int(float(img.size[1]) * float(w_percent))
    img = img.resize((base_width, h_size), PIL.Image.LANCZOS)

    filename += current_app.config['MOMENTS_PHOTO_SUFFIXES'][base_width] + ext
    img.save(current_app.config['MOMENTS_UPLOAD_PATH'] / filename, optimize=True, quality=85)
    return filename


def validate_image(filename):
    ext = Path(filename).suffix.lower()
    allowed_extensions = current_app.config['DROPZONE_ALLOWED_FILE_TYPE'].split(',')
    return '.' in filename and ext in allowed_extensions


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def redirect_back(default='main.index', **kwargs):
    for target in request.args.get('next'), request.referrer:
        if not target:
            continue
        if is_safe_url(target):
            return redirect(target)
    return redirect(url_for(default, **kwargs))


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'Error in the {getattr(form, field).label.text} field - {error}')


def generate_caption_and_labels(image_path) -> Tuple[str | None, list[str]]:
    """Return (caption, labels) using Azure Vision if configured; otherwise fall back to None/empty.

    This is a minimal helper to keep upload route concise and non-blocking on errors.
    """
    endpoint = current_app.config.get('AZURE_VISION_ENDPOINT')
    key = current_app.config.get('AZURE_VISION_KEY')
    if not endpoint or not key:
        # Try local models if available
        caption_local, labels_local = _try_local_caption_and_labels(image_path)
        return caption_local, labels_local

    caption_url = f"{endpoint.rstrip('/')}/computervision/imageanalysis:analyze?api-version=2024-02-01&features=caption,objects,tags"

    try:
        current_app.logger.info('[vision] analyzing image via Azure endpoint=%s', endpoint)
        # Prefer requests if available
        try:
            import requests  # type: ignore
            headers = {
                'Ocp-Apim-Subscription-Key': key,
                'Content-Type': 'application/octet-stream',
            }
            with open(image_path, 'rb') as f:
                resp = requests.post(caption_url, headers=headers, data=f.read(), timeout=10)
            if resp.status_code >= 400:
                current_app.logger.warning('[vision] HTTP %s: %s', resp.status_code, resp.text[:500])
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Fallback to urllib (no external dependency)
            import json as _json  # local alias to avoid shadowing
            from urllib import request as _ureq, error as _uerr

            with open(image_path, 'rb') as f:
                body = f.read()
            req = _ureq.Request(caption_url, data=body, method='POST')
            req.add_header('Ocp-Apim-Subscription-Key', key)
            req.add_header('Content-Type', 'application/octet-stream')
            try:
                with _ureq.urlopen(req, timeout=10) as resp:  # nosec B310
                    raw = resp.read()
            except _uerr.HTTPError as e:
                err_body = e.read()
                current_app.logger.warning('[vision] HTTPError %s: %s', getattr(e, 'code', ''), err_body[:500])
                raw = err_body
                raise
            data = _json.loads(raw.decode('utf-8'))
        caption = None
        labels: list[str] = []
        # Azure response shape (2024-02-01): { 'captionResult': { 'text': '...', 'confidence': ... }, 'tagsResult': { 'values': [{'name': 'cat', ...}, ...] } }
        caption_result = data.get('captionResult') or {}
        if isinstance(caption_result, dict):
            caption = caption_result.get('text')
        tags_result = data.get('tagsResult') or {}
        values = tags_result.get('values') if isinstance(tags_result, dict) else []
        if isinstance(values, list):
            labels = [str(v.get('name', '')).strip().lower() for v in values if isinstance(v, dict) and v.get('name')]
        current_app.logger.info('[vision] caption=%r labels=%s', caption, labels[:10])
        return caption, labels[:10]
    except Exception as ex:
        # If cloud fails, try local models, otherwise silent fallback
        current_app.logger.warning('[vision] exception: %s', ex)
        caption_local, labels_local = _try_local_caption_and_labels(image_path)
        return caption_local, labels_local


# ---- Optional local ML fallback (no API key needed) ----
_BLIP_PIPELINE = None
_RESNET_MODEL = None
_RESNET_TRANSFORM = None
_RESNET_CATEGORIES: list[str] | None = None


def _try_local_caption_and_labels(image_path) -> Tuple[str | None, list[str]]:
    caption = None
    labels: list[str] = []
    try:
        caption = _local_generate_caption(image_path)
    except Exception:
        caption = None
    try:
        labels = _local_generate_labels(image_path)
    except Exception:
        labels = []
    return caption, labels


def _ensure_local_captioner():
    global _BLIP_PIPELINE
    if _BLIP_PIPELINE is not None:
        return
    from transformers import pipeline  # type: ignore
    # Small model for CPU; downloads weights on first use
    _BLIP_PIPELINE = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')


def _local_generate_caption(image_path) -> str | None:
    try:
        _ensure_local_captioner()
        result = _BLIP_PIPELINE(image_path)
        if isinstance(result, list) and result:
            text = result[0].get('generated_text') or result[0].get('text')
            if text:
                return str(text)
        return None
    except Exception:
        return None


def _ensure_local_labeler():
    global _RESNET_MODEL, _RESNET_TRANSFORM, _RESNET_CATEGORIES
    if _RESNET_MODEL is not None:
        return
    from torchvision.models import resnet50, ResNet50_Weights  # type: ignore
    from torchvision import transforms  # type: ignore

    weights = ResNet50_Weights.DEFAULT
    _RESNET_MODEL = resnet50(weights=weights)
    _RESNET_MODEL.eval()
    _RESNET_TRANSFORM = weights.transforms()  # includes resize/normalize
    meta = getattr(weights, 'meta', {})
    _RESNET_CATEGORIES = meta.get('categories')


def _local_generate_labels(image_path) -> list[str]:
    from PIL import Image
    _ensure_local_labeler()
    img = Image.open(image_path).convert('RGB')
    inp = _RESNET_TRANSFORM(img).unsqueeze(0)
    # torch is a transitive dep of torchvision
    import torch  # type: ignore

    with torch.no_grad():
        logits = _RESNET_MODEL(inp)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        topk = torch.topk(probs, k=10)
    labels: list[str] = []
    for idx in topk.indices.tolist():
        name = None
        if _RESNET_CATEGORIES and 0 <= idx < len(_RESNET_CATEGORIES):
            name = _RESNET_CATEGORIES[idx]
        else:
            name = str(idx)
        if name:
            labels.append(name.lower())
    return labels
