import os
import hashlib
import logging
from pathlib import Path

import requests
from django.conf import settings

def get_ipfs_api_url():
    """Get IPFS API URL dynamically"""
    return f"http://{settings.IPFS_HOST}:{settings.IPFS_PORT}/api/v0"

logger = logging.getLogger(__name__)

FALLBACK_DIR = Path(settings.MEDIA_ROOT) / "ipfs_fallback"
FALLBACK_PREFIX = "local-"

def _store_fallback_file(path: str) -> dict:
    """
    Store the file locally when IPFS is unavailable.
    Returns a dict mimicking IPFS /add response.
    """
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

    with open(path, "rb") as src:
        data = src.read()
    cid = hashlib.sha256(data).hexdigest()
    dest_path = FALLBACK_DIR / cid
    if not dest_path.exists():
        with open(dest_path, "wb") as dest:
            dest.write(data)

    logger.info("Stored file in local IPFS fallback: %s", dest_path)
    return {
        "Name": os.path.basename(path),
        "Hash": f"{FALLBACK_PREFIX}{cid}",
        "LocalPath": str(dest_path),
    }

def add_file(path, mfs_path=None):
    """
    Upload a file to IPFS via HTTP API.
    If mfs_path is provided, copy file into MFS so it appears in WebUI.
    Returns the JSON response containing 'Name' and 'Hash'.
    """
    api_url = get_ipfs_api_url()
    try:
        with open(path, "rb") as f:
            files = {"file": f}
            res = requests.post(f"{api_url}/add?pin=true", files=files, timeout=10)
        res.raise_for_status()
        data = res.json()
        cid = data["Hash"]

        if mfs_path:
            folder = os.path.dirname(mfs_path)
            try:
                requests.post(f"{api_url}/files/mkdir?arg={folder}&parents=true", timeout=5)
                requests.post(f"{api_url}/files/rm?arg={mfs_path}&force=true", timeout=5)
                requests.post(f"{api_url}/files/cp?arg=/ipfs/{cid}&arg={mfs_path}", timeout=5)
            except Exception as mfs_error:
                logger.warning("IPFS MFS operations failed: %s", mfs_error)

        return data

    except Exception as e:
        logger.warning("IPFS upload failed (%s). Falling back to local storage.", e)
        return _store_fallback_file(path)

def get_file(cid):
    """
    Retrieve raw file bytes from IPFS via HTTP API.
    """
    if cid.startswith(FALLBACK_PREFIX):
        fallback_path = FALLBACK_DIR / cid[len(FALLBACK_PREFIX):]
        if fallback_path.exists():
            with open(fallback_path, "rb") as f:
                return f.read()
        raise FileNotFoundError(f"Local fallback file not found for CID {cid}")

    api_url = get_ipfs_api_url()
    res = requests.post(f"{api_url}/cat?arg={cid}", timeout=10)
    res.raise_for_status()
    return res.content
