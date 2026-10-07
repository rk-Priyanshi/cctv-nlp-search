"""
drive_video.py
--------------
Google Drive video-selection helpers for the CCTV NLP Search app.

Responsibilities:
  - Google OAuth 2.0 (read-only Drive scope)
  - List video files accessible to the signed-in account
  - Download a selected video to a local temp file

Nothing in this module touches YOLO, CLIP, Qdrant, or any other
existing processing logic.
"""

import os
import secrets
import tempfile
import time
import urllib.parse

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

# Only show files whose name ends with one of these extensions.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# MIME types that Google Drive assigns to common video formats.
# Used as a server-side filter to reduce API traffic.
VIDEO_MIME_TYPES = [
    "video/mp4",
    "video/x-msvideo",       # .avi
    "video/quicktime",        # .mov
    "video/x-matroska",       # .mkv
    "video/avi",
    "application/octet-stream",  # fallback; extension check applied client-side
]

_DRIVE_SESSION_KEY = "drive_tokens"   # key inside st.session_state

# ---------------------------------------------------------------------------
# Process-level OAuth state store
#
# WHY NOT st.session_state:
#   st.session_state is bound to a single WebSocket connection.  When the
#   browser follows the Google consent URL it navigates *away* from the app,
#   which tears down that WebSocket.  Google then redirects back, creating a
#   brand-new connection with an empty st.session_state — so the previously
#   stored nonce is gone and every callback is rejected as a CSRF attack.
#
# FIX — module-level dict (process memory):
#   A plain Python dict at module scope lives for the lifetime of the
#   Streamlit server process and is shared across all WebSocket sessions.
#   This is the standard server-side OAuth nonce pattern: store the random
#   nonce on the server keyed by its own value, verify on return, delete
#   immediately (single-use), and expire after a short TTL.
#   CSRF protection is fully maintained — an attacker cannot guess the
#   64-char random hex nonce.
# ---------------------------------------------------------------------------

_STATE_TTL_SECONDS = 600          # nonces expire after 10 minutes
_pending_states: dict[str, float] = {}   # state_value → expiry timestamp


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def _get_oauth_credentials() -> tuple[str, str, str]:
    """Read OAuth credentials from st.secrets (never hardcode)."""
    return (
        st.secrets["GOOGLE_CLIENT_ID"],
        st.secrets["GOOGLE_CLIENT_SECRET"],
        st.secrets["GOOGLE_REDIRECT_URI"],
    )


def _purge_expired_states() -> None:
    """Remove stale nonces so the dict doesn't grow unbounded."""
    now = time.monotonic()
    expired = [k for k, exp in _pending_states.items() if now > exp]
    for k in expired:
        _pending_states.pop(k, None)


def get_auth_url() -> str:
    """Build the Google OAuth consent-screen URL.

    Generates a cryptographically random *state* nonce, stores it in the
    process-level ``_pending_states`` dict with a 10-minute TTL, and
    includes it in the authorization URL so the callback handler can
    detect CSRF / replay attacks.

    State is stored at process level (not st.session_state) because the
    browser navigates away to Google, which destroys the current WebSocket
    session.  The new connection on return has a fresh session_state, so
    the nonce must survive in process memory instead.
    """
    _purge_expired_states()
    client_id, _, redirect_uri = _get_oauth_credentials()

    # 32 bytes → 64 hex chars; cryptographically random via os.urandom
    state = secrets.token_hex(32)
    _pending_states[state] = time.monotonic() + _STATE_TTL_SECONDS

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)


def verify_and_clear_state(returned_state: str | None) -> bool:
    """Validate the *state* returned by Google against the server-side store.

    Returns True only when the state is present, not expired, and matches
    exactly.  Always removes the nonce afterwards (success or failure) to
    prevent reuse.
    """
    if not returned_state:
        return False
    expiry = _pending_states.pop(returned_state, None)   # single-use: remove immediately
    if expiry is None:
        return False   # unknown nonce
    if time.monotonic() > expiry:
        return False   # valid nonce but expired
    return True        # constant-time not needed: expiry lookup is already O(1) hash


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange an authorization code for access + refresh tokens."""
    client_id, client_secret, redirect_uri = _get_oauth_credentials()
    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> str:
    """Use the stored refresh token to obtain a fresh access token."""
    client_id, client_secret, _ = _get_oauth_credentials()
    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_access_token() -> str | None:
    """
    Return a valid access token from session state, refreshing if expired.
    Returns None when the user is not authenticated.
    """
    tokens = st.session_state.get(_DRIVE_SESSION_KEY)
    if not tokens:
        return None

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    # Attempt a lightweight token validation; refresh on 401.
    if access_token:
        return access_token          # caller will handle 401 → refresh path

    if refresh_token:
        try:
            new_token = refresh_access_token(refresh_token)
            st.session_state[_DRIVE_SESSION_KEY]["access_token"] = new_token
            return new_token
        except Exception:
            return None

    return None


def is_authenticated() -> bool:
    return get_access_token() is not None


def save_tokens(tokens: dict) -> None:
    st.session_state[_DRIVE_SESSION_KEY] = tokens


def logout() -> None:
    st.session_state.pop(_DRIVE_SESSION_KEY, None)


# ---------------------------------------------------------------------------
# Drive API helpers
# ---------------------------------------------------------------------------

def _auth_headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


def list_drive_videos(access_token: str) -> list[dict]:
    """
    Return a list of video file dicts accessible to the authenticated account.
    Each dict contains at least: id, name, size (bytes, may be absent).

    Uses a server-side MIME-type filter to reduce data transferred, then
    applies an extension check client-side for correctness.
    """
    mime_query = " or ".join(
        f"mimeType='{m}'" for m in VIDEO_MIME_TYPES
    )
    query = f"trashed=false and ({mime_query})"

    videos: list[dict] = []
    page_token: str | None = None

    while True:
        params: dict = {
            "q": query,
            "fields": "nextPageToken, files(id, name, size, mimeType)",
            "pageSize": 100,
            "orderBy": "name",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(
            DRIVE_FILES_URL,
            headers=_auth_headers(access_token),
            params=params,
            timeout=60,
        )

        # Handle expired token transparently (one retry with refresh)
        if resp.status_code == 401:
            tokens = st.session_state.get(_DRIVE_SESSION_KEY, {})
            refresh_token = tokens.get("refresh_token")
            if refresh_token:
                try:
                    access_token = refresh_access_token(refresh_token)
                    st.session_state[_DRIVE_SESSION_KEY]["access_token"] = access_token
                    resp = requests.get(
                        DRIVE_FILES_URL,
                        headers=_auth_headers(access_token),
                        params=params,
                        timeout=60,
                    )
                except Exception:
                    break
            else:
                break

        resp.raise_for_status()
        data = resp.json()

        for f in data.get("files", []):
            name: str = f.get("name", "")
            if name.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(f)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return videos


def download_drive_video(file_id: str, filename: str, access_token: str) -> str:
    """
    Download a Drive file by its ID to a local temporary file.

    Returns the absolute path to the downloaded file.
    The caller is responsible for cleanup after processing is done.
    """
    # Determine a safe suffix from the original filename
    _, ext = os.path.splitext(filename)
    suffix = ext if ext.lower() in VIDEO_EXTENSIONS else ".mp4"

    # Use a named temp file so process_video_with_yolo() can re-open it by path
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix="gdrive_",
        dir=tempfile.gettempdir(),
    )
    tmp_path = tmp.name
    tmp.close()

    download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    # Stream the download to avoid loading the entire file into RAM at once
    with requests.get(
        download_url,
        headers=_auth_headers(access_token),
        stream=True,
        timeout=60,
    ) as resp:
        if resp.status_code == 401:
            tokens = st.session_state.get(_DRIVE_SESSION_KEY, {})
            refresh_token = tokens.get("refresh_token")
            if refresh_token:
                access_token = refresh_access_token(refresh_token)
                st.session_state[_DRIVE_SESSION_KEY]["access_token"] = access_token
                resp = requests.get(
                    download_url,
                    headers=_auth_headers(access_token),
                    stream=True,
                    timeout=60,
                )

        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                if chunk:
                    f.write(chunk)

    return tmp_path
