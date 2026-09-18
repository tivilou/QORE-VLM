#!/usr/bin/env python3
"""Upload manifest-declared exchange artifacts to the authenticated service.

The helper is intentionally dependency-free so collaborator wrappers can use
their discovered experiment Python without installing another package.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
from urllib.parse import quote, urlsplit


DEFAULT_URL = "http://117.50.198.37:18083"
DEFAULT_TOKEN_ENV = "QORE_EXCHANGE_TOKEN"
CHUNK_SIZE = 1024 * 1024
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ExchangeUploadError(RuntimeError):
    """Raised when an exchange request or receipt validation fails."""


def _safe_target(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExchangeUploadError("exchange path must be a non-empty POSIX path")
    if value.startswith("/") or value.endswith("/"):
        raise ExchangeUploadError("exchange path must be relative and file-like")
    segments = value.split("/")
    if len(segments) < 2 or segments[0] != "five_ideas":
        raise ExchangeUploadError("exchange path must stay under five_ideas/")
    if any(
        segment in {"", ".", ".."}
        or segment.startswith(".")
        or not SAFE_SEGMENT.fullmatch(segment)
        for segment in segments
    ):
        raise ExchangeUploadError("exchange path contains an unsafe segment")
    return "/".join(segments)


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _connection(base_url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ExchangeUploadError("exchange URL must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ExchangeUploadError("exchange URL must not contain credentials or query data")
    port = parsed.port
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname, port=port, timeout=timeout
        )
    else:
        connection = http.client.HTTPConnection(parsed.hostname, port=port, timeout=timeout)
    prefix = parsed.path.rstrip("/")
    return connection, prefix


def _read_response(response: http.client.HTTPResponse) -> tuple[int, dict[str, Any]]:
    raw = response.read(2 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExchangeUploadError(
            f"exchange returned non-JSON response (HTTP {response.status})"
        ) from exc
    if not isinstance(payload, dict):
        raise ExchangeUploadError("exchange response must be a JSON object")
    return response.status, payload


def _request_json(
    base_url: str,
    token: str,
    method: str,
    endpoint: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    if not endpoint.startswith("/"):
        raise ExchangeUploadError("endpoint must be absolute")
    connection, prefix = _connection(base_url, timeout)
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Accept": "application/json",
        "Connection": "close",
    }
    try:
        connection.request(method, prefix + endpoint, body=body, headers=headers)
        status, response = _read_response(connection.getresponse())
    except OSError as exc:
        raise ExchangeUploadError(f"exchange {method} {endpoint} failed: {type(exc).__name__}") from exc
    finally:
        connection.close()
    if status not in {200, 201}:
        raise ExchangeUploadError(f"exchange {method} {endpoint} rejected request (HTTP {status})")
    return response


def _upload_one(
    base_url: str,
    token: str,
    local_path: Path,
    exchange_path: str,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    exchange_path = _safe_target(exchange_path)
    if not local_path.is_file() or local_path.is_symlink():
        raise ExchangeUploadError(f"local upload file is missing or is a symlink: {local_path}")
    expected_size, expected_hash = _sha256(local_path)
    connection, prefix = _connection(base_url, timeout)
    endpoint = "/upload/" + quote(exchange_path, safe="/")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(expected_size),
        "Accept": "application/json",
        "Connection": "close",
    }
    try:
        connection.putrequest("PUT", prefix + endpoint)
        for key, value in headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        with local_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                connection.send(chunk)
        status, receipt = _read_response(connection.getresponse())
    except OSError as exc:
        raise ExchangeUploadError(f"exchange PUT failed for {exchange_path}: {type(exc).__name__}") from exc
    finally:
        connection.close()
    if status not in {200, 201}:
        raise ExchangeUploadError(f"exchange PUT rejected {exchange_path} (HTTP {status})")
    if receipt.get("path") != exchange_path:
        raise ExchangeUploadError(f"receipt path mismatch for {exchange_path}")
    if int(receipt.get("size_bytes", -1)) != expected_size:
        raise ExchangeUploadError(f"receipt byte count mismatch for {exchange_path}")
    if receipt.get("sha256") != expected_hash:
        raise ExchangeUploadError(f"receipt SHA-256 mismatch for {exchange_path}")
    return receipt


def _manifest_files(manifest: Mapping[str, Any], base_dir: Path) -> Iterable[tuple[Path, str]]:
    target_directory = _safe_target(str(manifest.get("target_directory", "")) + "/placeholder")
    target_directory = target_directory.rsplit("/", 1)[0]
    entries = manifest.get("exchange_files", ())
    if not isinstance(entries, list):
        raise ExchangeUploadError("manifest exchange_files must be a list")
    for item in entries:
        if isinstance(item, str):
            name = item
            exchange_path = f"{target_directory}/{name}"
        elif isinstance(item, Mapping):
            name = item.get("name")
            exchange_path = item.get("exchange_path") or f"{target_directory}/{name}"
        else:
            raise ExchangeUploadError("manifest exchange_files contains an invalid entry")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ExchangeUploadError("manifest exchange file names must be plain file names")
        local_path = (base_dir / name).resolve()
        local_path.relative_to(base_dir.resolve())
        yield local_path, _safe_target(str(exchange_path))


def upload_manifest(
    manifest_path: Path,
    *,
    base_url: str | None = None,
    token: str | None = None,
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 300.0,
) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExchangeUploadError(f"cannot read upload manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ExchangeUploadError("upload manifest root must be an object")
    token = token or os.environ.get(token_env)
    if not token:
        raise ExchangeUploadError(f"missing bearer token environment variable: {token_env}")
    target_directory = _safe_target(str(manifest.get("target_directory", "")) + "/placeholder").rsplit("/", 1)[0]
    base_url = base_url or os.environ.get("QORE_EXCHANGE_URL", DEFAULT_URL)
    _request_json(
        base_url,
        token,
        "POST",
        "/api/directories",
        payload={"path": target_directory},
        timeout=timeout,
    )
    receipts = []
    for local_path, exchange_path in _manifest_files(manifest, manifest_path.parent):
        receipts.append(_upload_one(base_url, token, local_path, exchange_path, timeout=timeout))
    return receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", default=None)
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV)
    args = parser.parse_args(argv)
    try:
        receipts = upload_manifest(args.manifest, base_url=args.url, token_env=args.token_env)
    except ExchangeUploadError as exc:
        print(f"exchange upload error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "uploaded", "files": receipts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
