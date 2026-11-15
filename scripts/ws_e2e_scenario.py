"""Minimal WebSocket e2e for consult→CTA→lead without third-party deps."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import sys
import urllib.request
from typing import Any
from urllib.parse import urlparse


def _http_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


class MiniWS:
    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self.sock: socket.socket | None = None
        self._pending = b""

    def connect(self) -> None:
        raw = socket.create_connection((self.host, self.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        raw.sendall(req.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = raw.recv(4096)
            if not chunk:
                raise RuntimeError("WS handshake closed")
            buf += chunk
        header, rest = buf.split(b"\r\n\r\n", 1)
        if b"101" not in header.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"WS handshake failed: {header[:200]!r}")
        self.sock = raw
        self._pending = rest

    def send_json(self, obj: dict[str, Any]) -> None:
        assert self.sock is not None
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < (1 << 16):
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def recv_json(self, timeout: float = 120.0) -> dict[str, Any]:
        assert self.sock is not None
        self.sock.settimeout(timeout)
        data = self._pending
        self._pending = b""
        while len(data) < 2:
            data += self.sock.recv(4096)
        b1, b2 = data[0], data[1]
        data = data[2:]
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            while len(data) < 2:
                data += self.sock.recv(4096)
            length = struct.unpack("!H", data[:2])[0]
            data = data[2:]
        elif length == 127:
            while len(data) < 8:
                data += self.sock.recv(4096)
            length = struct.unpack("!Q", data[:8])[0]
            data = data[8:]
        if masked:
            while len(data) < 4:
                data += self.sock.recv(4096)
            mask = data[:4]
            data = data[4:]
        else:
            mask = None
        while len(data) < length:
            data += self.sock.recv(4096)
        payload, self._pending = data[:length], data[length:]
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:
            raise RuntimeError("WS closed")
        if opcode == 0x9:  # ping -> pong omitted for e2e
            return self.recv_json(timeout)
        return json.loads(payload.decode("utf-8"))

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass


def turn(ws: MiniWS, content: str, dialog_id: str | None = None) -> tuple[str | None, str, dict | None]:
    payload: dict[str, Any] = {"type": "user_message", "content": content, "debug": True}
    if dialog_id:
        payload["dialog_id"] = dialog_id
    ws.send_json(payload)
    answer = ""
    dbg = None
    did = dialog_id
    while True:
        data = ws.recv_json()
        t = data.get("type")
        if t == "dialog":
            did = str(data.get("id") or did)
        elif t == "assistant_message":
            answer = str(data.get("content") or "")
            dbg = data.get("debug") if isinstance(data.get("debug"), dict) else None
        elif t == "status" and data.get("state") == "done":
            break
        elif t == "error":
            raise RuntimeError(data)
    return did, answer, dbg


def main() -> int:
    base = os.environ.get("BASE_URL", "http://127.0.0.1:8080")
    api = base.rstrip("/") + "/api"
    auth = _http_json(api + "/login", {"login": "admin", "password": "admin"})
    token = auth["access_token"]
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.rstrip("/") + f"/ws/chat?token={token}"
    ws = MiniWS(ws_url)
    ws.connect()
    ready = ws.recv_json(timeout=10)
    if ready.get("type") != "ready":
        raise SystemExit(f"expected ready, got {ready}")
    steps = [
        "хотим снизить счета за электричество на производстве что предложите",
        "производство, важнее снизить счета",
        "да, передайте менеджеру",
        "Иван Петров",
        "+79991234567",
        "ООО ТестЭнерго",
    ]
    did = None
    for i, q in enumerate(steps, 1):
        did, ans, dbg = turn(ws, q, did)
        print(f"\n=== step {i} ===")
        print("Q:", q)
        print("A:", ans[:240].replace("\n", " "))
        if dbg:
            print("debug:", {k: dbg.get(k) for k in ("total_ms", "route", "outcome", "top_score", "degraded_retrieval", "cited_ids")})
    ws.close()
    print("\nWS_E2E_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print("WS_E2E_FAIL", type(exc).__name__, exc, file=sys.stderr)
        raise
