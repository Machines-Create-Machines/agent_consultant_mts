"""WebSocket chat: JWT auth, dialog persist, agent via to_thread."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder

from auth.utils import decode_jwt
import services.agent_bridge as agent_bridge
from services.dialog_service import DialogService

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["websocket"])


def _json_safe(obj: Any) -> Any:
    """Serialize DB rows (datetime/UUID) for websocket JSON frames."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    return obj


async def _send(ws: WebSocket, payload: dict) -> None:
    await ws.send_json(_json_safe(payload))


def _history_for_agent(messages: list[dict], question: str) -> list[dict]:
    """Prior turns only: [{role, content}, ...] excluding the current user question."""
    history: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        history.append({"role": role, "content": content})
    if history and history[-1]["role"] == "user" and history[-1]["content"] == question.strip():
        history = history[:-1]
    return history


@ws_router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token") or ""
    user = decode_jwt(token) if token else None
    if user is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    busy = False

    try:
        await _send(websocket, {"type": "ready", "user_id": str(user.user_id)})

        while True:
            raw = await websocket.receive_json()
            if not isinstance(raw, dict):
                await _send(websocket, {"type": "error", "detail": "Ожидался JSON-объект"})
                continue

            msg_type = raw.get("type")
            if msg_type == "ping":
                await _send(websocket, {"type": "pong"})
                continue

            if msg_type != "user_message":
                await _send(websocket, {"type": "error", "detail": f"Неизвестный type: {msg_type}"})
                continue

            if busy:
                await _send(
                    websocket,
                    {"type": "error", "detail": "Дождитесь ответа ассистента"},
                )
                continue

            content = (raw.get("content") or "").strip()
            if not content:
                await _send(websocket, {"type": "error", "detail": "Пустое сообщение"})
                continue

            dialog_id = raw.get("dialog_id")
            if dialog_id is not None:
                dialog_id = str(dialog_id).strip() or None

            busy = True
            try:
                created_dialog = None
                if dialog_id is None:
                    title = content[:80] + ("…" if len(content) > 80 else "")
                    created_dialog = DialogService.create_dialog(user, title=title)
                    dialog_id = str(created_dialog["id"])
                    await _send(
                        websocket,
                        {"type": "dialog", **jsonable_encoder(created_dialog)},
                    )

                user_msg = DialogService.add_user_message(dialog_id, content, user)
                await _send(
                    websocket,
                    {
                        "type": "user_message",
                        **jsonable_encoder(user_msg),
                    },
                )

                await _send(websocket, {"type": "status", "state": "thinking"})

                messages = DialogService.get_messages(dialog_id, user)
                history = _history_for_agent(messages, content)

                result = await asyncio.to_thread(agent_bridge.answer, content, history)
                answer_text = result.get("answer") or ""
                documents = result.get("documents") or []

                assistant_msg = DialogService.add_assistant_message(
                    dialog_id, answer_text, documents, user
                )
                await _send(
                    websocket,
                    {
                        "type": "assistant_message",
                        **jsonable_encoder(assistant_msg),
                    },
                )
                await _send(websocket, {"type": "status", "state": "done"})
            except Exception as exc:  # noqa: BLE001
                logger.exception("ws_chat handler error: %s", exc)
                detail = getattr(exc, "detail", None) or str(exc) or "Ошибка обработки сообщения"
                try:
                    await _send(websocket, {"type": "error", "detail": detail})
                    await _send(websocket, {"type": "status", "state": "done"})
                except Exception:  # noqa: BLE001
                    pass
            finally:
                busy = False

    except WebSocketDisconnect:
        logger.info("ws_chat disconnected user_id=%s", user.user_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ws_chat fatal: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass
