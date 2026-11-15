from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from services.dialog_service import DialogService
from auth.dependencies import require_authenticated_user, require_client_role

dialog_router = APIRouter(prefix="/dialogs", tags=["dialogs"])


class MessageIn(BaseModel):
    content: str = Field(..., min_length=1, max_length=20000)


class AssistantMessageIn(MessageIn):
    # Документы, использованные LLM при ответе. Формат задаёт модуль LLM,
    # здесь храним как есть: [{"id_cr": "...", "title": "..."}, ...]
    documents: List[dict] = Field(default_factory=list)


class RenameIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@dialog_router.post("")
async def create_dialog(current_user=Depends(require_authenticated_user)):
    """Создать новый (пустой) диалог для текущего пользователя."""
    return DialogService.create_dialog(current_user)


@dialog_router.get("")
async def list_dialogs(current_user=Depends(require_authenticated_user)):
    """Список диалогов текущего пользователя."""
    return DialogService.get_dialogs(current_user)


@dialog_router.get("/{dialog_id}/messages")
async def get_messages(dialog_id: str, current_user=Depends(require_authenticated_user)):
    """Сообщения диалога: текст, время отправки, использованные документы."""
    return DialogService.get_messages(dialog_id, current_user)


@dialog_router.post("/{dialog_id}/messages")
async def add_message(dialog_id: str, body: MessageIn,
                    current_user=Depends(require_authenticated_user)):
    """Добавить сообщение пользователя в диалог (только запись в БД).

    Чат с агентом идёт через WebSocket /ws/chat; этот эндпоинт —
    legacy/отладка без вызова LLM.
    """
    return DialogService.add_user_message(dialog_id, body.content, current_user)


@dialog_router.post("/{dialog_id}/messages/assistant")
async def add_assistant_message(dialog_id: str, body: AssistantMessageIn,
                                current_user=Depends(require_authenticated_user)):
    """Сохранить ответ ассистента и documents (без вызова агента)."""
    return DialogService.add_assistant_message(
        dialog_id, body.content, body.documents, current_user)


@dialog_router.patch("/{dialog_id}")
async def rename_dialog(dialog_id: str, body: RenameIn,
                        current_user=Depends(require_authenticated_user)):
    dialog = DialogService.rename_dialog(dialog_id, body.title, current_user)
    return dialog


@dialog_router.delete("/{dialog_id}")
async def delete_dialog(dialog_id: str, current_user=Depends(require_authenticated_user)):
    return DialogService.delete_dialog(dialog_id, current_user)
