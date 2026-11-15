from fastapi import HTTPException
from db.dialog_base import DialogConnection

dialog_base = DialogConnection()


class DialogService:
    """Бизнес-логика диалогов. Ответы LLM здесь пока не генерируются —
    место для будущего подключения LLM отмечено в add_user_message."""

    @staticmethod
    def _user_id(current_user) -> str:
        return str(current_user.user_id)

    @staticmethod
    def create_dialog(current_user, title: str = "Новый диалог"):
        return dialog_base.create_dialog(DialogService._user_id(current_user), title)

    @staticmethod
    def get_dialogs(current_user):
        return dialog_base.get_user_dialogs(DialogService._user_id(current_user))

    @staticmethod
    def delete_dialog(dialog_id: str, current_user):
        if not dialog_base.delete_dialog(dialog_id, DialogService._user_id(current_user)):
            raise HTTPException(status_code=404, detail="Диалог не найден")
        return {"message": "Диалог удалён"}

    @staticmethod
    def rename_dialog(dialog_id: str, title: str, current_user):
        dialog = dialog_base.rename_dialog(dialog_id, DialogService._user_id(current_user), title)
        if not dialog:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        return dialog

    @staticmethod
    def get_messages(dialog_id: str, current_user):
        messages = dialog_base.get_dialog_messages(dialog_id, DialogService._user_id(current_user))
        if messages is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        return messages

    @staticmethod
    def add_user_message(dialog_id: str, content: str, current_user):
        user_id = DialogService._user_id(current_user)
        msg = dialog_base.add_message(dialog_id, user_id, "user", content)
        if msg is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        # TODO: здесь будет вызов LLM: формирование ответа и список
        # использованных документов, которые сохранятся в поле documents.
        return msg

    @staticmethod
    def add_assistant_message(dialog_id: str, content: str, documents, current_user):
        user_id = DialogService._user_id(current_user)
        msg = dialog_base.add_message(dialog_id, user_id, "assistant", content, documents)
        if msg is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        return msg
