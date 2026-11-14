import logging
import os
import sys

# Позволяет импортировать модули как с префиксом db.* (из корня проекта),
# так и напрямую (postgres), как это делает init_db.py.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from psycopg2.extras import RealDictCursor

try:
    from db.postgres import DataConnection
except ImportError:  # запуск, когда каталог db уже в sys.path
    from postgres import DataConnection

logger = logging.getLogger(__name__)


class DialogConnection(DataConnection):
    """Хранилище диалогов пользователя с LLM.

    Каждый диалог привязан к user_id (UUID из таблицы accounts),
    каждое сообщение хранит время создания (created_at) и список
    документов, использованных LLM (поле documents, JSONB).
    """

    def create_dialog(self, user_id: str, title: str = "Новый диалог"):
        query = """
            INSERT INTO dialogs (user_id, title)
            VALUES (%s, %s)
            RETURNING id, user_id, title, created_at, updated_at;
        """
        try:
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (user_id, title))
                    return dict(cur.fetchone())
        except Exception as e:
            logger.error(f"Ошибка при создании диалога: {e}")
            raise

    def get_user_dialogs(self, user_id: str):
        query = """
            SELECT d.id, d.title, d.created_at, d.updated_at,
                   COUNT(m.id) AS messages_count
            FROM dialogs d
            LEFT JOIN messages m ON m.dialog_id = d.id
            WHERE d.user_id = %s
            GROUP BY d.id
            ORDER BY d.updated_at DESC;
        """
        try:
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (user_id,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка при получении списка диалогов: {e}")
            return []

    def _get_dialog_or_none(self, dialog_id: str, user_id: str):
        query = "SELECT id, title FROM dialogs WHERE id = %s AND user_id = %s;"
        try:
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (dialog_id, user_id))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Ошибка при получении диалога: {e}")
            return None

    def delete_dialog(self, dialog_id: str, user_id: str):
        query = "DELETE FROM dialogs WHERE id = %s AND user_id = %s;"
        try:
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(query, (dialog_id, user_id))
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Ошибка при удалении диалога: {e}")
            raise

    def rename_dialog(self, dialog_id: str, user_id: str, title: str):
        query = """
            UPDATE dialogs SET title = %s, updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING id, title;
        """
        try:
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (title, dialog_id, user_id))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Ошибка при переименовании диалога: {e}")
            raise

    def add_message(self, dialog_id: str, user_id: str, role: str,
                    content: str, documents=None):
        """Добавляет сообщение в диалог.

        documents — список документов, использованных LLM
        (например [{"id_cr": "...", "title": "..."}]), хранится как JSONB.
        """
        if self._get_dialog_or_none(dialog_id, user_id) is None:
            return None
        query = """
            INSERT INTO messages (dialog_id, role, content, documents)
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING id, dialog_id, role, content, documents, created_at;
        """
        update_query = "UPDATE dialogs SET updated_at = NOW() WHERE id = %s;"
        try:
            import json
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (dialog_id, role, content,
                                         json.dumps(documents or [], ensure_ascii=False)))
                    msg = dict(cur.fetchone())
                    cur.execute(update_query, (dialog_id,))
                    return msg
        except Exception as e:
            logger.error(f"Ошибка при добавлении сообщения: {e}")
            raise

    def get_dialog_messages(self, dialog_id: str, user_id: str):
        if self._get_dialog_or_none(dialog_id, user_id) is None:
            return None
        query = """
            SELECT id, role, content, documents, created_at
            FROM messages
            WHERE dialog_id = %s
            ORDER BY created_at ASC;
        """
        try:
            with self._get_connection() as conn:
                conn.autocommit = True
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (dialog_id,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка при получении сообщений диалога: {e}")
            return []
