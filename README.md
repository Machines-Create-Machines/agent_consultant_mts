```markdown 
Фронтенд написан на Vue.js, бэкенд — на FastAPI, база данных — PostgreSQL. 
В приложении реализована авторизация, личный кабинет пользователя, а также интерактивный чат (диалоги) для общения с ИИ-ассистентом EnergyTool через WebSocket.
```

## Запуск через Docker (рекомендуемый способ)

1. Скопируйте шаблон и заполните секреты:

   ```bash
   cp .env.example .env
   # задайте JWT_SECRET, DB_PASSWORD
   # и PLATFORM_API_KEY / PLATFORM_BASE_URL / PLATFORM_MODEL (или LLM_*)
   ```

2. Соберите KB (если ещё нет `build/chunks.jsonl`):

   ```bash
   make kb
   ```

3. Поднимите стек:

   ```bash
   make up
   # или: docker compose up --build
   ```

   Сервисы:
   - `db` — PostgreSQL;
   - `init-db` — разовая инициализация;
   - `backend` — FastAPI + agent KB, http://localhost:8002 (Swagger: http://localhost:8002/docs);
   - `frontend` — Vue за nginx, **http://localhost:8080** (`/api` и `/ws` проксируются на backend).

4. Логи / остановка:

   ```bash
   make logs
   make down
   # полная очистка volume: docker compose down -v
   ```

5. Smoke агента без UI:

   ```bash
   make ask Q="Какие тарифы на АСКУЭ?"
   ```

### Чат с агентом (WebSocket)

- UI на http://localhost:8080 открывает **`/ws/chat?token=<access_token>`** (через nginx → backend).
- Прямой backend: `ws://localhost:8002/ws/chat?token=<JWT>`.
- Протокол: после `ready` клиент шлёт `user_message` `{dialog_id, content}`; сервер отвечает кадрами `dialog` (если новый), `user_message`, `status` (`thinking`/`done`), `assistant_message`, при сбое — `error`. Keepalive: `ping`/`pong`.
- REST `POST /dialogs/.../messages` **не** вызывает агента (только запись в БД). История — `GET /dialogs/{id}/messages`.

## Как пользоваться приложением

1. Откройте **http://localhost:8080**.
2. Зарегистрируйте пользователя (роль `client`) или войдите.
3. В **«Диалоги»** общайтесь с ассистентом: новый диалог появляется после первого сообщения.
4. В **«Мой профиль»** — данные аккаунта и редактирование ФИО.

## Локальный запуск без Docker (для разработки)

```bash
cp .env.example .env   # DB_HOST=localhost; JWT_SECRET; PLATFORM_* / LLM_*
make kb                # если нет build/
pip install -r backend/requirements.txt
cd backend && python run.py          # http://localhost:8002
# в другом терминале:
cd frontend && npm install && npm run serve   # http://localhost:8080
```

Dev-прокси: `vue.config.js` должен проксировать `/api` и `/ws` на `:8002` (`ws: true` для WebSocket).

Для HTTPS вне Docker: сертификаты в `certs/` (папка в `.gitignore`).

## Технологии

- **Frontend:** Vue 3, Vue CLI, ESLint, CSS.
- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic, PyJWT, agent pipeline (EnergyTool KB).
- **Database:** PostgreSQL.
- **Инфраструктура:** Docker, Docker Compose, Nginx (статика + `/api` + `/ws`).
