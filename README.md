```markdown 
Фронтенд написан на Vue.js, бэкенд — на FastAPI, база данных — PostgreSQL. 
В приложении реализована авторизация, личный кабинет пользователя, а также интерактивный чат (диалоги) для общения с ИИ-ассистентом.
```

## Запуск через Docker (рекомендуемый способ)

1. Скопируйте шаблон и заполните секреты:

   ```bash
   cp .env.example .env
   # задайте DB_PASSWORD и JWT_SECRET (например: openssl rand -hex 32)
   ```

2. Соберите и запустите всё одной командой:

   ```bash
   docker compose up --build
   ```

   Поднимутся сервисы:
   - `db` — PostgreSQL (таблицы создаются автоматически из `db/init.sql`);
   - `init-db` — разовая инициализация данных;
   - `backend` — FastAPI, http://localhost:8002 (Swagger: http://localhost:8002/docs);
   - `frontend` — Vue-приложение за nginx, **http://localhost:8080** (запросы к API проксируются через `/api`).

3. Остановка: `docker compose down` (данные БД сохраняются в volume `pgdata`; полная очистка с удалением данных — `docker compose down -v`).

## Как пользоваться приложением

1. Откройте **http://localhost:8080**.
2. Зарегистрируйте нового пользователя (по умолчанию выдаётся роль `client`) или войдите в существующий аккаунт.
3. В разделе **«Диалоги»** вы можете общаться с ИИ. Пустые диалоги не создаются: новый диалог появляется в базе только после отправки первого сообщения.
4. В разделе **«Мой профиль»** можно просмотреть свои данные (логин, ФИО, дата регистрации) и отредактировать имя и фамилию.

## Локальный запуск без Docker (для разработки)

Для разработки без Docker необходимо отдельно запустить БД, бэкенд и фронтенд.

```bash
cp .env.example .env   # при локальном запуске укажите DB_HOST=localhost
pip install -r requirements.txt
npm install
python db/init_db.py   # инициализация БД
python run.py          # backend (http/https, в зависимости от наличия certs/)
npm run serve          # frontend на http://localhost:8080
```

Для HTTPS-режима вне Docker сгенерируйте самоподписанные сертификаты локально (например, `mkcert -key-file certs/key.pem -cert-file certs/cert.pem localhost 127.0.0.1 ::1`) — папка `certs/` игнорируется git. 
В режиме разработки фронтенд (`npm run serve`) проксирует запросы к бэкенду через `vue.config.js`.

## Технологии

- **Frontend:** Vue 3, Vue CLI, ESLint, CSS.
- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic, PyJWT.
- **Database:** PostgreSQL.
- **Инфраструктура:** Docker, Docker Compose, Nginx (для раздачи статики и проксирования).