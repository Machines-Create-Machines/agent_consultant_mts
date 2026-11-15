from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router_page import page_router
from api.router_dialog import dialog_router
from api.router_ws import ws_router
from auth.user import router

# Загружаем .env до импорта остальных модулей (в Docker переменные
# приходят через env_file/docker-compose, поэтому отсутствие .env не ошибка)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from services.agent_bridge import cold_start

        cold_start()
    except Exception:
        logger.exception(
            "Agent KB cold_start failed; /ws/chat will use fallback until KB is available"
        )
    yield


app = FastAPI(title="Medical Support", lifespan=lifespan)

origins = [
    "http://localhost:8080",
    "https://localhost:8080",
    "http://127.0.0.1:8080",
    "https://127.0.0.1:8080"
]

app.include_router(page_router)
app.include_router(dialog_router)
app.include_router(ws_router)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
