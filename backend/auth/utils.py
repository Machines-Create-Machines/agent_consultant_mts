# auth/utils.py
import os
from datetime import datetime
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel

# Секрет берётся из переменных окружения (.env), а не хранится в репозитории.
SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError(
        "Переменная окружения JWT_SECRET не задана. "
        "Скопируйте .env.example в .env и укажите свой секрет."
    )
ALGORITHM = "HS256"

security = HTTPBearer()


class TokenData(BaseModel):
    user_id: str
    login: Optional[str] = None
    role: str


def decode_jwt(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        login: str = payload.get("login")
        role: str = payload.get("role")
        if user_id is None or login is None or role is None:
            return None
        return TokenData(user_id=user_id, login=login, role=role)
    except jwt.PyJWTError:
        return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    token = credentials.credentials
    token_data = decode_jwt(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return token_data
