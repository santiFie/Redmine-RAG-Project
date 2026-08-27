"""
bff/app/dependencies.py
=======================
Dependencias FastAPI reutilizables.
La dependencia `get_current_user` valida el JWT de cada request.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.auth.jwt import decode_access_token

# Esquema de seguridad: extrae el Bearer token del header Authorization
_bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """
    Dependencia FastAPI que extrae y valida el JWT del header Authorization.

    Returns:
        Payload del token como diccionario (contiene 'sub', 'display_name', 'role').

    Raises:
        HTTPException 401: Si el token es inválido o expirado.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(credentials.credentials)
        username: str | None = payload.get("sub")
        if not username:
            raise credentials_exception
        return payload
    except JWTError as exc:
        raise credentials_exception from exc
