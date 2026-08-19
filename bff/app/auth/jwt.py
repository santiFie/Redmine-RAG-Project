"""
bff/app/auth/jwt.py
===================
Utilidades para generación y verificación de JWT.
Usa python-jose con algoritmo HS256 (configurable vía settings).
"""

from datetime import UTC, datetime, timedelta

from jose import jwt

from app.config import settings

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------
_CREDENTIALS_EXCEPTION_MSG = "No se pudieron validar las credenciales"

# Usuario de demostración fijo (no requiere base de datos en el BFF)
_DEMO_USER = {
    "username": "admin",
    # La contraseña se compara en texto plano solo para demo.
    # En producción, usar hash bcrypt o un IdP externo.
    "password": "admin1234",
    "display_name": "Administrador",
    "role": "admin",
}


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Valida las credenciales contra el usuario demo hardcodeado.

    Returns:
        Diccionario con los datos del usuario si las credenciales son correctas.
        None en caso contrario.
    """
    if username == _DEMO_USER["username"] and password == _DEMO_USER["password"]:
        return {k: v for k, v in _DEMO_USER.items() if k != "password"}
    return None


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Genera un JWT firmado con los datos proporcionados.

    Args:
        data: Payload del token (se incluye 'sub' como identificador del usuario).
        expires_delta: Duración del token. Usa ACCESS_TOKEN_EXPIRE_MINUTES por defecto.

    Returns:
        Token JWT como string.
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict:
    """
    Decodifica y verifica un JWT.

    Args:
        token: JWT a verificar.

    Returns:
        Payload decodificado como diccionario.

    Raises:
        JWTError: Si el token es inválido, expirado o mal formado.
    """
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


__all__ = ["authenticate_user", "create_access_token", "decode_access_token"]
