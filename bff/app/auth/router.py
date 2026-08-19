"""
bff/app/auth/router.py
======================
Router de autenticación.

Endpoints:
  POST /auth/login   — Valida credenciales y retorna JWT.
  GET  /auth/me      — Retorna los datos del usuario autenticado.
  POST /auth/logout  — Logout simbólico (el cliente descarta el token).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth.jwt import authenticate_user, create_access_token
from app.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Autenticación"])


# ---------------------------------------------------------------------------
# Schemas de request / response
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Credenciales de login."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Respuesta con el JWT generado."""
    access_token: str
    token_type: str = "bearer"
    display_name: str
    role: str


class UserResponse(BaseModel):
    """Datos del usuario autenticado."""
    username: str
    display_name: str
    role: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Autenticar usuario y obtener JWT",
    responses={
        401: {"description": "Credenciales incorrectas"},
    },
)
async def login(body: LoginRequest) -> TokenResponse:
    """
    Valida las credenciales del usuario y retorna un JWT de acceso.
    Las credenciales de demo son admin / admin1234.
    """
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={
            "sub": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        }
    )
    return TokenResponse(
        access_token=token,
        display_name=user["display_name"],
        role=user["role"],
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Obtener datos del usuario autenticado",
)
async def me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    """Retorna los datos del usuario codificados en el JWT."""
    return UserResponse(
        username=current_user.get("sub", ""),
        display_name=current_user.get("display_name", ""),
        role=current_user.get("role", ""),
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout (el cliente descarta el token)",
)
async def logout(_: dict = Depends(get_current_user)) -> None:
    """
    Logout simbólico. Los JWT son stateless; el cliente es responsable
    de eliminar el token de su almacenamiento local.
    """
    return None
