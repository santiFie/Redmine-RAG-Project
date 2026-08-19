"""
bff/app/config.py
=================
Configuración centralizada del BFF mediante Pydantic Settings.
Lee las variables de entorno del archivo .env (o del entorno del sistema).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del BFF cargada desde variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Servidor ---
    port: int = 8000

    # --- JWT ---
    secret_key: str = "cambia_esta_clave_secreta_jwt_en_produccion_min32chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- LangGraph Platform ---
    langgraph_api_url: str = "http://localhost:8123"
    langgraph_graph_id: str = "redmine_agent"
    langgraph_api_key: str = ""

    # --- CORS ---
    allowed_origins: str = "http://localhost:4200,http://localhost"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Retorna los orígenes CORS como lista."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


# Instancia global — importar desde aquí en todos los módulos
settings = Settings()
