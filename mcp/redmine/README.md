# mcp/redmine/README.md
# Redmine MCP Server

MCP Server para la API REST de Redmine, construido con **FastMCP 3.x** y empaquetado en Docker.  
Corre en modo **stdio** — Claude Desktop / Claude Code arrancan el contenedor y se comunican vía stdin/stdout.

## Herramientas expuestas

| Herramienta | Descripción |
|---|---|
| `get_issue` | Obtiene un issue por ID (con journals, adjuntos, watchers, etc.) |
| `list_issues` | Lista issues con filtros (proyecto, estado, asignado, tracker) |
| `create_issue` | Crea un nuevo issue |
| `update_issue` | Actualiza un issue (campos parciales + nota al journal) |
| `delete_issue` | Elimina un issue ⚠️ |
| `add_watcher` | Añade un watcher a un issue |
| `remove_watcher` | Elimina un watcher de un issue |
| `list_projects` | Lista todos los proyectos accesibles |
| `get_project` | Obtiene un proyecto por ID/slug |
| `create_project` | Crea un nuevo proyecto |
| `update_project` | Actualiza un proyecto |
| `delete_project` | Elimina un proyecto ⚠️ |

## Configuración

Variables de entorno necesarias (del `.env` raíz del proyecto):

```
REDMINE_URL=http://localhost:3000
REDMINE_API_KEY=tu_api_key_aqui
```

## Build y uso

### 1. Build de la imagen

```bash
# Desde la raíz del proyecto:
make mcp-build

# O directamente:
docker build -t redmine-mcp-server ./mcp/redmine
```

### 2. Test local rápido

```bash
# Build
docker build -t redmine-mcp-server ./mcp/redmine

# Smoke test — debe imprimir el JSON de inicialización MCP
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' \
  | docker run --rm -i \
    -e REDMINE_URL=http://host.docker.internal:3000 \
    -e REDMINE_API_KEY=tu_api_key \
    redmine-mcp-server
```

## Desarrollo sin Docker

```bash
# Instalar dependencias en el venv del proyecto
pip install fastmcp httpx python-dotenv

# Correr el servidor
cd mcp/redmine
python server.py
```
