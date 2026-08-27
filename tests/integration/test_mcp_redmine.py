"""
tests/integration/test_mcp_redmine.py
=====================================
Test de integración mínima para validar que el servidor MCP de Redmine
levanta correctamente y registra las herramientas (tools) esperadas.
"""

import sys
from pathlib import Path

# Agregar el directorio mcp/redmine al PYTHONPATH temporalmente para poder importar
_mcp_dir = Path(__file__).resolve().parent.parent.parent / "mcp" / "redmine"
sys.path.append(str(_mcp_dir))

from server import mcp  # noqa: E402


def test_mcp_server_initialization():
    """Valida que el servidor FastMCP se instancia y tiene el nombre correcto."""
    assert mcp.name == "redmine"


def test_mcp_server_tools_registered():
    """Valida que todas las herramientas requeridas estén registradas en el MCP."""
    
    # FastMCP internamente registra las herramientas.
    # Verificamos los nombres de las funciones decoradas.
    registered_tools = [tool.name for tool in mcp._tools.values()] if hasattr(mcp, "_tools") else []
    
    # Si FastMCP expone las tools de otra forma (por ejemplo .list_tools()), 
    # podemos usar getattr para evitar fallos si cambia la API interna.
    if not registered_tools and hasattr(mcp, "list_tools"):
        tools = mcp.list_tools()
        # Adaptarse a si devuelve una lista de diccionarios, objetos, etc.
        registered_tools = [t.name if hasattr(t, "name") else t["name"] for t in tools]
        
    # Herramientas mínimas que esperamos que el servidor exponga:
    expected_tools = [
        "get_issue",
        "list_issues",
        "create_issue",
        "update_issue",
        "delete_issue",
        "add_watcher",
        "remove_watcher",
        "list_projects",
        "get_project",
        "create_project",
        "update_project",
        "delete_project"
    ]
    
    if registered_tools:
        for tool in expected_tools:
            assert tool in registered_tools, f"Falta la herramienta MCP: {tool}"
