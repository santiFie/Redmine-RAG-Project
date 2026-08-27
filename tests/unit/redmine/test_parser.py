"""
tests/unit/redmine/test_parser.py
=================================
Pruebas unitarias para la lógica de parseo de JSON a TextNodes de LlamaIndex.
"""

from src.redmine.parser import parse_redmine_issue_to_nodes


def test_parse_redmine_issue_to_nodes_completo():
    issue_data = {
        "id": 42,
        "subject": "Falla de BD",
        "description": "La base de datos se cayó.",
        "project": {"id": 1, "name": "Infraestructura"},
        "status": {"id": 2, "name": "En progreso"},
        "author": {"id": 3, "name": "Juan Perez"},
        "journals": [
            {
                "id": 100,
                "user": {"name": "Maria DB"},
                "notes": "Reiniciando servidor",
                "created_on": "2023-10-01T10:00:00Z",
            }
        ],
    }

    nodes = parse_redmine_issue_to_nodes(issue_data)

    # Debe haber 1 nodo principal y 1 nodo de comentario
    assert len(nodes) == 2

    main_node = nodes[0]
    assert main_node.id_ == "redmine_issue_42_main"
    assert "Falla de BD" in main_node.text
    assert main_node.metadata["issue_id"] == 42
    assert main_node.metadata["project"] == "Infraestructura"
    assert main_node.metadata["entity_type"] == "issue_description"

    comment_node = nodes[1]
    assert comment_node.id_ == "redmine_issue_42_journal_100"
    assert "Reiniciando servidor" in comment_node.text
    assert comment_node.metadata["entity_type"] == "issue_comment"
    assert comment_node.metadata["comment_author"] == "Maria DB"


def test_parse_redmine_issue_sin_journals():
    issue_data = {
        "id": 42,
        "subject": "Solo ticket",
    }

    nodes = parse_redmine_issue_to_nodes(issue_data)
    assert len(nodes) == 1
    assert nodes[0].id_ == "redmine_issue_42_main"


def test_parse_redmine_issue_journal_vacio():
    issue_data = {
        "id": 42,
        "subject": "Ticket con journal vacio",
        "journals": [
            {
                "id": 100,
                "user": {"name": "Maria DB"},
                "notes": "",  # Sin texto
            },
            {
                "id": 101,
                "user": {"name": "Maria DB"},
                # notes no provisto
            },
        ],
    }

    nodes = parse_redmine_issue_to_nodes(issue_data)
    # Los journals vacíos no deben generar nodos
    assert len(nodes) == 1
