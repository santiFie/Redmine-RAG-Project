---
name: obsidian-log-error
description: >-
  Registra un error resuelto o en proceso en las notas de Obsidian organizándolo 
  por categoría y utilizando la estructura estándar del proyecto. Usar cuando el usuario 
  pida registrar o documentar un error en Obsidian o mencione la frase u opción de log error.
---

# Skill: Registro de Errores en Obsidian

Cuando el usuario pida registrar o documentar un error en Obsidian:
1. Revisa la conversación y extrae la información del error resuelto.
2. Identifica la categoría de la carpeta (Langgraph, LlamaIndex, Python, CORS, etc.). En caso de que la categoría del error no corresponda a ninguna carpeta, consultale al usuario si desea generar una nueva carpeta con la categoría sugerida.
3. Genera la ruta: `[Categoría]/[nombre-corto-error].md`.
4. Utiliza la herramienta MCP de Obsidian para crear/actualizar la nota con la siguiente estructura:

# [Título corto del error]
Fecha: YYYY-MM-DD
Estado: resuelto/en proceso

## Error
```text
[Mensaje de error]
Contexto
Herramienta: [Herramienta]
```
Versión (herramienta): [Versión]

Python: [Versión]

Paquetes implicados: [Paquetes]

SO: [Sistema Operativo]

Qué estaba intentando hacer: [Descripción]

Causa
[Explicación de la causa raíz]

Solución
[Resumen de la solución]

Comandos / código
```Python
# Antes
```

```Python
# Después
```

Fuentes / enlaces
[Links o docs consultadas]

Notas
