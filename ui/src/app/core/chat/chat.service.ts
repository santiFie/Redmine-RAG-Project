// src/app/core/chat/chat.service.ts
// ====================================
// Servicio que consume el endpoint SSE del BFF y parsea los eventos
// tipados del agente LangGraph en tiempo real.

import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { AuthService } from '../auth/auth.service';
import { environment } from '../../../environments/environment';

// ── Tipos de eventos SSE ──────────────────────────────────────────────────

/** Fragmento de texto del LLM */
export interface TokenEvent {
  type: 'token';
  text: string;
}

/** El agente invocó una herramienta */
export interface ToolCallEvent {
  type: 'tool_call';
  name: string;
  args: Record<string, unknown>;
  id: string;
  node?: string;
}

/** Resultado de ejecución de una herramienta */
export interface ToolResultEvent {
  type: 'tool_result';
  tool_call_id: string;
  name: string;
  content: string;
}

/** Un nodo del grafo comenzó su ejecución */
export interface NodeStartEvent {
  type: 'node_start';
  node: string;
}

/** Error durante la ejecución del agente */
export interface ErrorEvent {
  type: 'error';
  message: string;
}

/** El stream finalizó */
export interface DoneEvent {
  type: 'done';
  thread_id: string;
}

export type ChatEvent =
  | TokenEvent
  | ToolCallEvent
  | ToolResultEvent
  | NodeStartEvent
  | ErrorEvent
  | DoneEvent;

// ── Nombres legibles de nodos del grafo ──────────────────────────────────

export const NODE_LABELS: Record<string, string> = {
  analyze_safe_query: 'Verificando seguridad de la consulta...',
  analyze_intent:     'Clasificando intención...',
  redmine_agent:      'Consultando Redmine...',
  rag_query:          'Buscando en base de conocimiento...',
  respond:            'Generando respuesta RAG...',
  respond_general:    'Procesando consulta general...',
  output_guardrail:   'Validando respuesta...',
};

// ── Servicio ──────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly auth = inject(AuthService);

  /**
   * Abre un stream SSE hacia el BFF y retorna un Observable de eventos tipados.
   *
   * Usa `fetch` + `ReadableStream` en lugar de `EventSource` para poder
   * enviar el JWT en el header Authorization (EventSource no soporta headers).
   *
   * @param message   Mensaje del usuario a enviar al agente.
   * @param threadId  ID del thread existente. Null para crear uno nuevo.
   */
  streamMessage(message: string, threadId: string | null): Observable<ChatEvent> {
    return new Observable<ChatEvent>((subscriber) => {
      const controller = new AbortController();
      const token = this.auth.getToken();

      fetch(`${environment.apiUrl}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ message, thread_id: threadId }),
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            const error = await response.text();
            subscriber.error(new Error(`BFF error ${response.status}: ${error}`));
            return;
          }
          if (!response.body) {
            subscriber.error(new Error('No response body'));
            return;
          }

          const reader  = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer    = '';

          // Leer el stream SSE línea por línea
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Procesar bloques de eventos SSE (separados por \n\n)
            const blocks = buffer.split('\n\n');
            buffer = blocks.pop() ?? '';

            for (const block of blocks) {
              const event = this._parseSSEBlock(block);
              if (event) subscriber.next(event);
            }
          }

          subscriber.complete();
        })
        .catch((err) => {
          if (err.name !== 'AbortError') {
            subscriber.error(err);
          }
        });

      // Teardown: cancelar el fetch al desuscribirse
      return () => controller.abort();
    });
  }

  /**
   * Crea un nuevo thread de conversación en LangGraph vía el BFF.
   */
  createThread(): Observable<{ thread_id: string }> {
    return new Observable((subscriber) => {
      const token = this.auth.getToken();

      fetch(`${environment.apiUrl}/chat/threads`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      })
        .then((res) => res.json())
        .then((data) => {
          subscriber.next(data);
          subscriber.complete();
        })
        .catch((err) => subscriber.error(err));
    });
  }

  // ── Parseo SSE ────────────────────────────────────────────────────────────

  /** Parsea un bloque SSE (event: xxx\ndata: {...}) a un ChatEvent tipado. */
  private _parseSSEBlock(block: string): ChatEvent | null {
    let eventType = '';
    let dataLine  = '';

    for (const line of block.split('\n')) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        dataLine = line.slice(6).trim();
      }
    }

    if (!eventType || !dataLine) return null;

    try {
      const data = JSON.parse(dataLine);

      switch (eventType) {
        case 'token':
          return { type: 'token', text: data.text ?? '' };
        case 'tool_call':
          return { type: 'tool_call', name: data.name, args: data.args ?? {}, id: data.id ?? '', node: data.node };
        case 'tool_result':
          return { type: 'tool_result', tool_call_id: data.tool_call_id, name: data.name, content: data.content };
        case 'node_start':
          return { type: 'node_start', node: data.node };
        case 'error':
          return { type: 'error', message: data.message };
        case 'done':
          return { type: 'done', thread_id: data.thread_id };
        default:
          return null;
      }
    } catch {
      return null;
    }
  }
}
