// src/app/features/chat/components/message-bubble/message-bubble.component.ts
// =============================================================================
// Burbuja de mensaje del chat. Soporta Markdown básico y el cursor de typing.

import { Component, Input } from '@angular/core';

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  /** Contenido textual acumulado. Puede crecer durante el streaming. */
  content: string;
  /** True mientras se está recibiendo el streaming de este mensaje */
  isStreaming?: boolean;
  timestamp: Date;
}

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  template: `
    <div class="message-bubble" [class.user]="message.role === 'user'" [class.assistant]="message.role === 'assistant'">
      <!-- Avatar -->
      <div class="avatar" [class.avatar--user]="message.role === 'user'">
        @if (message.role === 'assistant') {
          <svg width="18" height="18" viewBox="0 0 32 32" fill="none">
            <path d="M16 2L30 9V23L16 30L2 23V9L16 2Z" fill="url(#g2)" />
            <path d="M10 16L14 20L22 12" stroke="white" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <defs>
              <linearGradient id="g2" x1="2" y1="2" x2="30" y2="30">
                <stop offset="0%" stop-color="#6366F1"/>
                <stop offset="100%" stop-color="#06B6D4"/>
              </linearGradient>
            </defs>
          </svg>
        } @else {
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
        }
      </div>

      <!-- Contenido -->
      <div class="bubble-body">
        <div class="bubble-content" [class.markdown-content]="message.role === 'assistant'">
          <!-- Renderizado de texto con soporte a saltos de línea -->
          <span [innerHTML]="renderedContent"></span>
          <!-- Cursor de typing -->
          @if (message.isStreaming) {
            <span class="typing-cursor" aria-hidden="true"></span>
          }
        </div>
        <div class="bubble-meta">
          {{ formatTime(message.timestamp) }}
        </div>
      </div>
    </div>
  `,
  styles: [`
    .message-bubble {
      display: flex;
      gap: var(--space-3);
      align-items: flex-start;
      animation: fadeSlideUp 0.25s ease both;
      max-width: 100%;

      &.user {
        flex-direction: row-reverse;

        .bubble-body { align-items: flex-end; }
        .bubble-content {
          background: var(--color-msg-user-bg);
          border: 1px solid var(--color-msg-user-border);
          border-radius: var(--radius-xl) var(--radius-xl) var(--radius-sm) var(--radius-xl);
          color: var(--color-text-primary);
        }

        .bubble-meta { text-align: right; }
      }

      &.assistant {
        .bubble-content {
          background: var(--color-msg-ai-bg);
          border: 1px solid var(--color-msg-ai-border);
          border-radius: var(--radius-xl) var(--radius-xl) var(--radius-xl) var(--radius-sm);
        }
      }
    }

    /* Avatar */
    .avatar {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-lg);
      background: var(--color-bg-elevated);
      border: 1px solid var(--color-border);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      color: var(--color-text-secondary);
    }

    .avatar--user {
      background: rgba(99,102,241,0.15);
      border-color: rgba(99,102,241,0.3);
      color: var(--color-accent-primary);
    }

    /* Cuerpo del mensaje */
    .bubble-body {
      display: flex;
      flex-direction: column;
      gap: var(--space-1);
      max-width: calc(100% - 52px);
      min-width: 0;
    }

    .bubble-content {
      padding: var(--space-3) var(--space-4);
      color: var(--color-text-primary);
      font-size: var(--font-size-md);
      line-height: 1.65;
      word-break: break-word;
      white-space: pre-wrap;
    }

    /* Cursor de typing */
    .typing-cursor {
      display: inline-block;
      width: 2px;
      height: 1em;
      background: var(--color-accent-primary);
      margin-left: 2px;
      vertical-align: text-bottom;
      animation: typingCursor 0.8s step-end infinite;
    }

    /* Meta */
    .bubble-meta {
      font-size: var(--font-size-xs);
      color: var(--color-text-muted);
      padding: 0 var(--space-1);
    }
  `],
})
export class MessageBubbleComponent {
  @Input({ required: true }) message!: ChatMessage;

  /** Convierte saltos de línea a <br> para el HTML */
  get renderedContent(): string {
    return this.message.content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }

  protected formatTime(date: Date): string {
    return date.toLocaleTimeString('es-AR', { hour: '2-digit', minute: '2-digit' });
  }
}
