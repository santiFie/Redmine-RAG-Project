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
  templateUrl: './message-bubble.component.html',
  styleUrl: './message-bubble.component.scss',
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
