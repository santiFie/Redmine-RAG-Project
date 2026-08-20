// src/app/features/chat/components/message-bubble/message-bubble.component.ts
// =============================================================================
// Burbuja de mensaje del chat. Renderiza Markdown con `marked` y muestra un
// indicador minimalista del nodo actual durante el streaming.

import { Component, Input, SecurityContext } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';
import { NODE_LABELS } from '../../../../core/chat/chat.service';

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  /** Contenido textual acumulado. Puede crecer durante el streaming. */
  content: string;
  /** True mientras se está recibiendo el streaming de este mensaje */
  isStreaming?: boolean;
  /** Nodo del grafo actualmente en ejecución (solo durante streaming) */
  currentNode?: string;
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

  constructor(private sanitizer: DomSanitizer) {}

  /** Label legible del nodo actual, o string vacío si no hay nodo */
  get nodeLabel(): string {
    if (!this.message.currentNode) return '';
    return NODE_LABELS[this.message.currentNode] ?? this.message.currentNode;
  }

  /** Renderiza el contenido: indicador de nodo durante streaming, Markdown al finalizar */
  get renderedContent(): SafeHtml {
    if (this.message.isStreaming && this.message.currentNode) {
      const escaped = this.nodeLabel
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      return this.sanitizer.bypassSecurityTrustHtml(
        `<span class="node-status">${escaped}</span>`
      );
    }

    if (!this.message.content) {
      return '';
    }

    const html = marked.parse(this.message.content, { async: false }) as string;
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  protected formatTime(date: Date): string {
    return date.toLocaleTimeString('es-AR', { hour: '2-digit', minute: '2-digit' });
  }
}
