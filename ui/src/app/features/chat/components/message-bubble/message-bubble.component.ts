// src/app/features/chat/components/message-bubble/message-bubble.component.ts
// =============================================================================
// Burbuja de mensaje del chat. Renderiza Markdown con `marked` y muestra un
// indicador minimalista del nodo actual durante el streaming.

import { Component, Input, Output, EventEmitter, SecurityContext } from '@angular/core';
import { FormsModule } from '@angular/forms';
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
  runId?: string;
  timestamp: Date;
  /** Indica si ocurrió un error durante el procesamiento */
  hasError?: boolean;
  /** Mensaje amigable descriptivo del error */
  errorMessage?: string;
}

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './message-bubble.component.html',
  styleUrl: './message-bubble.component.scss',
})
export class MessageBubbleComponent {
  @Input({ required: true }) message!: ChatMessage;
  @Output() feedback = new EventEmitter<{ runId: string, score: number, comment?: string, value?: string }>();
  @Output() retry = new EventEmitter<void>();

  showFeedbackForm = false;
  feedbackCategory = '';
  feedbackComment = '';
  feedbackSubmitted = false;

  constructor(private sanitizer: DomSanitizer) { }

  onThumbsUp() {
    if (!this.message.runId || this.feedbackSubmitted) return;
    this.feedback.emit({ runId: this.message.runId, score: 1 });
    this.feedbackSubmitted = true;
  }

  onThumbsDown() {
    if (!this.message.runId || this.feedbackSubmitted) return;
    this.showFeedbackForm = true;
  }

  submitNegativeFeedback() {
    if (!this.message.runId) return;
    this.feedback.emit({
      runId: this.message.runId,
      score: 0,
      comment: this.feedbackComment,
      value: this.feedbackCategory || undefined
    });
    this.showFeedbackForm = false;
    this.feedbackSubmitted = true;
  }

  cancelFeedback() {
    this.showFeedbackForm = false;
  }

  /** Label legible del nodo actual, o string vacío si no hay nodo */
  get nodeLabel(): string {
    if (!this.message.currentNode) return '';
    return NODE_LABELS[this.message.currentNode] ?? this.message.currentNode;
  }

  /** Renderiza el contenido: indicador de nodo si aún no hay tokens, Markdown si hay contenido */
  get renderedContent(): SafeHtml {
    if (!this.message.content) {
      if (this.message.isStreaming && this.message.currentNode) {
        const escaped = this.nodeLabel
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
        return this.sanitizer.bypassSecurityTrustHtml(
          `<span class="node-status">${escaped}</span>`
        );
      }
      return '';
    }

    const html = marked.parse(this.message.content, { async: false }) as string;
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  protected formatTime(date: Date): string {
    return date.toLocaleTimeString('es-AR', { hour: '2-digit', minute: '2-digit' });
  }
}
