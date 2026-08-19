// src/app/features/chat/components/tool-indicator/tool-indicator.component.ts
// =============================================================================
// Componente animado que muestra el progreso de ejecución de una herramienta.
// Se puede expandir para ver los argumentos y el resultado.

import { Component, Input, OnInit, signal } from '@angular/core';
import { ToolCallEvent, ToolResultEvent, NODE_LABELS } from '../../../../core/chat/chat.service';

/** Íconos SVG por nombre de herramienta/nodo */
const TOOL_ICONS: Record<string, string> = {
  redmine_agent:     '🔧',
  rag_query:         '📚',
  analyze_intent:    '🧠',
  analyze_safe_query:'🛡️',
  respond:           '✍️',
  respond_general:   '💬',
  output_guardrail:  '🔍',
  default:           '⚙️',
};

export interface ToolStep {
  toolCall: ToolCallEvent;
  result?: ToolResultEvent;
  durationMs?: number;
  startTime: number;
}

@Component({
  selector: 'app-tool-indicator',
  standalone: true,
  template: `
    <div class="tool-indicator" [class.expanded]="isExpanded()" [class.done]="step.result">
      <!-- Cabecera clickeable -->
      <button
        class="tool-header"
        (click)="isExpanded.set(!isExpanded())"
        [attr.aria-expanded]="isExpanded()"
        type="button"
      >
        <!-- Ícono de estado -->
        <span class="tool-status-icon" [class.spinning]="!step.result">
          @if (step.result) {
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          } @else {
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="animate-spin">
              <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
          }
        </span>

        <!-- Nombre y estado -->
        <div class="tool-info">
          <span class="tool-emoji">{{ getIcon() }}</span>
          <span class="tool-name">{{ getLabel() }}</span>
          @if (step.result) {
            <span class="tool-duration">{{ getDuration() }}</span>
          } @else {
            <span class="tool-running">ejecutando…</span>
          }
        </div>

        <!-- Chevron -->
        <svg class="tool-chevron" width="14" height="14" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>

      <!-- Detalles expandibles -->
      @if (isExpanded()) {
        <div class="tool-details animate-fade-slide-up">
          <!-- Args de la herramienta -->
          @if (hasArgs()) {
            <div class="detail-section">
              <span class="detail-label">Argumentos</span>
              <pre class="detail-code">{{ formatArgs() }}</pre>
            </div>
          }

          <!-- Resultado -->
          @if (step.result) {
            <div class="detail-section">
              <span class="detail-label">Resultado</span>
              <pre class="detail-code result-code">{{ step.result.content }}</pre>
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    .tool-indicator {
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      background: var(--color-bg-tertiary);
      overflow: hidden;
      transition: border-color var(--transition-fast);
      font-size: var(--font-size-sm);

      &.done {
        border-color: rgba(16, 185, 129, 0.25);
        background: rgba(16, 185, 129, 0.04);
      }

      &:not(.done) {
        border-color: rgba(99, 102, 241, 0.3);
        animation: glow-pulse 2s ease-in-out infinite;
      }
    }

    .tool-header {
      width: 100%;
      display: flex;
      align-items: center;
      gap: var(--space-2);
      padding: var(--space-2) var(--space-3);
      background: none;
      border: none;
      cursor: pointer;
      text-align: left;
      color: var(--color-text-primary);
    }

    .tool-status-icon {
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      flex-shrink: 0;
      color: var(--color-accent-success);

      .tool-indicator:not(.done) & {
        color: var(--color-accent-primary);
      }
    }

    .tool-info {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      flex: 1;
      min-width: 0;
    }

    .tool-emoji {
      font-size: 14px;
      flex-shrink: 0;
    }

    .tool-name {
      font-weight: 500;
      color: var(--color-text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .tool-duration {
      font-size: var(--font-size-xs);
      color: var(--color-accent-success);
      font-family: var(--font-mono);
      flex-shrink: 0;
    }

    .tool-running {
      font-size: var(--font-size-xs);
      color: var(--color-accent-primary);
      animation: pulse 1.5s ease-in-out infinite;
    }

    .tool-chevron {
      flex-shrink: 0;
      color: var(--color-text-muted);
      transition: transform var(--transition-fast);

      .expanded & {
        transform: rotate(180deg);
      }
    }

    /* Detalles */
    .tool-details {
      border-top: 1px solid var(--color-border-subtle);
      padding: var(--space-3);
      display: flex;
      flex-direction: column;
      gap: var(--space-3);
    }

    .detail-section {
      display: flex;
      flex-direction: column;
      gap: var(--space-1);
    }

    .detail-label {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      color: var(--color-text-muted);
    }

    .detail-code {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--color-text-secondary);
      background: var(--color-bg-primary);
      border: 1px solid var(--color-border-subtle);
      border-radius: var(--radius-sm);
      padding: var(--space-2) var(--space-3);
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 120px;
      overflow-y: auto;
      line-height: 1.5;
    }

    .result-code {
      color: var(--color-accent-success);
    }
  `],
})
export class ToolIndicatorComponent {
  @Input({ required: true }) step!: ToolStep;

  protected readonly isExpanded = signal(false);

  protected getIcon(): string {
    const node = this.step.toolCall.node ?? this.step.toolCall.name;
    return TOOL_ICONS[node] ?? TOOL_ICONS['default'];
  }

  protected getLabel(): string {
    const node = this.step.toolCall.node ?? '';
    if (NODE_LABELS[node]) return NODE_LABELS[node].replace('...', '');
    return this.step.toolCall.name || 'Herramienta desconocida';
  }

  protected getDuration(): string {
    if (!this.step.durationMs) return '';
    return this.step.durationMs < 1000
      ? `${this.step.durationMs}ms`
      : `${(this.step.durationMs / 1000).toFixed(1)}s`;
  }

  protected hasArgs(): boolean {
    return Object.keys(this.step.toolCall.args ?? {}).length > 0;
  }

  protected formatArgs(): string {
    return JSON.stringify(this.step.toolCall.args, null, 2);
  }
}
