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
  interrupted?: boolean;
}

@Component({
  selector: 'app-tool-indicator',
  standalone: true,
  templateUrl: './tool-indicator.component.html',
  styleUrl: './tool-indicator.component.scss',
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
