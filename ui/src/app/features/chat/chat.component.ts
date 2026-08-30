// src/app/features/chat/chat.component.ts
import {
  Component, OnDestroy, OnInit, ViewChild, ElementRef,
  inject, signal, computed
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService } from '../../core/auth/auth.service';
import { ChatService, ChatEvent, ToolCallEvent, ToolResultEvent, NODE_LABELS } from '../../core/chat/chat.service';
import { MessageBubbleComponent, ChatMessage } from './components/message-bubble/message-bubble.component';
import { ToolIndicatorComponent, ToolStep } from './components/tool-indicator/tool-indicator.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, MessageBubbleComponent, ToolIndicatorComponent, RouterLink, RouterLinkActive],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
})
export class ChatComponent implements OnInit, OnDestroy {
  @ViewChild('messagesArea') private messagesArea!: ElementRef<HTMLDivElement>;
  @ViewChild('inputEl') private inputEl!: ElementRef<HTMLTextAreaElement>;

  protected readonly auth = inject(AuthService);
  private readonly chat = inject(ChatService);
  private readonly router = inject(Router);
  private streamSub?: Subscription;

  protected readonly messages = signal<ChatMessage[]>([]);
  protected readonly threadId = signal<string | null>(null);
  protected readonly isStreaming = signal(false);
  protected readonly currentNode = signal<string | null>(null);
  protected readonly currentStreamingMsgId = signal<string | null>(null);
  protected readonly errorMsg = signal<string | null>(null);
  protected inputText = '';

  /** Mapa msgId → tool steps para ese mensaje */
  private toolStepsMap = new Map<string, ToolStep[]>();
  /** Tool call pendiente (esperando su resultado) */
  private pendingToolCall?: ToolCallEvent & { startTime: number };

  readonly suggestions = [
    'Mostrame los issues abiertos de alta prioridad',
    '¿Cómo configuro la VPN para nuevos empleados?',
    'Creá un issue de prueba en Redmine',
    '¿Cuál es el estado del issue #5?',
  ];

  protected readonly userInitial = computed(() =>
    (this.auth.currentUser()?.displayName ?? 'U')[0].toUpperCase()
  );

  protected readonly canSend = computed(() =>
    !!this.inputText.trim() && !this.isStreaming()
  );

  protected readonly nodeLabel = computed(() =>
    this.currentNode() ? (NODE_LABELS[this.currentNode()!] ?? this.currentNode()) : ''
  );

  ngOnInit(): void {
    if (!this.auth.isAuthenticated()) {
      this.router.navigate(['/login']);
    }
  }

  ngOnDestroy(): void {
    this.streamSub?.unsubscribe();
  }

  protected getToolSteps(msgId: string): ToolStep[] {
    return this.toolStepsMap.get(msgId) ?? [];
  }

  protected onEnter(event: Event): void {
    const ke = event as KeyboardEvent;
    if (!ke.shiftKey) {
      ke.preventDefault();
      this.sendMessage();
    }
  }

  protected autoResize(): void {
    const el = this.inputEl?.nativeElement;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    }
  }

  protected sendSuggestion(text: string): void {
    this.inputText = text;
    this.sendMessage();
  }

  protected newConversation(): void {
    this.streamSub?.unsubscribe();
    this.messages.set([]);
    this.threadId.set(null);
    this.toolStepsMap.clear();
    this.isStreaming.set(false);
    this.currentNode.set(null);
    this.errorMsg.set(null);
  }

  protected sendMessage(): void {
    const text = this.inputText.trim();
    if (!text || this.isStreaming()) return;

    // Agregar mensaje del usuario
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    this.messages.update(msgs => [...msgs, userMsg]);
    this.inputText = '';
    this.autoResize();

    // Preparar mensaje del asistente (se va a llenar con streaming)
    const assistantMsgId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: new Date(),
    };
    this.messages.update(msgs => [...msgs, assistantMsg]);
    this.toolStepsMap.set(assistantMsgId, []);

    this.isStreaming.set(true);
    this.currentStreamingMsgId.set(assistantMsgId);
    this.scrollToBottom();

    this.streamSub = this.chat.streamMessage(text, this.threadId()).subscribe({
      next: (event: ChatEvent) => this.handleEvent(event, assistantMsgId),
      error: (err) => {
        this.errorMsg.set(`Error de conexión: ${err.message}`);
        this.finalizeStreaming(assistantMsgId);
      },
      complete: () => this.finalizeStreaming(assistantMsgId),
    });
  }

  private handleEvent(event: ChatEvent, msgId: string): void {
    switch (event.type) {
      case 'token':
        this.messages.update(msgs =>
          msgs.map(m => m.id === msgId ? { ...m, content: m.content + event.text } : m)
        );
        this.scrollToBottom();
        break;

      case 'node_start':
        this.currentNode.set(event.node);
        this.messages.update(msgs =>
          msgs.map(m => m.id === msgId ? { ...m, currentNode: event.node } : m)
        );
        break;

      case 'tool_call': {
        this.pendingToolCall = { ...event, startTime: Date.now() };
        const step: ToolStep = { toolCall: event, startTime: Date.now() };
        this.toolStepsMap.get(msgId)?.push(step);
        // Forzar re-detección en Angular
        this.toolStepsMap = new Map(this.toolStepsMap);
        break;
      }

      case 'tool_result': {
        const steps = this.toolStepsMap.get(msgId) ?? [];
        const idx = steps.findIndex(s => s.toolCall.id === event.tool_call_id);
        if (idx >= 0) {
          steps[idx] = {
            ...steps[idx],
            result: event,
            durationMs: Date.now() - steps[idx].startTime,
          };
          this.toolStepsMap = new Map(this.toolStepsMap);
        }
        break;
      }

      case 'done':
        if (!this.threadId()) this.threadId.set(event.thread_id);
        this.finalizeStreaming(msgId, event.run_id);
        break;

      case 'error':
        this.errorMsg.set(event.message);
        this.finalizeStreaming(msgId);
        break;
    }
  }

  private finalizeStreaming(msgId: string, runId?: string): void {
    this.messages.update(msgs =>
      msgs.map(m => m.id === msgId ? { ...m, isStreaming: false, currentNode: undefined, runId: runId ?? m.runId } : m)
    );
    this.isStreaming.set(false);
    this.currentStreamingMsgId.set(null);
    this.currentNode.set(null);
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    requestAnimationFrame(() => {
      const el = this.messagesArea?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  protected handleFeedback(event: { runId: string, score: number, comment?: string, value?: string }): void {
    this.chat.submitFeedback(event.runId, event.score, event.comment, event.value).subscribe({
      error: (err) => {
        this.errorMsg.set(`Error al enviar feedback: ${err.message}`);
      }
    });
  }
}
