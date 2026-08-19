// src/app/features/chat/chat.component.ts
import {
  Component, OnDestroy, OnInit, ViewChild, ElementRef,
  inject, signal, computed
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService } from '../../core/auth/auth.service';
import { ChatService, ChatEvent, ToolCallEvent, ToolResultEvent, NODE_LABELS } from '../../core/chat/chat.service';
import { MessageBubbleComponent, ChatMessage } from './components/message-bubble/message-bubble.component';
import { ToolIndicatorComponent, ToolStep } from './components/tool-indicator/tool-indicator.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, MessageBubbleComponent, ToolIndicatorComponent],
  template: `
<div class="chat-layout">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="brand">
        <div class="brand-icon">
          <svg width="22" height="22" viewBox="0 0 32 32" fill="none">
            <path d="M16 2L30 9V23L16 30L2 23V9L16 2Z" fill="url(#g1)"/>
            <path d="M10 16L14 20L22 12" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <defs>
              <linearGradient id="g1" x1="2" y1="2" x2="30" y2="30">
                <stop offset="0%" stop-color="#6366F1"/>
                <stop offset="100%" stop-color="#06B6D4"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        <span class="brand-name gradient-text">AgentOS</span>
      </div>
    </div>

    <div class="sidebar-section">
      <button class="btn btn-primary new-chat-btn" (click)="newConversation()" id="new-chat-btn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
        Nueva conversación
      </button>
    </div>

    <div class="sidebar-section sidebar-history">
      <p class="sidebar-label">Conversación actual</p>
      @if (threadId()) {
        <div class="thread-item active">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span class="text-truncate">{{ messages()[0]?.content.slice(0, 30) || 'Sin mensajes aún' }}</span>
        </div>
      } @else {
        <p class="sidebar-empty">Enviá un mensaje para empezar.</p>
      }
    </div>

    <div class="sidebar-footer">
      <div class="user-info">
        <div class="user-avatar">{{ userInitial() }}</div>
        <div class="user-details">
          <span class="user-name">{{ auth.currentUser()?.displayName }}</span>
          <span class="user-role">{{ auth.currentUser()?.role }}</span>
        </div>
        <button class="btn btn-icon logout-btn" (click)="auth.logout()" title="Cerrar sesión" id="logout-btn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
            <polyline points="16 17 21 12 16 7"/>
            <line x1="21" y1="12" x2="9" y2="12"/>
          </svg>
        </button>
      </div>
    </div>
  </aside>

  <!-- Main chat -->
  <main class="chat-main">
    <!-- Header -->
    <header class="chat-header">
      <div class="chat-header-info">
        <div class="agent-status" [class.active]="!isStreaming()">
          <span class="status-dot"></span>
        </div>
        <div>
          <h2 class="chat-title">Orquestador Redmine + RAG</h2>
          <p class="chat-subtitle">
            @if (currentNode()) { {{ nodeLabel() }} }
            @else if (isStreaming()) { Generando respuesta… }
            @else { Listo para recibir consultas }
          </p>
        </div>
      </div>
      @if (threadId()) {
        <span class="thread-badge">
          Thread: {{ threadId()!.slice(0, 8) }}…
        </span>
      }
    </header>

    <!-- Messages -->
    <div class="messages-area" #messagesArea>
      @if (messages().length === 0) {
        <div class="empty-state animate-fade-in">
          <div class="empty-icon">
            <svg width="48" height="48" viewBox="0 0 32 32" fill="none">
              <path d="M16 2L30 9V23L16 30L2 23V9L16 2Z" fill="url(#g3)" opacity="0.6"/>
              <defs>
                <linearGradient id="g3" x1="2" y1="2" x2="30" y2="30">
                  <stop offset="0%" stop-color="#6366F1"/>
                  <stop offset="100%" stop-color="#06B6D4"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h3>¿En qué puedo ayudarte?</h3>
          <p>Consultá issues de Redmine, buscá en la base de conocimiento o hacé preguntas generales.</p>
          <div class="suggestions">
            @for (s of suggestions; track s) {
              <button class="suggestion-chip" (click)="sendSuggestion(s)">{{ s }}</button>
            }
          </div>
        </div>
      }

      @for (msg of messages(); track msg.id) {
        <!-- Tool indicators before assistant messages -->
        @if (msg.role === 'assistant' && getToolSteps(msg.id).length > 0) {
          <div class="tool-steps animate-fade-slide-up">
            @for (step of getToolSteps(msg.id); track step.toolCall.id) {
              <app-tool-indicator [step]="step" />
            }
          </div>
        }
        <app-message-bubble [message]="msg" />
      }

      <!-- Typing indicator -->
      @if (isStreaming() && !currentStreamingMsgId()) {
        <div class="typing-indicator animate-fade-in">
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
          <span class="typing-label">El agente está procesando…</span>
        </div>
      }
    </div>

    <!-- Input área -->
    <div class="input-area">
      @if (errorMsg()) {
        <div class="error-banner">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
          </svg>
          {{ errorMsg() }}
          <button class="dismiss-btn" (click)="errorMsg.set(null)">✕</button>
        </div>
      }
      <div class="input-row">
        <textarea
          #inputEl
          class="chat-input"
          [(ngModel)]="inputText"
          placeholder="Escribí tu consulta… (Enter para enviar, Shift+Enter para nueva línea)"
          rows="1"
          [disabled]="isStreaming()"
          (keydown.enter)="onEnter($event)"
          (input)="autoResize()"
          id="chat-input"
          aria-label="Mensaje"
        ></textarea>
        <button
          class="btn btn-primary send-btn"
          [disabled]="!canSend()"
          (click)="sendMessage()"
          id="send-btn"
          aria-label="Enviar mensaje"
        >
          @if (isStreaming()) {
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="animate-spin">
              <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
          } @else {
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          }
        </button>
      </div>
      <p class="input-hint">Redmine MCP + LlamaIndex RAG — <strong>admin / admin1234</strong></p>
    </div>
  </main>
</div>
  `,
  styles: [`
    .chat-layout {
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    /* ── Sidebar ── */
    .sidebar {
      width: var(--sidebar-width);
      background: var(--color-bg-secondary);
      border-right: 1px solid var(--color-border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }

    .sidebar-header {
      padding: var(--space-4) var(--space-5);
      border-bottom: 1px solid var(--color-border);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: var(--space-3);
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.2));
      border: 1px solid var(--color-border);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .brand-name {
      font-size: var(--font-size-lg);
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .sidebar-section {
      padding: var(--space-4) var(--space-3);
    }

    .sidebar-label {
      font-size: var(--font-size-xs);
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      color: var(--color-text-muted);
      padding: 0 var(--space-2);
      margin-bottom: var(--space-2);
    }

    .sidebar-history { flex: 1; overflow-y: auto; }

    .thread-item {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      padding: var(--space-2) var(--space-3);
      border-radius: var(--radius-md);
      font-size: var(--font-size-sm);
      color: var(--color-text-secondary);
      cursor: pointer;

      &.active {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        color: var(--color-text-primary);
      }
    }

    .sidebar-empty {
      font-size: var(--font-size-sm);
      color: var(--color-text-muted);
      padding: 0 var(--space-2);
      font-style: italic;
    }

    .new-chat-btn { width: 100%; justify-content: center; }

    .sidebar-footer {
      padding: var(--space-4) var(--space-3);
      border-top: 1px solid var(--color-border);
    }

    .user-info {
      display: flex;
      align-items: center;
      gap: var(--space-3);
    }

    .user-avatar {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--color-accent-primary), var(--color-accent-secondary));
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: var(--font-size-sm);
      font-weight: 700;
      color: white;
      flex-shrink: 0;
    }

    .user-details {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
    }

    .user-name {
      font-size: var(--font-size-sm);
      font-weight: 500;
      color: var(--color-text-primary);
    }

    .user-role {
      font-size: var(--font-size-xs);
      color: var(--color-text-muted);
      text-transform: capitalize;
    }

    .logout-btn {
      color: var(--color-text-muted);
      &:hover { color: var(--color-accent-error); background: rgba(239,68,68,0.1); }
    }

    /* ── Main chat ── */
    .chat-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      background: var(--color-bg-primary);
    }

    /* Header */
    .chat-header {
      padding: 0 var(--space-6);
      height: var(--header-height);
      border-bottom: 1px solid var(--color-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--color-bg-glass);
      backdrop-filter: blur(10px);
      flex-shrink: 0;
    }

    .chat-header-info { display: flex; align-items: center; gap: var(--space-3); }

    .agent-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--color-text-muted);
      transition: background var(--transition-normal);

      &.active {
        background: var(--color-accent-success);
        box-shadow: 0 0 6px rgba(16,185,129,0.5);
      }
    }

    .chat-title {
      font-size: var(--font-size-md);
      font-weight: 600;
    }

    .chat-subtitle {
      font-size: var(--font-size-xs);
      color: var(--color-text-secondary);
    }

    .thread-badge {
      font-size: var(--font-size-xs);
      font-family: var(--font-mono);
      color: var(--color-text-muted);
      background: var(--color-bg-tertiary);
      border: 1px solid var(--color-border);
      padding: 2px 8px;
      border-radius: var(--radius-full);
    }

    /* Messages */
    .messages-area {
      flex: 1;
      overflow-y: auto;
      padding: var(--space-6);
      display: flex;
      flex-direction: column;
      gap: var(--space-5);
    }

    /* Empty state */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: var(--space-4);
      padding: var(--space-12);
      height: 100%;

      h3 { font-size: var(--font-size-xl); }
      p { color: var(--color-text-secondary); font-size: var(--font-size-md); max-width: 400px; }
    }

    .empty-icon {
      width: 80px; height: 80px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(99,102,241,0.1), transparent);
      display: flex; align-items: center; justify-content: center;
    }

    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
      justify-content: center;
      margin-top: var(--space-2);
    }

    .suggestion-chip {
      padding: var(--space-2) var(--space-4);
      background: var(--color-bg-tertiary);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-full);
      font-size: var(--font-size-sm);
      color: var(--color-text-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);

      &:hover {
        border-color: var(--color-accent-primary);
        color: var(--color-text-primary);
        background: rgba(99,102,241,0.08);
      }
    }

    /* Tool steps */
    .tool-steps {
      display: flex;
      flex-direction: column;
      gap: var(--space-2);
      padding-left: 48px;
    }

    /* Typing indicator */
    .typing-indicator {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      padding-left: 48px;
    }

    .typing-dots {
      display: flex;
      gap: 4px;

      span {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--color-accent-primary);
        animation: pulse 1.4s ease-in-out infinite;

        &:nth-child(2) { animation-delay: 0.2s; }
        &:nth-child(3) { animation-delay: 0.4s; }
      }
    }

    .typing-label { font-size: var(--font-size-sm); color: var(--color-text-muted); }

    /* Input area */
    .input-area {
      padding: var(--space-4) var(--space-6);
      border-top: 1px solid var(--color-border);
      background: var(--color-bg-secondary);
      display: flex;
      flex-direction: column;
      gap: var(--space-3);
    }

    .error-banner {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      padding: var(--space-2) var(--space-4);
      background: rgba(239,68,68,0.1);
      border: 1px solid rgba(239,68,68,0.25);
      border-radius: var(--radius-md);
      font-size: var(--font-size-sm);
      color: #FCA5A5;
    }

    .dismiss-btn {
      margin-left: auto;
      background: none;
      border: none;
      cursor: pointer;
      color: inherit;
      opacity: 0.7;
      &:hover { opacity: 1; }
    }

    .input-row {
      display: flex;
      gap: var(--space-3);
      align-items: flex-end;
    }

    .chat-input {
      flex: 1;
      min-height: 52px;
      max-height: 200px;
      padding: var(--space-3) var(--space-4);
      font-family: var(--font-sans);
      font-size: var(--font-size-md);
      color: var(--color-text-primary);
      background: var(--color-bg-elevated);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-xl);
      outline: none;
      resize: none;
      transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
      line-height: 1.5;

      &::placeholder { color: var(--color-text-muted); }
      &:focus { border-color: var(--color-accent-primary); box-shadow: 0 0 0 3px var(--color-accent-primary-glow); }
      &:disabled { opacity: 0.5; cursor: not-allowed; }
    }

    .send-btn {
      width: 52px;
      height: 52px;
      border-radius: 50%;
      flex-shrink: 0;
      transition: all var(--transition-fast);

      &:not(:disabled):hover { transform: scale(1.05); box-shadow: var(--shadow-glow-primary); }
    }

    .input-hint {
      font-size: var(--font-size-xs);
      color: var(--color-text-muted);
      text-align: center;
    }
  `],
})
export class ChatComponent implements OnInit, OnDestroy {
  @ViewChild('messagesArea') private messagesArea!: ElementRef<HTMLDivElement>;
  @ViewChild('inputEl')      private inputEl!: ElementRef<HTMLTextAreaElement>;

  protected readonly auth    = inject(AuthService);
  private readonly chat      = inject(ChatService);
  private readonly router    = inject(Router);
  private streamSub?: Subscription;

  protected readonly messages              = signal<ChatMessage[]>([]);
  protected readonly threadId              = signal<string | null>(null);
  protected readonly isStreaming           = signal(false);
  protected readonly currentNode           = signal<string | null>(null);
  protected readonly currentStreamingMsgId = signal<string | null>(null);
  protected readonly errorMsg              = signal<string | null>(null);
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
        this.finalizeStreaming(msgId);
        break;

      case 'error':
        this.errorMsg.set(event.message);
        this.finalizeStreaming(msgId);
        break;
    }
  }

  private finalizeStreaming(msgId: string): void {
    this.messages.update(msgs =>
      msgs.map(m => m.id === msgId ? { ...m, isStreaming: false } : m)
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
}
