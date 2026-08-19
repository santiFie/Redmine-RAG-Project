// src/app/features/login/login.component.ts
// ===========================================
// Pantalla de login con diseño dark corporativo y animaciones.

import {
  Component,
  inject,
  signal,
} from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [ReactiveFormsModule],
  template: `
    <div class="login-page">
      <!-- Fondo animado con partículas/gradiente -->
      <div class="login-bg">
        <div class="bg-orb bg-orb--1"></div>
        <div class="bg-orb bg-orb--2"></div>
        <div class="bg-orb bg-orb--3"></div>
      </div>

      <!-- Card de login -->
      <div class="login-card animate-fade-slide-up">
        <!-- Logo -->
        <div class="login-logo">
          <div class="logo-icon" aria-hidden="true">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <path d="M16 2L30 9V23L16 30L2 23V9L16 2Z"
                    fill="url(#grad)" stroke="none"/>
              <path d="M10 16L14 20L22 12" stroke="white" stroke-width="2.5"
                    stroke-linecap="round" stroke-linejoin="round"/>
              <defs>
                <linearGradient id="grad" x1="2" y1="2" x2="30" y2="30">
                  <stop offset="0%" stop-color="#6366F1"/>
                  <stop offset="100%" stop-color="#06B6D4"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div class="logo-text">
            <h1 class="gradient-text">AgentOS</h1>
            <p class="logo-subtitle">Redmine + RAG Orchestrator</p>
          </div>
        </div>

        <!-- Formulario -->
        <form
          class="login-form"
          [formGroup]="loginForm"
          (ngSubmit)="onSubmit()"
          novalidate
          id="login-form"
        >
          <h2 class="login-title">Iniciar sesión</h2>
          <p class="login-desc">Ingresá tus credenciales para acceder al orquestador.</p>

          <!-- Usuario -->
          <div class="field-group">
            <label for="username" class="field-label">Usuario</label>
            <input
              id="username"
              type="text"
              class="input-field"
              formControlName="username"
              placeholder="admin"
              autocomplete="username"
              [class.input-error]="isFieldInvalid('username')"
            />
            @if (isFieldInvalid('username')) {
              <span class="field-error">El usuario es requerido</span>
            }
          </div>

          <!-- Contraseña -->
          <div class="field-group">
            <label for="password" class="field-label">Contraseña</label>
            <div class="input-wrapper">
              <input
                id="password"
                [type]="showPassword() ? 'text' : 'password'"
                class="input-field"
                formControlName="password"
                placeholder="••••••••"
                autocomplete="current-password"
                [class.input-error]="isFieldInvalid('password')"
              />
              <button
                type="button"
                class="btn btn-icon toggle-password"
                (click)="showPassword.set(!showPassword())"
                [attr.aria-label]="showPassword() ? 'Ocultar contraseña' : 'Mostrar contraseña'"
                id="toggle-password-btn"
              >
                @if (showPassword()) {
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
                    <line x1="1" y1="1" x2="23" y2="23"/>
                  </svg>
                } @else {
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                    <circle cx="12" cy="12" r="3"/>
                  </svg>
                }
              </button>
            </div>
            @if (isFieldInvalid('password')) {
              <span class="field-error">La contraseña es requerida</span>
            }
          </div>

          <!-- Error de autenticación -->
          @if (authError()) {
            <div class="alert alert-error" role="alert">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
              {{ authError() }}
            </div>
          }

          <!-- Botón de submit -->
          <button
            type="submit"
            class="btn btn-primary login-submit"
            [disabled]="isLoading()"
            id="login-submit-btn"
          >
            @if (isLoading()) {
              <span class="spinner" aria-hidden="true"></span>
              Autenticando...
            } @else {
              Ingresar
            }
          </button>
        </form>

        <!-- Hint de demo -->
        <div class="demo-hint">
          <span class="demo-badge">DEMO</span>
          <code>admin</code> / <code>admin1234</code>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .login-page {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      background: var(--color-bg-primary);
      position: relative;
      overflow: hidden;
    }

    /* Orbes de fondo animados */
    .login-bg {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }

    .bg-orb {
      position: absolute;
      border-radius: 50%;
      filter: blur(80px);
      opacity: 0.12;
      animation: pulse 6s ease-in-out infinite;
    }

    .bg-orb--1 {
      width: 500px;
      height: 500px;
      background: var(--color-accent-primary);
      top: -120px;
      right: -100px;
      animation-delay: 0s;
    }

    .bg-orb--2 {
      width: 400px;
      height: 400px;
      background: var(--color-accent-secondary);
      bottom: -100px;
      left: -80px;
      animation-delay: 2s;
    }

    .bg-orb--3 {
      width: 300px;
      height: 300px;
      background: var(--color-accent-primary);
      bottom: 20%;
      right: 15%;
      animation-delay: 4s;
    }

    /* Card principal */
    .login-card {
      position: relative;
      z-index: 1;
      width: 100%;
      max-width: 420px;
      padding: 2.5rem;
      background: var(--color-bg-glass);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-2xl);
      box-shadow: var(--shadow-lg), 0 0 0 1px rgba(255,255,255,0.03);

      @media (max-width: 480px) {
        margin: 1rem;
        padding: 1.75rem;
      }
    }

    /* Logo */
    .login-logo {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      margin-bottom: 2rem;
    }

    .logo-icon {
      width: 48px;
      height: 48px;
      border-radius: var(--radius-lg);
      background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.2));
      border: 1px solid var(--color-border);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .logo-text h1 {
      font-size: var(--font-size-xl);
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .logo-subtitle {
      font-size: var(--font-size-xs);
      color: var(--color-text-muted);
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    /* Formulario */
    .login-title {
      font-size: var(--font-size-xl);
      font-weight: 600;
      margin-bottom: var(--space-2);
    }

    .login-desc {
      font-size: var(--font-size-sm);
      color: var(--color-text-secondary);
      margin-bottom: var(--space-6);
    }

    .login-form {
      display: flex;
      flex-direction: column;
      gap: var(--space-5);
    }

    .field-group {
      display: flex;
      flex-direction: column;
      gap: var(--space-2);
    }

    .field-label {
      font-size: var(--font-size-sm);
      font-weight: 500;
      color: var(--color-text-secondary);
    }

    .input-wrapper {
      position: relative;
    }

    .input-wrapper .input-field {
      padding-right: 44px;
    }

    .toggle-password {
      position: absolute;
      right: var(--space-2);
      top: 50%;
      transform: translateY(-50%);
      color: var(--color-text-muted);

      &:hover { color: var(--color-text-primary); background: none; }
    }

    .input-error {
      border-color: var(--color-accent-error) !important;
      box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.15) !important;
    }

    .field-error {
      font-size: var(--font-size-xs);
      color: var(--color-accent-error);
    }

    /* Alert */
    .alert {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      padding: var(--space-3) var(--space-4);
      border-radius: var(--radius-md);
      font-size: var(--font-size-sm);
    }

    .alert-error {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #FCA5A5;
    }

    .login-submit {
      width: 100%;
      height: 44px;
      font-size: var(--font-size-md);
      margin-top: var(--space-2);
    }

    /* Spinner */
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    /* Demo hint */
    .demo-hint {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: var(--space-2);
      margin-top: var(--space-6);
      font-size: var(--font-size-sm);
      color: var(--color-text-muted);

      code {
        background: var(--color-bg-elevated);
        border-color: var(--color-border-subtle);
        color: var(--color-accent-secondary);
      }
    }

    .demo-badge {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 1px;
      padding: 2px 6px;
      background: rgba(6,182,212,0.15);
      border: 1px solid rgba(6,182,212,0.3);
      border-radius: var(--radius-sm);
      color: var(--color-accent-secondary);
    }
  `],
})
export class LoginComponent {
  private readonly fb     = inject(FormBuilder);
  private readonly auth   = inject(AuthService);
  private readonly router = inject(Router);

  protected readonly showPassword = signal(false);
  protected readonly isLoading    = signal(false);
  protected readonly authError    = signal<string | null>(null);

  protected readonly loginForm = this.fb.group({
    username: ['', Validators.required],
    password: ['', Validators.required],
  });

  protected isFieldInvalid(field: string): boolean {
    const control = this.loginForm.get(field);
    return !!(control?.invalid && control.touched);
  }

  protected onSubmit(): void {
    if (this.loginForm.invalid) {
      this.loginForm.markAllAsTouched();
      return;
    }

    this.isLoading.set(true);
    this.authError.set(null);

    const { username, password } = this.loginForm.value;

    this.auth.login(username!, password!).subscribe({
      next: () => this.router.navigate(['/chat']),
      error: (err) => {
        this.isLoading.set(false);
        this.authError.set(
          err.status === 401
            ? 'Usuario o contraseña incorrectos.'
            : 'Error de conexión. Verificá que el servidor esté disponible.'
        );
      },
    });
  }
}
