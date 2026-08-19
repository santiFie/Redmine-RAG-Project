// src/app/core/auth/auth.service.ts
// ===================================
// Servicio de autenticación con Signals.
// Gestiona el JWT y el estado de sesión del usuario.

import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';
import { environment } from '../../../environments/environment';

/** Respuesta del endpoint POST /auth/login */
export interface LoginResponse {
  access_token: string;
  token_type: string;
  display_name: string;
  role: string;
}

/** Datos del usuario autenticado almacenados en el servicio */
export interface CurrentUser {
  username: string;
  displayName: string;
  role: string;
}

const TOKEN_KEY = 'ag_access_token';
const USER_KEY  = 'ag_current_user';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http   = inject(HttpClient);
  private readonly router = inject(Router);

  // ── Signals ──────────────────────────────────────────────────────────────
  private readonly _token = signal<string | null>(localStorage.getItem(TOKEN_KEY));
  private readonly _user  = signal<CurrentUser | null>(
    JSON.parse(localStorage.getItem(USER_KEY) ?? 'null')
  );

  /** Token JWT actual. Null si no hay sesión activa. */
  readonly token = this._token.asReadonly();

  /** Usuario actualmente autenticado. Null si no hay sesión. */
  readonly currentUser = this._user.asReadonly();

  /** True si hay un token almacenado (puede estar expirado; el servidor valida). */
  readonly isAuthenticated = computed(() => !!this._token());

  // ── API ───────────────────────────────────────────────────────────────────

  /**
   * Autentica al usuario con el BFF y almacena el JWT.
   * @param username Nombre de usuario (demo: admin)
   * @param password Contraseña (demo: admin1234)
   */
  login(username: string, password: string): Observable<LoginResponse> {
    return this.http
      .post<LoginResponse>(`${environment.apiUrl}/auth/login`, { username, password })
      .pipe(
        tap((res) => {
          const user: CurrentUser = {
            username,
            displayName: res.display_name,
            role: res.role,
          };
          this._token.set(res.access_token);
          this._user.set(user);
          localStorage.setItem(TOKEN_KEY, res.access_token);
          localStorage.setItem(USER_KEY, JSON.stringify(user));
        })
      );
  }

  /** Cierra sesión, limpia el estado local y redirige al login. */
  logout(): void {
    this._token.set(null);
    this._user.set(null);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    this.router.navigate(['/login']);
  }

  /** Retorna el token actual para uso en el interceptor y el chat service. */
  getToken(): string | null {
    return this._token();
  }
}
