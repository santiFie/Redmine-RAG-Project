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
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
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
