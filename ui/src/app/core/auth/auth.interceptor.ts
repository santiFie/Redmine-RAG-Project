// src/app/core/auth/auth.interceptor.ts
// =======================================
// Interceptor HTTP funcional (Angular 15+) que inyecta el JWT Bearer token
// en todas las peticiones al BFF.

import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from './auth.service';
import { environment } from '../../../environments/environment';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();

  // Solo agregar el header si hay token y la request es hacia el BFF
  if (token && req.url.startsWith(environment.apiUrl)) {
    req = req.clone({
      setHeaders: { Authorization: `Bearer ${token}` },
    });
  }

  return next(req);
};
