// src/app/app.routes.ts — Rutas de la aplicación
import { Routes } from '@angular/router';
import { authGuard } from './core/auth/auth.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'chat',
    pathMatch: 'full',
  },
  {
    path: 'login',
    loadComponent: () =>
      import('./features/login/login.component').then(m => m.LoginComponent),
  },
  {
    path: 'chat',
    loadComponent: () =>
      import('./features/chat/chat.component').then(m => m.ChatComponent),
    canActivate: [authGuard],
  },
  {
    path: 'tests',
    loadComponent: () =>
      import('./features/tests/tests.component').then(m => m.TestsComponent),
    canActivate: [authGuard],
  },
  {
    path: '**',
    redirectTo: 'chat',
  },
];
