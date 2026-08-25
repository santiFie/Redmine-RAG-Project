// src/app/features/tests/tests.component.ts
import { Component, ElementRef, ViewChild, computed, inject, signal } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../core/auth/auth.service';

@Component({
  selector: 'app-tests',
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './tests.component.html',
  styleUrl: './tests.component.scss',
})
export class TestsComponent {
  @ViewChild('dashboardIframe') private dashboardIframe?: ElementRef<HTMLIFrameElement>;

  protected readonly auth = inject(AuthService);
  private readonly sanitizer = inject(DomSanitizer);
  private readonly router = inject(Router);

  protected readonly isLoading = signal(true);
  protected readonly dashboardUrl = signal<SafeResourceUrl>(
    this.sanitizer.bypassSecurityTrustResourceUrl('/results/dashboard.html')
  );

  protected readonly userInitial = computed(() =>
    (this.auth.currentUser()?.displayName ?? 'U')[0].toUpperCase()
  );

  onIframeLoad(): void {
    this.isLoading.set(false);
  }

  reloadDashboard(): void {
    this.isLoading.set(true);
    if (this.dashboardIframe?.nativeElement) {
      this.dashboardIframe.nativeElement.src = '/results/dashboard.html';
    }
  }

  openExternal(): void {
    window.open('/results/dashboard.html', '_blank');
  }
}
