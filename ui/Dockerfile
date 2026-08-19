# ==============================================================================
# UI (Angular) — Dockerfile
# Multi-stage: build con Node → serve con nginx
# ==============================================================================

FROM node:22-alpine AS builder

WORKDIR /app

# Instalar dependencias
COPY package*.json ./
RUN npm ci --prefer-offline

# Copiar fuentes y construir en modo producción
COPY . .
RUN npm run build -- --configuration production

# ── Stage: nginx ───────────────────────────────────────────────────────────

FROM nginx:1.27-alpine AS production

# Configuración de nginx (SPA + proxy /api → bff)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar build de Angular
COPY --from=builder /app/dist/ui/browser /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD wget --spider -q http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]
