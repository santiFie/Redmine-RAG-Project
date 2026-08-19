// Entorno de producción — en Docker, nginx proxea /api al BFF
// El frontend se sirve desde el mismo dominio, por lo que la URL es relativa.
export const environment = {
  production: true,
  apiUrl: '/api',
};
