// Entorno de desarrollo — apunta al BFF local
// En producción, nginx hace proxy de /api → bff, así que la URL base es relativa.
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000',
};
