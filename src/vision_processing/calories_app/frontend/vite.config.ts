import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Écoute sur toutes les adresses IP (0.0.0.0)
    port: 5173,
    proxy: {
      '/token': 'http://127.0.0.1:8000',
      '/users': 'http://127.0.0.1:8000',
      '/coach': 'http://127.0.0.1:8000',
      '/meals': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
    }
  }
})
