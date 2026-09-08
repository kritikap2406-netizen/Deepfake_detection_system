import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxies /analyze, /result, /health, /static to the FastAPI backend
// (Phase 14, running on :8000 by default) so the frontend can call
// relative paths in dev without CORS configuration.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/analyze': 'http://127.0.0.1:8000',
      '/result': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
    },
  },
})
