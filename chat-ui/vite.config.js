import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      // All API routes forwarded to FastAPI on port 8000
      '/health':             { target: 'http://localhost:8000', changeOrigin: true },
      '/chat':               { target: 'http://localhost:8000', changeOrigin: true },
      '/sources':            { target: 'http://localhost:8000', changeOrigin: true },
      '/feedback':           { target: 'http://localhost:8000', changeOrigin: true },
      '/cache':              { target: 'http://localhost:8000', changeOrigin: true },
      '/transcribe':         { target: 'http://localhost:8000', changeOrigin: true },
      '/api':                { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
