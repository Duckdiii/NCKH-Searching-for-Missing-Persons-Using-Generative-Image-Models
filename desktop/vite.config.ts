import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      '/outputs': 'http://127.0.0.1:8000',
      '/data': 'http://127.0.0.1:8000',
    },
  },
  // @ts-ignore
  test: {
    environment: 'jsdom',
    globals: true,
  }
})
