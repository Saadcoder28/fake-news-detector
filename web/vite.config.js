import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/', // ✅ Ensures proper asset paths
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8080',
      '/batch_predict': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
    },
  },
});
