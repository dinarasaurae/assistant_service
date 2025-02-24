import { defineConfig } from 'vite';
import angular from '@analogjs/vite-plugin-angular';

export default defineConfig({
  plugins: [angular()],
  define: {
    'import.meta.env.VITE_API_URL': JSON.stringify(process.env['VITE_API_URL'] || 'http://94.126.205.209:8001/ask')
  }
});
