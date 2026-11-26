import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite default dev port 5173. The dashboard talks to the ROS backend over
// rosbridge (ws://localhost:9090) and web_video_server (http://localhost:8080)
// directly from the browser — no Vite proxy needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
  },
});
