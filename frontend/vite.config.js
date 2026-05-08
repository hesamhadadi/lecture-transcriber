import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: {
    outDir: "../backend/static",
    emptyOutDir: true
  },
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/videos": "http://127.0.0.1:8000",
      "/transcribe": "http://127.0.0.1:8000"
    }
  }
});
