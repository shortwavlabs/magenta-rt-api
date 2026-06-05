import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_MAGENTA_RT_PROXY_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (proxyPath) => proxyPath.replace(/^\/api/, ""),
      },
    },
  },
});
