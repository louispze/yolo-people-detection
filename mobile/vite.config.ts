import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Capacitor charge les assets en chemins relatifs -> base: "./"
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: { host: true, port: 5173 },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
