import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["src/App.test.jsx"],
    setupFiles: ["./src/setupTests.js"],
    pool: "forks",
    maxWorkers: 1,
    minWorkers: 1,
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});