/* eslint-disable import/no-extraneous-dependencies */
import { loadEnvConfig } from '@next/env';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';
import { defineConfig } from 'vitest/config';

loadEnvConfig(process.cwd());

export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    // https://github.com/vitest-dev/vitest/issues/990
    globals: true,
    include: ['components/**/*.test.tsx'],
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    deps: {
      inline: ['vitest-canvas-mock'],
    },
    environmentOptions: {
      jsdom: {
        resources: 'usable',
      },
    },
    coverage: {
      provider: 'v8',
      include: ['components/**/*.{ts,tsx}'],
      exclude: ['**/*.test.{ts,tsx}', '**/__tests__/**'],
    },
  },
});
