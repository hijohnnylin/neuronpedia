import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    // Mirrors the `@/*` aliases in tsconfig.json; vitest does not read those itself.
    alias: { '@': import.meta.dirname },
  },
  test: {
    environment: 'node',
    // Unit tests only here.
    include: ['lib/**/*.test.ts', 'app/**/*.test.ts', 'components/**/*.test.ts'],
  },
});
