import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    // Mirrors the `@/*` aliases in tsconfig.json; vitest does not read those itself.
    alias: { '@': import.meta.dirname },
  },
  // Next compiles JSX itself, so tsconfig leaves it as `preserve` -- which esbuild reads as "this
  // file is already JS" and then fails to parse. Stated here so a test may import a module that
  // happens to hold a component, rather than only the pure `.ts` files beside it.
  oxc: { jsx: 'automatic' },
  test: {
    environment: 'node',
    // Unit tests only here.
    include: ['lib/**/*.test.ts', 'app/**/*.test.ts', 'components/**/*.test.ts'],
  },
});
