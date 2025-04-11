import { test as base } from '@playwright/test';

type MyFixtures = {
    baseUrl: string;
    gemmaUrl: string;
};

export const test = base.extend<MyFixtures>({
  baseUrl: ['http://localhost:3000', { option: true }],
  gemmaUrl: async ({ baseUrl }, use) => {
    await use(`${baseUrl}/gemma-scope`);
  },
});