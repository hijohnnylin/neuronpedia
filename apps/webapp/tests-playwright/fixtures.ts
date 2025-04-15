import { test as base } from '@playwright/test';

type MyFixtures = {
    baseUrl: string;
    gemmaUrl: string;
    steerUrl: string;
    searchUrl: string;
};

export const test = base.extend<MyFixtures>({
  baseUrl: ['http://localhost:3000', { option: true }],
  gemmaUrl: async ({ baseUrl }, use) => {
    await use(`${baseUrl}/gemma-scope`);
  },
  steerUrl: async ({ baseUrl }, use) => {
    await use(`${baseUrl}/steer`)
  },
  searchUrl: async ({ baseUrl }, use) => {
    await use(`${baseUrl}/search`);
  }
});