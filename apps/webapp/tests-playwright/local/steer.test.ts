import { expect } from '@playwright/test';
import { test } from '../fixtures';

// http://localhost:3000/steer defaults to localhost:3000/gemma-2-2b-it/steer

test('steer page loads', async ({ page, steerUrl }) => {
  await page.goto(steerUrl);
  await expect(page.locator('text="Steer Models"')).toBeVisible();
});

test('model selector', async ({ page, steerUrl }) => {
  await page.goto(steerUrl);
  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();

  const modelNames = [
    'DeepSeek-R1-Llama-8B',
    'GEMMA-2-2B',
    'GEMMA-2-2B-IT',
    // 'GEMMA-2-9B',
    // 'GEMMA-2-9B-IT',
    'GPT2-SM',
    // 'LLAMA3.1-8B',
  ];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

// tests designed for production
// test('deepseek-r1-llama-8b steer', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/deepseek-r1-distill-llama-8b/steer');

//   // user first occurring demo to search
//   await page.getByRole('button', { name: 'Demo' }).first().click();

//   // check to see if text changed due to a new response
//   await expect(page.getByText("Hey, I'm steered deepseek-r1-distill-llama-8b")).not.toBeVisible({ timeout: 30000 });
// });

// test('gemma-2-2b steer', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gemma-2-2b/steer');

//   // user first occurring demo to search
//   await page.getByRole('button', { name: 'Demo' }).first().click();

//   // check to see if text changed due to a new response
//   await expect(page.getByText("Hey, I'm steered gemma-2-2b!")).not.toBeVisible({ timeout: 30000 });
// });

// test('gemma-2-2b-it steer', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gemma-2-2b-it/steer');

//   // user first occurring demo to search
//   await page.getByRole('button', { name: 'Demo' }).first().click();

//   // check to see if text changed due to a new response
//   await expect(page.getByText("Hey, I'm steered gemma-2-2b-it!")).not.toBeVisible({ timeout: 30000 });
// });

// // No presets
// // test('gemma-2-9b steer', async({ page}) => {
// //     await page.goto('https://neuronpedia.org/gemma-2-9b/steer');

// //     //user first occurring demo to search
// //     await page.getByRole('button', { name: 'Demo' }).first().click();

// //     //check to see if text changed due to a new response
// //     await expect(page.getByText("Hey, I'm steered gemma-2-9b!")).not.toBeVisible({ timeout: 30000 });
// // });

// test('gemma-2-9b-it steer', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gemma-2-9b-it/steer');

//   // user first occurring demo to search
//   await page.getByRole('button', { name: 'Demo' }).first().click();

//   // check to see if text changed due to a new response
//   await expect(page.getByText("Hey, I'm steered gemma-2-9b-it!")).not.toBeVisible({ timeout: 30000 });
// });

// test('gpt2-small steer', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gpt2-small/steer');

//   // user first occurring demo to search
//   await page.getByRole('button', { name: 'Demo' }).first().click();

//   // check to see if text changed due to a new response
//   await expect(page.getByText("Hey, I'm steered gpt2-small!")).not.toBeVisible({ timeout: 30000 });
// });

// test('llama3.1-8b steer', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/llama3.1-8b/steer');

//   // user first occurring demo to search
//   await page.getByRole('button', { name: 'Demo' }).first().click();

//   // check to see if text changed due to a new response
//   await expect(page.getByText("Hey, I'm steered llama3.1-8b!")).not.toBeVisible({ timeout: 30000 });
// });
