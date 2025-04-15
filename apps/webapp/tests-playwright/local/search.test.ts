import { expect } from '@playwright/test';
import { test } from '../fixtures';

// http:localhost:3000/search
test('search models', async ({ page, searchUrl }) => {
  await page.goto(searchUrl);

  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  const modelNames = [
    'DEEPSEEK-R1-LLAMA-8B',
    'GEMMA-2-2B',
    'GEMMA-2-2B-IT',
    'GEMMA-2-9B',
    'GEMMA-2-9B-IT',
    'GPT2-SM',
    'LLAMA3.1-8B',
  ];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('deepseek sae', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  await page.getByText('DEEPSEEK-R1-LLAMA-8B').click();

  await expect(page.getByText('llamascope-openr1-res-32k')).toBeVisible();
});

test('gemma-2-2b sae', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  // currently default search model
  // await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  // await page.getByText('GEMMA-2-2B').click();

  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').click();
  const saeNames = ['gemmascope-att-16k', 'gemmascope-res-16k', 'gemmascope-res-65k'];

  await Promise.all(
    saeNames.map(saeset =>
      expect(page.getByText(saeset, { exact: true }).first()).toBeVisible()
    )
  );
});

test('gemma-2-9b sae', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  await page.getByText('GEMMA-2-9B').first().click();

  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').click();
  const saeNames = ['gemmascope-res-16k'];

  await Promise.all(
    saeNames.map(saeset =>
      expect(page.getByText(saeset, { exact: true }).first()).toBeVisible()
    )
  );
});

test('gemma-2-9b-it sae', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  await page.getByText('GEMMA-2-9B-IT').first().click();

  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').click();
  const saeNames = ['gemmascope-res-131k', 'gemmascope-res-16k'];

  await Promise.all(
    saeNames.map(saeset =>
      expect(page.getByText(saeset, { exact: true }).first()).toBeVisible()
    )
  );
});

test('gpt2-sm sae', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  await page.getByText('GPT2-SM').first().click();

  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').click();
  const saeNames = [
    'att_32k-oai',
    'att-kk',
    'mlp_32k-oai',
    'res_fs12288-jb',
    'res_fs1536-jb',
    'res_fs24576-jb',
    'res_fs3072-jb',
    'res_fs49152-jb',
    'res_fs6144-jb',
    'res_fs768-jb',
    'res_fs98304-jb',
    'res_mid_32k-oai',
    'res_post_32k-oai',
    'res_sce-ajt',
    'res_scefr-ajt',
    'res_scl-ajt',
    'res_sle-ajt',
    'res_slefr-ajt',
    'res_sll-ajt',
    'res-jb',
  ];

  await Promise.all(
    saeNames.map(saeset =>
      expect(page.getByText(saeset, { exact: true }).first()).toBeVisible()
    )
  );
});

test('llama3.1-8b sae', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();
  await page.getByText('LLAMA3.1-8B').first().click();

  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').click();
  const saeNames = ['llamascope-res-32k'];

  await Promise.all(
    saeNames.map(saeset =>
      expect(page.getByText(saeset, { exact: true }).first()).toBeVisible()
    )
  );
});

// checks for new url in the case that the default model is changed
test('random button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Random').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('food button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Food').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('news button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('News').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('literary button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Literary').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('personal button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Personal').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('Programming button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Programming').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('Techinical button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Technical').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('academic button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Academic').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('business button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Business').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('legal button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Legal').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('educational button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Educational').click();
  await expect(page).not.toHaveURL(searchUrl);
});

test('Cultural button', async ({ page,searchUrl }) => {
  await page.goto(searchUrl);

  await page.getByText('Cultural').click();
  await expect(page).not.toHaveURL(searchUrl);
});
