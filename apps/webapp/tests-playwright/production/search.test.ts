import { expect } from '@playwright/test';
import { test } from '../fixtures';

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
