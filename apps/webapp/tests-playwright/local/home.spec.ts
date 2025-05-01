import { expect, test } from '@playwright/test';

test('basic home page sanity check', async ({ page }) => {
  await page.goto('/');
  await page.click('text=Get Started');
  await expect(page.getByText('Introduction + Docs')).toBeTruthy();
});
