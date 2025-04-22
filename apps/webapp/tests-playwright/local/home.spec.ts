import { expect, test } from '@playwright/test';

test.skip('should navigate to an SAE source page', async ({ page }) => {
  await page.goto('/');
  await page.click('text=Get Started');
  await expect(page.getByText('Introduction + Docs')).toBeTruthy();
});
