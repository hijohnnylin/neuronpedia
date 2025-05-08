import { expect, test } from '@playwright/test';

test.skip('should navigate to the SAE Evals page', async ({ page }) => {
  await page.goto('/');
  await page.click('text=SAE Evals');
  await expect(page).toHaveURL('/sae-bench');
  await expect(page.getByText('SAEBench')).toBeTruthy();
});
