import { expect } from '@playwright/test';
import { test } from '../fixtures';

test('SAE Evals link', async ({ page, baseUrl }) => {
    await page.goto(baseUrl);
  
    // Find and click the SAE Evals link
    const saeEvalsLink = page.getByRole('link', { name: 'SAE Evals', exact: true });
    await saeEvalsLink.waitFor({ state: 'visible' });
    await saeEvalsLink.click();
  
    // Wait for navigation and check exact URL
    await page.waitForLoadState('networkidle');
    await expect(page).toHaveURL(`${baseUrl}/sae-bench`);
  });