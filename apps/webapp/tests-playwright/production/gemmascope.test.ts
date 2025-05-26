import { expect } from '@playwright/test';
import { test } from '../fixtures';

test('gemma steer demo', async ({ page }) => {
    await page.goto('https://neuronpedia.org/gemma-scope#steer');
  
    await page.click('text=');
    await page.getByRole('button', { name: 'Tell me about yourself.' }).first().click();
    await page.waitForTimeout(3000);
    await expect(page.getByText("Sorry, your message could not be sent at this time. Please try again later.")).not.toBeVisible({ timeout: 30000 });
  });

  test('puzzles labels', async ({ page, gemmaUrl }) => {
    await page.goto(`${gemmaUrl}#analyze`);
  
    // unlocked after doing analyze steer, or pressing skip analyze
    await page.getByText('skip analyze').first().click();
  
    // test all 3 buttons
    const labelButtons = await page.getByRole('button', { name: 'Reveal Our Label' }).all();
    // click each button
    await Promise.all(
      labelButtons.map(button => button.click())
    );
  
    // check labels
    await expect(page.getByText('Lies and falsehoods')).toBeVisible();
    await expect(page.getByText('Misspellings or typos')).toBeVisible();
    await expect(page.getByText('Bad/cringe stories')).toBeVisible();
  });