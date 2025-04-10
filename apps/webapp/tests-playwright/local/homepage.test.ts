import { expect, test } from '@playwright/test';

const baseUrl = 'http://localhost:3000';
const mainUrl = 'https://neuronpedia.org';

test('homepage has expected title', async ({ page }) => {
  await page.goto(baseUrl);
  await expect(page).toHaveTitle(/Neuronpedia/);
});

test('API navigation link', async ({ page }) => {
  await page.goto(baseUrl);

  // Find and click the API link
  const apiLink = page.getByRole('link', { name: 'API', exact: true });
  await apiLink.waitFor({ state: 'visible' });
  await apiLink.click();

  // Wait for navigation and check exact URL
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(baseUrl + '/api-doc');
});

test('SAE Evals link', async ({ page }) => {
  await page.goto(baseUrl);

  // Find and click the SAE Evals link
  const saeEvalsLink = page.getByRole('link', { name: 'SAE Evals', exact: true });
  await saeEvalsLink.waitFor({ state: 'visible' });
  await saeEvalsLink.click();

  // Wait for navigation and check exact URL
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(baseUrl + '/sae-bench');
});

test('Steer', async ({ page }) => {
  await page.goto(baseUrl);

  // Find and click the Steer link
  const steerLink = page.getByRole('link', { name: 'Steer', exact: true });
  await steerLink.waitFor({ state: 'visible' });
  await steerLink.click();

  await expect(page).toHaveURL(/\/steer$/);
});

// Must be logged in to test this
// test('My lists', async({ page }) => {
//   await page.goto('https://neuronpedia.org');

//   // Find and click the My lists link
//   const myListsLink = page.getByRole('link', { name: 'My lists', exact: true });
//   await myListsLink.waitFor({ state: 'visible' });
//   await myListsLink.click();

//   await expect(page).toHaveURL(/.*neuronpedia\.org\/user\/.*\/lists/);
// })

// Must be logged in to test this
// /* test('My vectors', async({ page }) => {
//   await page.goto('https://neuronpedia.org');
//   await expect(page).toHaveURL(/.*neuronpedia\.org\/user\/.*\/vectors/);
// }) */

// no local docs currently
// test('Getting Started', async ({ page }) => {
//   await page.goto(baseUrl);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.getByRole('link', { name: 'Getting Started', exact: true }).click(),
//   ]);

//   await expect(newPage).toHaveURL('https://docs.neuronpedia.org/');
// });

test('MIT Technology Review link', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.locator('a[target="_blank"][href*="technologyreview.com"]').click(),
  ]);

  await expect(newPage).toHaveURL(/.*technologyreview\.com/);
});

test('Google Deepmind', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.locator('a[target="_blank"][href*="gemma-scope"]').click(),
  ]);

  await expect(newPage).toHaveURL(/.*gemma-scope/);
});

test('Fudan University', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.locator('a[target="_blank"][href*="llama-scope"]').click(),
  ]);

  await expect(newPage).toHaveURL(/.*llama-scope/);
});

test('Apollo Research', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.locator('a[target="_blank"][href*="gpt2sm-apollojt"]').click(),
  ]);

  await expect(newPage).toHaveURL(/.*gpt2sm-apollojt/);
});

// no link
// test('ML Alignment & Theory Scholars', async({ page }) => {
//   await page.goto('https://neuronpedia.org');

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.getByRole('link', { name: 'MATS' }).click()
//   ]);

//   await expect(newPage).toHaveURL('#mats');
// });

test('EleutherAI', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.locator('a[target="_blank"][href*="llama3.1-8b-eleuther_gp"]').click(),
  ]);

  await expect(newPage).toHaveURL(/.*llama3\.1-8b-eleuther_gp/);
});

// no local docs
// test('latents/features', async ({ page }) => {
//   await page.goto(baseUrl);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.getByRole('link', { name: 'latents/features', exact: true }).click(),
//   ]);

//   await expect(newPage).toHaveURL('https://docs.neuronpedia.org/features');
// });

test('concepts', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('link', { name: 'concepts', exact: true }).click(),
  ]);

  await expect(newPage).toHaveURL(baseUrl + '/axbench');
});

test('releases display', async ({ page }) => {
  await page.goto(baseUrl);
  await expect(page.getByText('Llama Scope R1: SAEs for DeepSeek-R1-Distill-Llama-8B')).toBeVisible();
});

test('models display', async ({ page }) => {
  await page.goto(baseUrl);
  await expect(page.getByText('DeepSeek-R1-Llama-8B')).toBeVisible();
});

test('jump to display', async ({ page }) => {
  await page.goto(baseUrl);
  await expect(page.getByText('jump to source/sae')).toBeVisible();
});

test('cat steering', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('link', { name: 'Try It: Gemma 2 - Cat Steering', exact: true }).click(),
  ]);

  await expect(newPage).toHaveURL(baseUrl + '/gemma-2-9b-it/steer?saved=cm7cp63af00jx1q952neqg6e5');
});

test('search by explanation', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('link', { name: 'Try It: Search by Explanation', exact: true }).click(),
  ]);

  await expect(newPage).toHaveURL( baseUrl + '/search-explanations');
});

// no local docs
// test('search via interference', async ({ page }) => {
//   await page.goto(baseUrl);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.getByRole('link', { name: 'Docs: Search via Inference', exact: true }).click(),
//   ]);

//   await expect(newPage).toHaveURL('https://docs.neuronpedia.org/search');
// });

test('api + libraries', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('link', { name: 'API Playground', exact: true }).click(),
  ]);

  await expect(newPage).toHaveURL( baseUrl + '/api-doc');
});

test('searcher is embedded in the page', async ({ page }) => {
  await page.goto(baseUrl);
  await page.locator('textarea[name="searchQuery"]').isVisible();
});

test('searcher example', async ({ page }) => {
  await page.goto(baseUrl);
  await page.locator('textarea[name="Test activation with custom text./"]').isVisible();
});

// no local docs
// test('Docs: Lists', async ({ page }) => {
//   await page.goto(baseUrl);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.getByRole('link', { name: 'Docs: Lists', exact: true }).click(),
//   ]);

//   await expect(newPage).toHaveURL('https://docs.neuronpedia.org/lists');
// });

// test('Docs: Embed', async ({ page }) => {
//   await page.goto(baseUrl);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.getByRole('link', { name: 'Docs: Embed', exact: true }).click(),
//   ]);

//   await expect(newPage).toHaveURL('https://docs.neuronpedia.org/embed-iframe');
// });

test('Slack', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('button', { name: 'Slack' }).click(),
  ]);

  await expect(newPage).toHaveURL(/.*slack\.com*/);
});

test('Donate', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('button', { name: 'Donate' }).click(),
  ]);

  await expect(newPage).toHaveURL(/.*every\.org*/);
});

test('Upskill', async ({ page }) => {
  await page.goto(baseUrl);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('button', { name: 'Upskill' }).click(),
  ]);

  await expect(newPage).toHaveURL('https://www.arena.education/');
});

// online service
// test('all services are online', async ({ page }) => {
//   await page.goto(baseUrl);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page
//       .locator('iframe[title="Neuronpedia Status"]')
//       .contentFrame()
//       .getByRole('link', { name: 'All services are online' })
//       .click(),
//   ]);

//   await expect(newPage).toHaveURL('https://status.neuronpedia.org/');
// });
