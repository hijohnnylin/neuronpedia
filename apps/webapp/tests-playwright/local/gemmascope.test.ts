import { expect } from '@playwright/test';
import { test } from '../fixtures';

// http://localhost:3000/gemma-scope#main

test('main page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#main`);

  await page.locator('textarea[name="The inner workings"]').isVisible();
});

test('microscope button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#main`);

  await page.getByText("scan Gemma 2's brain to see what it's thinking").click();
  await expect(page).toHaveURL(`${gemmaUrl}#micropscope`);
});

test('analyze features button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#main`);

  await page.getByText('Make features fire and figure out what they do').click();
  await expect(page).toHaveURL(`${gemmaUrl}#analyze`);
});

test('steer gemma button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#main`);

  await page.getByText("Change Gemma's behavior by manipulating features").click();
  await expect(page).toHaveURL(`${gemmaUrl}#steer`);
});

test('do more button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#main`);

  await page.getByText('Dive deeper into the exciting world of AI interpretability').click();
  await expect(page).toHaveURL(`${gemmaUrl}#learn`);
});

test('browse & search saes button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#main`);

  await page.getByText('Directly explore the SAEs in Gemma Scope').click();
  await expect(page).toHaveURL(`${gemmaUrl}#browse`);
});

test('microscope page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.locator('textarea[name="To understand what AI is thinking"]').isVisible();
});

test('demo button feeling lucky', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.getByText("I'm Feeling Lucky").click();
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('demo button food', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.getByText('Food').click();
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('demo button News', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.getByText('News').click();
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('demo button Literary', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.getByText('Literary').click();
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('demo button Personal', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.getByText('Personal').click();
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('demo button Programming', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  await page.getByText('Programming').click();
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('challenge gemma', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  // challenge is unlocked after doing the demo, or pressing skip demo
  await page.getByText('skip demo').first().click();

  const gemmaSearch = page.getByPlaceholder('Hint: Try a short sentence instead of a word or phrase.');
  await gemmaSearch.fill('testing inputs');
  await gemmaSearch.press('Enter');
  await expect(page.getByText('Click a preset above to send a text to Gemma')).not.toBeVisible({ timeout: 30000 });
});

test('next analyze features button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#microscope`);

  // next is unlocked after doing the demo, or pressing skip demo
  await page.getByText('skip demo').nth(1).click();
  await page.getByText(' Next - Analyze Features').click();
  await expect(page).toHaveURL(`${gemmaUrl}#analyze`);
});

test('analyze features page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);

  await page.locator('textarea[name="The features in Gemma Scope have labels"]').isVisible();
});

test('olympic sports button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);
  await page.getByText('Olympic sports').click();
  await expect(page.getByText('What do these activations (in green) have in common?')).toBeVisible();
});

test('famous cities button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);
  await page.getByText('famous cities').click();
  await expect(page.getByText('What do these activations (in green) have in common?')).toBeVisible();
});

test('reference to animals button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);
  await page.getByText('references to animals').click();
  await expect(page.getByText('Nice! This was tricky, because there are two possibly correct answers.')).toBeVisible();
});

test('none of these button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);
  await page.getByText('none of these').click();
  await expect(page.getByText('Nice! This was tricky, because there are two possibly correct answers.')).toBeVisible();
});

test('analyze steer', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);

  // unlocked after doing the demo, or pressing skip demo
  await page.getByText('skip example').first().click();
  const analyzeSteer = page.getByPlaceholder('Can you write a text snippet that activates it?');
  await analyzeSteer.fill('test');
  await analyzeSteer.press('Enter');

  await expect(page.getByText('0.00')).toBeVisible();
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

test('next steer gemma button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#analyze`);

  // unlocked after puzzles, or pressing skip analyze
  await page.getByText('skip analyze').nth(1).click();
  await page.getByText(' Next - Steer Gemma').click();
  await expect(page).toHaveURL(`${gemmaUrl}#steer`);
});

// test('steer page', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gemma-scope#steer');

//   await page.locator('textarea[name="Let\'s put these features to use"]').isVisible();
// });

// test('steer demo', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gemma-scope#steer');

//   // check for demo question
//   await expect(page.getByText('What are you?').first()).toBeVisible();
// });

// test('next do more button', async ({ page }) => {
//   await page.goto('https://neuronpedia.org/gemma-scope#steer');

//   await page.getByText('Next - Do More').click();
//   await expect(page).toHaveURL('https://www.neuronpedia.org/gemma-scope#learn');
// });

test('do more page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  await page.locator('textarea[name="This demo of Gemma Scope"]').isVisible();
});

test('advanced steer button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    await page.locator('a[target="_blank"][href*="http://localhost:3000/steer"]').click(),
  ]);
  await expect(newPage).toHaveURL(/\/steer$/);
});

test('browse saes button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  await page.getByText('Browse SAEs').click();
  await expect(page).toHaveURL(`${gemmaUrl}#browse`);
});

// test('Contact Us', async({ page }) => {
//   await page.goto(`${gemmaUrl}#learn`);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.locator('a[target="_blank"][href*="mailto:johnny@neuronpedia\.org"]').click()
//   ]);

//   await expect(newPage).toHaveURL(/mailto:johnny@neuronpedia\.org*/);
// });

test('deepmind blog post button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('DeepMind Blog Post').click()]);

  await expect(newPage).toHaveURL(
    'https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/',
  );
});

test('coding tutorial button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('Coding Tutorial').click()]);

  await expect(newPage).toHaveURL(
    'https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp?usp=sharing',
  );
});

// testing it the default method with expect().toHaveURL doesnt work, url ends up being blank
// only tests for the correct link being on the page
test('technical report button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const href = await page.getByText('Technical Report').getAttribute('href');

  expect(href).toContain('storage.googleapis.com/gemma-scope/gemma-scope-report.pdf');
});

test('huggingface button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('HuggingFace').click()]);

  await expect(newPage).toHaveURL('https://huggingface.co/google/gemma-scope');
});

test('get started with mech interp button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    await page.getByText('Get Started with Mech Interp').click(),
  ]);

  await expect(newPage).toHaveURL('https://www.neelnanda.io/mechanistic-interpretability/getting-started');
});

test('favourite mech interp papers button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    await page.getByText('Favourite Mech Interp Papers').click(),
  ]);

  await expect(newPage).toHaveURL(
    'https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1',
  );
});

test('toward monosemanticity button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    await page.getByText('Toward Monosemanticity').click(),
  ]);

  await expect(newPage).toHaveURL('https://transformer-circuits.pub/2023/monosemantic-features');
});

test('saelens button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('SAELens').click()]);

  await expect(newPage).toHaveURL('https://github.com/jbloomAus/SAELens');
});

test('transformerlens button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('TransformerLens').click()]);

  await expect(newPage).toHaveURL('https://github.com/TransformerLensOrg/TransformerLens');
});

test('nnsight button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('NNsight').click()]);

  await expect(newPage).toHaveURL('https://github.com/ndif-team/nnsight');
});

test('mats program button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('MATS Program').click()]);

  await expect(newPage).toHaveURL('https://www.matsprogram.org/');
});

test('arena education button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('ARENA Education').click()]);

  await expect(newPage).toHaveURL('https://www.arena.education/');
});

test('twitter button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), await page.getByText('Twitter').first().click()]);

  await expect(newPage).toHaveURL('https://x.com/neuronpedia');
});

// second contact us button on same page
// test('Contact Us', async({ page }) => {
//   await page.goto(`${gemmaUrl}#learn`);

//   const [newPage] = await Promise.all([
//     page.waitForEvent('popup'),
//     page.locator('a[target="_blank"][href*="mailto:johnny@neuronpedia\.org"]').click()
//   ]);

//   await expect(newPage).toHaveURL(/mailto:johnny@neuronpedia\.org*/);
// });

test('open problems button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  await page.getByRole('button', { name: 'Open Problems' }).click();
  await expect(page).toHaveURL(`${gemmaUrl}#openproblems`);
});

test('playground button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#learn`);

  await page.getByRole('button', { name: 'Playground' }).click();
  await expect(page).toHaveURL(`${gemmaUrl}#playground`);
});

test('open problems page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  await page.locator('textarea[name="Interpretability has many unsolved problems"]').isVisible();
});

test('slack link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Slack').click()]);

  await expect(newPage).toHaveURL(
    'https://opensourcemechanistic.slack.com/join/shared_invite/zt-2o756ku1c-_yKBeUQMVfS_p_qcK6QLeA#/shared-invite/email',
  );
});

// test('lesswrong link', async ({ page }) => {
//     await page.goto(`${gemmaUrl}#openproblems`);

//     const [newPage] = await Promise.all([
//         page.waitForEvent('popup'),
//         page.getByText('LessWrong').click()
//     ]);

//     await expect(newPage).toHaveURL('https://www.lesswrong.com');
// });

test('wattenberg link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Wattenberg et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2407.14662');
});

test('Ziegler link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Ziegler et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2205.01663');
});

test('Steering Vectors link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('steering vectors').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2308.10248');
});

test('SAE feature steering link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('SAE feature steering').click()]);

  await expect(newPage).toHaveURL(
    'https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Activation_Steering_with_SAEs',
  );
});

test('clamping link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('clamping').click()]);

  await expect(newPage).toHaveURL('https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html');
});

test('huang et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Huang et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2309.10312');
});

test('bills et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Bills et al').click()]);

  await expect(newPage).toHaveURL('https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html');
});

// same link appears twice
test('engels et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [firstPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Engels et al').first().click()]);

  await expect(firstPage).toHaveURL('https://arxiv.org/abs/2405.14860');
  await firstPage.close();

  const [secondNewPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByText('Engels et al').nth(1).click(),
  ]);
  await expect(secondNewPage).toHaveURL('https://arxiv.org/abs/2405.14860');
});

test('toy models link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByText('as shown to happen in toy models').click(),
  ]);

  await expect(newPage).toHaveURL(
    'https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy',
  );
});

test('stolfo et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Stolfo et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2305.15054');
});

test('marks et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Marks et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2403.19647');
});

test('mlp transcoders link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('MLP transcoders').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2406.11944v1');
});

test('jain et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Jain et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2407.10264');
});

test('hendel et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Hendel et al').click()]);

  await expect(newPage).toHaveURL('https://arxiv.org/abs/2310.15916');
});

test('Todd et al link', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  const [newPage] = await Promise.all([page.waitForEvent('popup'), page.getByText('Todd et al').click()]);

  await expect(newPage).toHaveURL('https://functions.baulab.info/');
});

test('do more button open problems', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  await page.getByRole('button', { name: 'Do More' }).click();
  await expect(page).toHaveURL(`${gemmaUrl}#learn`);
});

test('home button open problems', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#openproblems`);

  await page.getByRole('button', { name: 'Home' }).click();
  await expect(page).toHaveURL(`${gemmaUrl}#main`);
});

test('playground page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#playground`);

  await page.locator('textarea[name="Enter something below to see"]').isVisible();
});

test('im feeling lucky button', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#playground`);

  await page.getByRole('button', { name: "I'm Feeling Lucky" }).click();
  await expect(
    page.getByText("Enter something below to see what features activate, or try I'm Feeling Lucky."),
  ).not.toBeVisible();
});

test('model selector', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#playground`);
  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').click();

  const modelNames = ['GEMMA-2-2B', 'GEMMA-2-2B-IT', 'GEMMA-2-9B', 'GEMMA-2-9B-IT'];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('source set selector', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#playground`);
  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').click();

  const modelNames = ['gemmascope-att-16k', 'gemmascope-res-16k', 'gemmascope-res-65k'];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('playground steering', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#playground`);

  const gemmaSearch = page.getByPlaceholder('Make gemma-2-2b think about');
  await gemmaSearch.fill('testing inputs');
  await page.getByRole('button', { name: 'Go' }).click();
  await expect(page.getByRole('button', { name: 'testing' })).toBeVisible();
});

test('browse & search page', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  await page.locator('textarea[name="Search 50,000,000+ features"]').isVisible();
});

test('jump to appears', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // Look for the h3 element with the specific classes and text
  const jumpToHeading = page.locator(
    'h3.cursor-default.select-none.text-xl.font-semibold.leading-none.tracking-tight[data-sentry-element="CardTitle"][data-sentry-source-file="jump-to-pane.tsx"]',
  );

  await expect(jumpToHeading).toBeVisible();
  await expect(jumpToHeading).toHaveText('Jump To');
});

test('jump to sae/source models', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // 3 occurences of this combobox on this page
  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').first().click();
  const modelNames = ['GEMMA-2-2B', 'GEMMA-2-2B-IT', 'GEMMA-2-9B', 'GEMMA-2-9B-IT'];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('jump to sae/source source/sae', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // 2 occurences of this combobox on this page
  await page.locator('[data-state="closed"][data-sentry-source-file="source-selector.tsx"]').first().click();
  const modelNames = [
    'gemmascope-att-16k',
    'gemmascope-att-65k',
    'gemmascope-mlp-16k',
    'gemmascope-mlp-65k',
    'gemmascope-res-16k',
    'gemmascope-res-1m',
    'gemmascope-res-262k',
    'gemmascope-res-32k',
    'gemmascope-res-524k',
    'gemmascope-res-65k',
  ];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('jump to feature models', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // 3 occurences of this combobox on this page
  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').nth(1).click();
  const modelNames = ['GEMMA-2-2B', 'GEMMA-2-2B-IT', 'GEMMA-2-9B', 'GEMMA-2-9B-IT'];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('jump to feature source/sae', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // 2 occurences of this combobox on this page
  await page.locator('[data-state="closed"][data-sentry-source-file="source-selector.tsx"]').nth(1).click();
  const modelNames = [
    'gemmascope-att-16k',
    'gemmascope-att-65k',
    'gemmascope-mlp-16k',
    'gemmascope-mlp-65k',
    'gemmascope-res-16k',
    'gemmascope-res-1m',
    'gemmascope-res-262k',
    'gemmascope-res-32k',
    'gemmascope-res-524k',
    'gemmascope-res-65k',
  ];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('search explanations appears', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // Look for the h3 element with the specific classes and text
  const jumpToHeading = page.locator(
    'h3.cursor-default.select-none.text-xl.font-semibold.leading-none.tracking-tight[data-sentry-element="CardTitle"][data-sentry-source-file="search-explanations-pane.tsx"]',
  );

  await expect(jumpToHeading).toBeVisible();
  await expect(jumpToHeading).toHaveText('Search Explanations');
});

test('search via interference appears', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  const jumpToHeading = page.locator(
    'h3.cursor-default.select-none.text-xl.font-semibold.leading-none.tracking-tight[data-sentry-element="CardTitle"][data-sentry-source-file="search-inference-release-pane.tsx"]',
  );

  await expect(jumpToHeading).toBeVisible();
  await expect(jumpToHeading).toHaveText('Search via Inference');
});

test('search via inference models', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  // 3 occurences of this combobox on this page
  await page.locator('[data-state="closed"][data-sentry-source-file="model-selector.tsx"]').nth(2).click();
  const modelNames = ['GEMMA-2-2B', 'GEMMA-2-2B-IT', 'GEMMA-2-9B', 'GEMMA-2-9B-IT'];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});

test('search via inference source/sae', async ({ page, gemmaUrl }) => {
  await page.goto(`${gemmaUrl}#browse`);

  await page.locator('[data-state="closed"][data-sentry-source-file="sourceset-selector.tsx"]').first().click();
  const modelNames = [
    // 'gemmascope-res-131k',
    'gemmascope-res-16k',
  ];

  await Promise.all(
    modelNames.map(model =>
      expect(page.getByText(model, { exact: true }).first()).toBeVisible()
    )
  );
});
