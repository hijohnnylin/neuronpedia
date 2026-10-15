export const NT_TITLE = 'Drift Calibrator';

// Llama 3.1 8B Instruct, the checkpoint the trait readouts are fitted for. The axes are `Vector`
// rows in the database.
export const NT_MODEL_ID = 'llama3.1-8b-it';

export const NT_PAPER_URL = 'https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets/blog/mit-drift-calibrator.pdf';
export const NT_BLOG_URL = '/blog/jpp-drift-ui';
export const NT_CONTACT_URL = 'mailto:skarny@mit.edu,acbaez@mit.edu';

/** Show the behavioral swing into each reply, beside the model's name. */
export const NT_SHOW_BEHAVIOR_SWINGS = true;

/** Show, in a trait's tooltip, example replies its vector was trained on. */
export const NT_SHOW_TRAIT_EXAMPLES = false;

/** The system prompt a new Free Chat starts with. Saved chats keep the one they were made with. */
export const NT_FREE_CHAT_SYSTEM_PROMPT = 'You are an AI assistant. Keep responses concise and conversational.';

/** Saved chats shown in the example bar. Each id is a `SteerOutput` row, loaded with `?savedId=`. */
export const NT_EXAMPLE_CHATS = [
  { emoji: '🐈', title: 'Lost Cat', savedId: 'cmv32w3jb001525gzq33x1j2p' },
  { emoji: '📜', title: 'Poem Review', savedId: 'cmv8zxgji00071snf9hrlyv8y' },
  { emoji: '💔', title: 'Romance Advice', savedId: 'cmv8zzmxg000d1snf6uo18x9d' },
  // Hidden for now.
  // { emoji: '🤝', title: 'Friendship Help', savedId: 'cmv1xx03m000lyiqiriudq4g7' },
  // { emoji: '🫂', title: 'Emotional Support', savedId: 'cmv1tbarw001xnry3lgc6zqwk' },
  // { emoji: '📝', title: 'Poem Feedback', savedId: 'cmv1rlrbo0019nry3lyabhvsq' },
];

/**
 * The trait readouts this page draws, and the order they are placed around the wheel.
 *
 * A list rather than "every axis this model has" because the order is a design decision: the
 * wheel mirrors desirable poles against their opposites, so which axes it holds and where has to
 * be stable between turns. The page looks these names up as `PersonaAxis` rows and draws what it
 * finds -- so this decides what is *asked for*, and the rows decide what an axis says.
 */
export const NT_AXIS_IDS = [
  'mit_empathy',
  // 'mit_erudite' -- simplistic/sophisticated, held back for now. Commented out rather than
  // dropped, since the row is still there and this is the one place that decides.
  'mit_robotic',
  'mit_romantic',
  'mit_sycophantic',
  'mit_toxic',
];
