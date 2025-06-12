import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Activation, ExplanationModelType, Neuron, UserSecretType } from '@prisma/client';
import OpenAI from 'openai';
import { AutoInterpModelType, OPENROUTER_BASE_URL } from '../utils/autointerp';
import { makeAnthropicMessage, makeGeminiMessage, makeOaiMessage } from './autointerp-shared';

const TOKENS_AROUND_MAX_ACTIVATING_TOKEN = 24;

const systemMessage = `You are explaining the behavior of a neuron in a neural network. Your response should be a very concise explanation (1-6 words) that captures what the neuron detects or predicts by finding patterns in lists.

To determine the explanation, you are given four lists:

- MAX_ACTIVATING_TOKENS, which are the top activating tokens in the top activating texts.
- TOKENS_AFTER_MAX_ACTIVATING_TOKEN, which are the tokens immediately after the max activating token.
- TOP_POSITIVE_LOGITS, which are the most likely words or tokens associated with this neuron.
- TOP_ACTIVATING_TEXTS, which are top activating texts.

You should look for a pattern by trying the following methods in order. Once you find a pattern, stop and return that pattern. Do not proceed to the later methods.
Method 1: Look at MAX_ACTIVATING_TOKENS. If they share something specific in common, or are all the same token or a variation of the same token (like different cases or conjugations), respond with that token.
Method 2: Look at TOKENS_AFTER_MAX_ACTIVATING_TOKEN. Try to find a specific pattern or similarity in all the tokens. A common pattern is that they all start with the same letter. If you find a pattern (like 's word', 'the ending -ing', 'number 8'), respond with 'say [the pattern]'. You can ignore uppercase/lowercase differences for this.
Method 3: Look at TOP_POSITIVE_LOGITS for similarities and describe it very briefly (1-3 words).
Method 4: Look at TOP_ACTIVATING_TEXTS and make a best guess by describing the broad theme or context, ignoring the max activating tokens.

Rules:
- Keep your explanation extremely concise (1-6 words, mostly 1-3 words).
- Do not add unnecessary phrases like "words related to", "concepts related to", or "variations of the word".
- Do not mention "tokens" or "patterns" in your explanation.
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".
- Remember to use the 'say [the pattern]' when using Method 2 above (pattern found in TOKENS_AFTER_MAX_ACTIVATING_TOKEN).
- If you absolutely cannot make any guesses, return the first token in MAX_ACTIVATING_TOKENS.

Respond by going through each method number until you find one that helps you find an explanation for what this neuron is detecting or predicting. If a method does not help you find an explanation, briefly explain why it does not, then go on to the next method.
Finally, end your response with the method number you used, the reason for your explanation, and then the explanation.`;

const firstUserMessage = `

Neuron 1

<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>

was
watching

</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>


<MAX_ACTIVATING_TOKENS>

She
enjoy

</MAX_ACTIVATING_TOKENS>


<TOP_POSITIVE_LOGITS>

walking
WA
waiting
was
we
WHAM
wish
win
wake
whisper

</TOP_POSITIVE_LOGITS>


<TOP_ACTIVATING_TEXTS>

She was taking a nap when her phone started ringing.
I enjoy watching movies with my family.

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 1 behavior: `;

const firstAssistantMessage = `Method 1 fails: MAX_ACTIVATING_TOKENS (She, enjoy) are not similar tokens.
Method 2 succeeds: All TOKENS_AFTER_MAX_ACTIVATING_TOKEN have a pattern in common: they all start with "w". Explanation: say "w" words`;

const secondUserMessage = `

Neuron 2

<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>

are
,

</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>


<MAX_ACTIVATING_TOKENS>

banana
blueberries

</MAX_ACTIVATING_TOKENS>


<TOP_POSITIVE_LOGITS>

apple
orange
pineapple
watermelon
kiwi
peach
pear
grape
cherry
plum

</TOP_POSITIVE_LOGITS>


<TOP_ACTIVATING_TEXTS>

The apple and banana are delicious foods that provide essential vitamins and nutrients.
I enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 2 behavior: `;

const secondAssistantMessage =
  'Method 1 succeeds: All MAX_ACTIVATING_TOKENS (banana, blueberries) are fruits. Explanation: fruits';

const thirdUserMessage = `

Neuron 3

<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>

warm
the

</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>


<MAX_ACTIVATING_TOKENS>

and
And

</MAX_ACTIVATING_TOKENS>


<TOP_POSITIVE_LOGITS>

elephant
guitar
mountain
bicycle
ocean
telescope
candle
umbrella
tornado
butterfly

</TOP_POSITIVE_LOGITS>


<TOP_ACTIVATING_TEXTS>

It was a beautiful day outside with clear skies and warm sunshine.
And the garden has roses and tulips and daisies and sunflowers blooming together.

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 3 behavior: `;

const thirdAssistantMessage = 'Method 1 succeeds: All MAX_ACTIVATING_TOKENS are the word "and". Explanation: and';

const fourthUserMessage = `

Neuron 4

<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>

was
places

</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>


<MAX_ACTIVATING_TOKENS>

war
some

</MAX_ACTIVATING_TOKENS>


<TOP_POSITIVE_LOGITS>

4
four
fourth
4th
IV
Four
FOUR
~4
4.0
quartet

</TOP_POSITIVE_LOGITS>


<TOP_ACTIVATING_TEXTS>

the civil war was a major topic in history class .
seasons of the year are winter , spring , summer , and fall or autumn in some places .

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 4 behavior: `;

const fourthAssistantMessage = `Method 1 fails: MAX_ACTIVATING_TOKENS (war, some) are not all the same token.
Method 2 fails: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (was, places) are not all similar tokens and don't have a text pattern in common.
Method 3 succeeds: All TOP_POSITIVE_LOGITS are the number 4. Explanation: 4`;

const formatTokensAfterMaxActivatingToken = (activations: Activation[]): string => {
  const formattedTexts: string[] = [];

  for (const activation of activations) {
    const { tokens, values } = activation;
    const maxActivationIndex = values.indexOf(Math.max(...values));

    // Only get the first token after the max activating token
    if (maxActivationIndex + 1 < tokens.length) {
      const tokenAfterMaxActivating = tokens[maxActivationIndex + 1].replace('\n', '').trim();
      formattedTexts.push(tokenAfterMaxActivating);
    } else {
      // Handle case where max activation is the last token
      formattedTexts.push('');
    }
  }

  return formattedTexts.join('\n');
};

const formatMaxActivatingTokens = (activations: Activation[]): string => {
  const formattedTokens: string[] = [];

  for (const activation of activations) {
    const { tokens, values } = activation;
    const maxActivationIndex = values.indexOf(Math.max(...values));
    const maxActivatingToken = tokens[maxActivationIndex].replace('\n', '').trim();
    formattedTokens.push(maxActivatingToken);
  }

  return formattedTokens.join('\n');
};

const formatTopActivatingTexts = (activations: Activation[]): string => {
  const formattedTexts: string[] = [];

  for (const activation of activations) {
    const { tokens, values } = activation;
    const maxActivationIndex = values.indexOf(Math.max(...values));

    // Calculate window bounds
    const startIndex = Math.max(0, maxActivationIndex - TOKENS_AROUND_MAX_ACTIVATING_TOKEN);
    const endIndex = Math.min(tokens.length, maxActivationIndex + TOKENS_AROUND_MAX_ACTIVATING_TOKEN + 1);

    // Create trimmed text with max activating token
    const trimmedTokens = [
      ...tokens.slice(startIndex, maxActivationIndex),
      tokens[maxActivationIndex],
      ...tokens.slice(maxActivationIndex + 1, endIndex),
    ];

    const trimmedText = trimmedTokens.join('').replace('\n', '  ');
    formattedTexts.push(trimmedText);
  }

  return formattedTexts.join('\n');
};

const postProcessExplanation = (explanation: string | null) => {
  if (!explanation) {
    throw new Error('Explanation is null');
  }
  // Extract the explanation from the response and clean it up
  let cleanedExplanation = explanation.trim();

  // Remove trailing period if it exists
  if (cleanedExplanation.endsWith('.')) {
    cleanedExplanation = cleanedExplanation.slice(0, -1);
  }

  // Split by "Explanation: " and take the last segment if it exists
  if (cleanedExplanation.includes('Explanation: ')) {
    cleanedExplanation = cleanedExplanation.split('Explanation: ').pop() || '';
  } else if (cleanedExplanation.includes('explanation: ')) {
    cleanedExplanation = cleanedExplanation.split('explanation: ').pop() || '';
  }

  // Filter out responses that contain "method [number]" in the explanation
  // eslint-disable-next-line no-plusplus
  for (let i = 1; i <= 5; i++) {
    if (cleanedExplanation.toLowerCase().includes(`method ${i}`)) {
      console.error("Skipping output that contains 'method' in response text");
      return '';
    }
  }

  cleanedExplanation = cleanedExplanation.trim();
  if (cleanedExplanation.length === 0) {
    throw new Error('Explanation is empty');
  }

  return cleanedExplanation;
};

export const generateExplanationNpMaxActLogits = async (
  activations: Activation[],
  feature: Neuron,
  explanationModel: ExplanationModelType,
  explanationModelOpenRouterId: string | null,
  explainerModelType: AutoInterpModelType,
  explainerKeyType: UserSecretType,
  explainerKey: string,
) => {
  // Replace all ▁ characters in feature.pos_str with spaces and clean up newlines
  const processedTopPositiveLogits = feature.pos_str.map((logit) => logit.replace(/▁/g, ' ').replace(/\n/g, '').trim());

  const newMessage = `

Neuron 5

<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>

${formatTokensAfterMaxActivatingToken(activations)}

</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>


<MAX_ACTIVATING_TOKENS>

${formatMaxActivatingTokens(activations)}

</MAX_ACTIVATING_TOKENS>


<TOP_POSITIVE_LOGITS>

${processedTopPositiveLogits.join('\n')}

</TOP_POSITIVE_LOGITS>


<TOP_ACTIVATING_TEXTS>

${formatTopActivatingTexts(activations)}

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 5 behavior: `;

  console.log(newMessage);

  if (explainerModelType === AutoInterpModelType.OPENAI || explainerKeyType === UserSecretType.OPENROUTER) {
    const openai = new OpenAI({
      baseURL: explainerKeyType === UserSecretType.OPENROUTER ? OPENROUTER_BASE_URL : undefined,
      apiKey: explainerKey,
    });
    const messages = [
      makeOaiMessage('system', systemMessage),
      makeOaiMessage('user', firstUserMessage),
      makeOaiMessage('assistant', firstAssistantMessage),
      makeOaiMessage('user', secondUserMessage),
      makeOaiMessage('assistant', secondAssistantMessage),
      makeOaiMessage('user', thirdUserMessage),
      makeOaiMessage('assistant', thirdAssistantMessage),
      makeOaiMessage('user', fourthUserMessage),
      makeOaiMessage('assistant', fourthAssistantMessage),
      makeOaiMessage('user', newMessage),
    ];
    try {
      const chatCompletion = await openai.chat.completions.create({
        messages,
        model:
          explanationModelOpenRouterId && explainerKeyType === UserSecretType.OPENROUTER
            ? explanationModelOpenRouterId
            : explanationModel.name,
        max_completion_tokens: 4096,
        temperature: 1.0,
        top_p: 1.0,
      });
      const explanationString = chatCompletion.choices[0].message.content;
      console.log(explanationString);
      const cleanedExplanation = postProcessExplanation(explanationString);
      console.log(cleanedExplanation);
      return cleanedExplanation;
    } catch (error) {
      console.error(error);
      throw error;
    }
  }
  if (explainerModelType === AutoInterpModelType.ANTHROPIC) {
    const anthropic = new Anthropic({ apiKey: explainerKey });
    const messages = [
      makeAnthropicMessage('user', firstUserMessage),
      makeAnthropicMessage('assistant', firstAssistantMessage),
      makeAnthropicMessage('user', secondUserMessage),
      makeAnthropicMessage('assistant', secondAssistantMessage),
      makeAnthropicMessage('user', thirdUserMessage),
      makeAnthropicMessage('assistant', thirdAssistantMessage),
      makeAnthropicMessage('user', fourthUserMessage),
      makeAnthropicMessage('assistant', fourthAssistantMessage),
      makeAnthropicMessage('user', newMessage),
    ];
    const msg = await anthropic.messages.create({
      model: explanationModel.name,
      max_tokens: 240,
      temperature: 1.0,
      system: systemMessage,
      messages,
    });
    const explanationString = (msg.content[0] as Anthropic.TextBlock).text;
    console.log(explanationString);
    const cleanedExplanation = postProcessExplanation(explanationString);
    console.log(cleanedExplanation);
    return cleanedExplanation;
  }
  if (explainerModelType === AutoInterpModelType.GOOGLE) {
    const gemini = new GoogleGenerativeAI(explainerKey);
    const model = gemini.getGenerativeModel({
      model: explanationModel.name,
      systemInstruction: systemMessage,
    });
    const chat = model.startChat({
      history: [
        makeGeminiMessage('user', firstUserMessage),
        makeGeminiMessage('model', firstAssistantMessage),
        makeGeminiMessage('user', secondUserMessage),
        makeGeminiMessage('model', secondAssistantMessage),
        makeGeminiMessage('user', thirdUserMessage),
        makeGeminiMessage('model', thirdAssistantMessage),
        makeGeminiMessage('user', fourthUserMessage),
        makeGeminiMessage('model', fourthAssistantMessage),
      ],
    });
    const result = await chat.sendMessage(newMessage);
    const explanationString = result.response.text();
    console.log(explanationString);
    const cleanedExplanation = postProcessExplanation(explanationString);
    console.log(cleanedExplanation);
    return cleanedExplanation;
  }
  return '';
};
