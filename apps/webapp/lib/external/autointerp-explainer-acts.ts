import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Activation, ExplanationModelType, Neuron, UserSecretType } from '@prisma/client';
import OpenAI from 'openai';
import { OPENROUTER_BASE_URL } from '../constants';
import { AutoInterpModelType } from '../utils/autointerp';
import { makeAnthropicMessage, makeGeminiMessage, makeOaiMessage } from './autointerp-shared';

const TOKENS_AROUND_MAX_ACTIVATING_TOKEN = 24;

const systemMessage = `You are explaining the behavior of a neuron in a neural network. Your response should be a very concise explanation (1-6 words) that captures what the neuron detects or predicts by finding patterns in lists.

To determine the explanation, you are given two lists:

- MAX_ACTIVATING_TOKENS, which are the top activating tokens in the top activating texts.
- TOP_ACTIVATING_TEXTS, which are top activating texts.

You should look for a pattern by trying the following methods in order. Once you find a pattern, stop and return that pattern. Do not proceed to the later methods.
Method 1: Look at MAX_ACTIVATING_TOKENS. If they share something specific in common, or are all the same token or a variation of the same token (like different cases or conjugations), respond with that token.
Method 2: Look at TOP_ACTIVATING_TEXTS and make a best guess by describing the broad theme or context, ignoring the max activating tokens.

Rules:
- Keep your explanation extremely concise (1-6 words, mostly 1-3 words).
- Do not add unnecessary phrases like "words related to", "concepts related to", or "variations of the word".
- Do not mention "tokens" or "patterns" in your explanation.
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".
- If you absolutely cannot make any guesses, return the first token in MAX_ACTIVATING_TOKENS.

Respond by going through each method number until you find one that helps you find an explanation for what this neuron is detecting or predicting. If a method does not help you find an explanation, briefly explain why it does not, then go on to the next method. Finally, end your response with the method number you used, the reason for your explanation, and then the explanation.`;

const firstUserMessage = `

Neuron 1

<MAX_ACTIVATING_TOKENS>

banana
blueberries

</MAX_ACTIVATING_TOKENS>


<TOP_ACTIVATING_TEXTS>

The apple and banana are delicious foods that provide essential vitamins and nutrients.
I enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 1 behavior: `;

const firstAssistantMessage = `Method 1 succeeds: All MAX_ACTIVATING_TOKENS (banana, blueberries) are fruits.
Explanation: fruits`;

const secondUserMessage = `

Neuron 2

<MAX_ACTIVATING_TOKENS>

and
And

</MAX_ACTIVATING_TOKENS>


<TOP_ACTIVATING_TEXTS>

It was a beautiful day outside with clear skies and warm sunshine.
And the garden has roses and tulips and daisies and sunflowers blooming together.

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 2 behavior: `;

const secondAssistantMessage = `Method 1 succeeds: All MAX_ACTIVATING_TOKENS are the word "and".
Explanation: and`;

const thirdUserMessage = `

Neuron 3

<MAX_ACTIVATING_TOKENS>

class
teaches

</MAX_ACTIVATING_TOKENS>


<TOP_ACTIVATING_TEXTS>

the civil war was a major topic in history class .
 her professor teaches psychology courses and is a tough grader .

</TOP_ACTIVATING_TEXTS>


Explanation of neuron 3 behavior: `;

const thirdAssistantMessage = `Method 1 fails: MAX_ACTIVATING_TOKENS (class, teaches) are not all the same token.
Method 2 succeeds: The TOP_ACTIVATING_TEXTS are broadly about education.
Explanation: education`;

// // It doesn't seem to work as well when you give a 'fallback' example like this one - the model is too tempted to always use the fallback. So we comment it out.
// const fourthUserMessage = `

// Neuron 4

// <MAX_ACTIVATING_TOKENS>

// apple
// launch

// </MAX_ACTIVATING_TOKENS>

// <TOP_ACTIVATING_TEXTS>

// the bright red apple fell from the tree and rolled down the hill.
//  the rocket launch was scheduled for midnight but weather conditions caused a delay.

// </TOP_ACTIVATING_TEXTS>

// Explanation of neuron 4 behavior: `;

// const fourthAssistantMessage = `
// Method 1 fails: MAX_ACTIVATING_TOKENS (apple, launch) are not all the same token.
// Method 2 fails: The TOP_ACTIVATING_TEXTS are about completely different topics (fruit falling vs space launch) with no clear connection.
// Since both fail, I'll return the first token in MAX_ACTIVATING_TOKENS.
// Explanation: apple`;

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

export const generateExplanationNpMaxAct = async (
  activations: Activation[],
  feature: Neuron,
  explanationModel: ExplanationModelType,
  explanationModelOpenRouterId: string | null,
  explainerModelType: AutoInterpModelType,
  explainerKeyType: UserSecretType,
  explainerKey: string,
) => {
  const newMessage = `

Neuron 4

<MAX_ACTIVATING_TOKENS>

${formatMaxActivatingTokens(activations)}

</MAX_ACTIVATING_TOKENS>


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
      // makeOaiMessage('user', fourthUserMessage),
      // makeOaiMessage('assistant', fourthAssistantMessage),
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
      // console.log(
      //   `Tokens used - prompt: ${chatCompletion.usage?.prompt_tokens}, completion: ${chatCompletion.usage?.completion_tokens}, total: ${chatCompletion.usage?.total_tokens}`,
      // );
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
      // makeAnthropicMessage('user', fourthUserMessage),
      // makeAnthropicMessage('assistant', fourthAssistantMessage),
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
        // makeGeminiMessage('user', fourthUserMessage),
        // makeGeminiMessage('model', fourthAssistantMessage),
        makeGeminiMessage('user', newMessage),
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
