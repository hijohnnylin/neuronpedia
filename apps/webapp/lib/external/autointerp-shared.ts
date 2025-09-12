import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import { MessageParam } from '@anthropic-ai/sdk/resources/messages.mjs';
import { Content } from '@google/generative-ai';
import { Activation } from '@prisma/client';
import { ChatCompletionMessageParam } from 'openai/resources/index.mjs';

export function makeOaiMessage(role: 'user' | 'system' | 'assistant', content: string): ChatCompletionMessageParam {
  return {
    role,
    content,
  };
}

export function makeAnthropicMessage(role: 'user' | 'assistant', content: string): MessageParam {
  return {
    role,
    content: [
      {
        type: 'text',
        text: content,
      },
    ],
  };
}

export function makeGeminiMessage(role: 'user' | 'model', content: string): Content {
  return {
    role,
    parts: [
      {
        text: content,
      },
    ],
  };
}

export function makeOAIactivation(activation: Activation) {
  let str = `
  <start>`;
  activation.tokens.forEach((token, tokenIndex) => {
    const tokenCleaned = replaceHtmlAnomalies(token, false);
    str += `\n${tokenCleaned}\t${activation.values[tokenIndex]}`;
  });
  str += '\n<end>';
  return str;
}

export function makeOAIattentionActivation(activation: Activation) {
  const dfaMaxIndex = activation.dfaValues?.indexOf(Math.max(...activation.dfaValues)) || 0;

  let str = '\n';
  activation.tokens.forEach((token, tokenIndex) => {
    const tokenCleaned = replaceHtmlAnomalies(token, false);
    if (tokenIndex === dfaMaxIndex) {
      str += `**${tokenCleaned}**`;
    } else if (tokenIndex === activation.dfaTargetIndex) {
      str += `[[${tokenCleaned}]]`;
    } else {
      str += tokenCleaned;
    }
  });
  return str;
}
