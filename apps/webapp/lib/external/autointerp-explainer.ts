import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Activation, ExplanationModelType, UserSecretType } from '@prisma/client';
import OpenAI from 'openai';
import { AutoInterpModelType, isReasoningModel, OPENROUTER_BASE_URL } from '../utils/autointerp';
import { makeAnthropicMessage, makeGeminiMessage, makeOAIactivation, makeOaiMessage } from './autointerp-shared';

const systemMessage = `We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.`;

const firstUserMessage = `

Neuron 1
Activations:
<start>
t	0
urt	0
ur	0
ro	0
 is	0
 fab	0
ulously	0
 funny	0
 and	0
 over	0
 the	0
 top	0
 as	0
 a	0
 '	0
very	0
 sneaky	0
'	1
 but	0
ler	0
 who	0
 excel	0
s	0
 in	0
 the	0
 art	0
 of	0
 impossible	0
 disappearing	6
/	0
re	0
app	0
earing	10
 acts	0
<end>
<start>
esc	0
aping	9
 the	4
 studio	0
 ,	0
 pic	0
col	0
i	0
 is	0
 warm	0
ly	0
 affecting	3
 and	0
 so	0
 is	0
 this	0
 ad	0
roit	0
ly	0
 minimalist	0
 movie	0
 .	0
<end>

Same activations, but with all zeros filtered out:
<start>
'	1
 disappearing	6
earing	10
<end>
<start>
aping	9
 the	4
 affecting	3
<end>

Explanation of neuron 1 behavior: the main thing this neuron does is find`;

const firstAssistantMessage = "present tense verbs ending in 'ing'.";

const secondUserMessage = `

Neuron 2
Activations:
<start>
as	0
 sac	0
char	0
ine	0
 movies	0
 go	0
 ,	0
 this	0
 is	0
 likely	0
 to	0
 cause	0
 massive	0
 cardiac	0
 arrest	10
 if	0
 taken	0
 in	0
 large	0
 doses	0
 .	0
<end>
<start>
shot	0
 perhaps	0
 '	0
art	0
istically	0
'	0
 with	0
 handheld	0
 cameras	0
 and	0
 apparently	0
 no	0
 movie	0
 lights	0
 by	0
 jo	0
aquin	0
 b	0
aca	0
-	0
as	0
ay	0
 ,	0
 the	0
 low	0
-	0
budget	0
 production	0
 swings	0
 annoy	0
ingly	0
 between	0
 vert	0
igo	9
 and	0
 opacity	0
 .	0
<end>

Same activations, but with all zeros filtered out:
<start>
 arrest	10
<end>
<start>
igo	9
<end>

Explanation of neuron 2 behavior: the main thing this neuron does is find`;

const secondAssistantMessage = 'words related to physical medical conditions.';

const thirdUserMessage = `

Neuron 3
Activations:
<start>
the	0
 sense	0
 of	0
 together	3
ness	7
 in	0
 our	0
 town	1
 is	0
 strong	0
 .	0
<end>
<start>
a	0
 buoy	0
ant	0
 romantic	0
 comedy	0
 about	0
 friendship	0
 ,	0
 love	0
 ,	0
 and	0
 the	0
 truth	0
 that	0
 we	2
're	4
 all	3
 in	7
 this	10
 together	5
 .	0
<end>

Explanation of neuron 3 behavior: the main thing this neuron does is find`;

const thirdAssistantMessage = 'phrases related to community.';

export const generateExplanationOaiTokenActivationPair = async (
  activations: Activation[],
  explanationModel: ExplanationModelType,
  explanationModelOpenRouterId: string | null,
  explainerModelType: AutoInterpModelType,
  explainerKeyType: UserSecretType,
  explainerKey: string,
) => {
  let newMessage = `
        
Neuron 4
Activations:`;

  // eslint-disable-next-line no-restricted-syntax
  for (const activation of activations) {
    newMessage += makeOAIactivation(activation);
  }
  newMessage += `
${
  isReasoningModel(explanationModel.name)
    ? 'Only respond with the explanation itself, which should not be a full sentence, just the completion of "the main thing..." sentence. Do NOT include the whole phrase "Explanation of neuron 4 behavior: the main thing this neuron does is find...". Do not mention "this neuron...".'
    : ''
}
Explanation of neuron 4 behavior: the main thing this neuron does is find`;

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
      return explanationString;
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
      makeAnthropicMessage('user', newMessage),
    ];
    const msg = await anthropic.messages.create({
      model: explanationModel.name,
      max_tokens: 60,
      temperature: 1.0,
      system: systemMessage,
      messages,
    });
    const explanationString = (msg.content[0] as Anthropic.TextBlock).text;
    console.log(explanationString);
    return explanationString;
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
      ],
    });
    const result = await chat.sendMessage(newMessage);
    const explanationString = result.response.text();
    console.log(explanationString);
    return explanationString;
  }
  return '';
};
