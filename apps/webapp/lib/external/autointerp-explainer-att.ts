/* eslint-disable no-useless-escape */

import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Activation, ExplanationModelType, UserSecretType } from '@prisma/client';
import OpenAI from 'openai';
import { AutoInterpModelType, OPENROUTER_BASE_URL } from '../utils/autointerp';
import {
  makeAnthropicMessage,
  makeGeminiMessage,
  makeOAIattentionActivation,
  makeOaiMessage,
} from './autointerp-shared';

const ATTENTION_SEQUENCE_SEPARATOR = '<|sequence_separator|>';

const systemMessage = `We're studying attention heads in a neural network. Each head looks at every pair of tokens in a short token sequence and activates for pairs of tokens that fit what it is looking for. Attention heads always attend from a token to a token earlier in the sequence (or from a token to itself).

We will display multiple instances of sequences with the "to" token surrounded by double asterisks (e.g., **token**) and the "from" token surrounded by double square brackets (e.g., [[token]]). If a token attends from itself to itself, it will be surrounded by both (e.g., [[**token**]]).

Look at the pairs of tokens the head activates for and summarize in a single sentence what pattern the head is looking for. We do not display every activating pair of tokens in a sequence; you must generalize from limited examples.

Remember, the head always attends to tokens earlier in the sentence (marked with ** **) from tokens later in the sentence (marked with [[ ]]), except when the head attends from a token to itself (marked with [[** **]]).

The explanation takes the form: "This attention head attends to {pattern of tokens marked with ** **, which appear earlier} from {pattern of tokens marked with [[ ]], which appear later}." The explanation does not include any of the markers (** **, [[ ]]), as these are just for your reference.

Sequences are separated by ${ATTENTION_SEQUENCE_SEPARATOR}.`;

const firstUserMessage = `

Attention head 1
Activations:
dreams of a future like her biggest idol, who was also born visually impaired.

"My ultimate dream would be to sing at Carols [by Candlelight**]** and to become a famous musician like Andrea Bocelli[[ ...]] and to show people that if you have a disability it doesn't matter," she said.
<|sequence_separator|>
omes Ever Sequenced**]**[[
]]
One mystery of cat development is how cats have come to have such varied coats, from solid colours to "mackerel" tabby patterns of thin vertical stripes. The researchers were particularly interested in what turns the mackerel pattern into a "blotched" tabby pattern,
<|sequence_separator|>
, 6, 8, 4**]**[[':]]rb.sort.slice(1,2); # More advanced, this is Ruby's map and each_with_index # This shows the :rb postfix-operator sugar instead of EVAL "[1,2,3,4]":rb .map(-> $
<|sequence_separator|>
 him a WN [white nationalist**]** until there is an indication as such[[...]] The fact that he targeted a church gives me an inkling that it was religion-related," wrote WhiteVirginian.
 
 "Yep, bad news for gun rights advocates as well," wrote maththeorylover2008. "Another
 <|sequence_separator|>
 23**]**
 [[
 ]]While preparing to take the fight to Primordus, Balthazar learned about Taimi's machine and how it could supposedly kill two Elder Dragons with a single blow, which piqued his interest. This piece of news, as well as Marjory's sudden departure from his side which
 Explanation of attention head 1 behavior: this attention head"
`;

const firstAssistantMessage = ' attends to the latest closing square bracket from arbitrary subsequent tokens.';

const secondUserMessage = `

Attention head 2
Activations:
 he said. "Coming off winning the year before, I love playing links** golf**, and I love playing the week before a major. It was tough to miss it. I\'m just glad to be back."

Fowler outplayed his partners[[ Rory]] McIlroy (74) and Henrik Stenson (72
<|sequence_separator|>
 Club:

1. World renowned** golf**[[ course]]

2. Vern Morcom designed golf course

3. Great family holiday destination

4. Play amid our resident Eastern Grey kangaroo population

5. Terrific friendly staff

6. Natural picturesque bush setting

7. L
<|sequence_separator|>
615 rpm on average). As a result, each player was hitting longer drives on their best shots, while achieving a straighter ball flight that was less affected by wind.

Every** Golf**WRX Member gained yardage with a new[[ Taylor]]Made driver; the largest distance gain was an impressive +10.1 yards,
<|sequence_separator|>
 of being? Well, having perfected the art of swimming, Phelps has moved on to another cherished summer pastime –** golf**. Here he is participating in the Dunhill Links Championship at Kingsbarns in Scotland today. The[[ greens]] over there are really big, so the opportunity for 50-yard putts exist. Of
<|sequence_separator|>
OTUS is getting to see aliens.

RELATED: Barack Obama joins second D.C.-area** golf** club

"He goes, 'they're freaking crazy looking.' And then he walks up, makes his putt, turns back, walks off the[[ green]], leaves it at that and gives me a wink,"
Explanation of attention head 2 behavior: this attention head`;

const secondAssistantMessage = ' attends to the token "golf" from golf-related tokens.';

const thirdUserMessage = `

Attention head 3
Activations:
 security by requiring the user to enter a numeric code sent to** his** or[[ her]] cellphone in addition to a password. A lot of websites have offered this feature for years, but Intuit just made it widely available earlier this year.

"When you give your most sensitive data and that of your family to a company,
<|sequence_separator|>
 3 months, they separated the** men** and[[ women]] here. I don\'t know where they took the men and the children, but they took us women to Syria. They kept us in an underground prison. My only wish is that my children and husband escape ISIS. They brought us here from Raqqa, my sisters from the PKK
<|sequence_separator|>
 an emphasis on the pursuit of power despite interpersonal costs."

The study, which involved over 600 young** men** and[[ women]], makes a strong case for assessing such traits as "ruthless ambition," "discomfort with leadership" and "hubristic pride" to understand psychopathologies.

The researchers looked at
<|sequence_separator|>
 4 hours. These results, differ between** men** and[[ women]], however. We can see that although both groups have a large cluster of people at exactly 40 hours per week, there are more men reporting hours above 40, whereas there are more women reporting hours below 40. Result 3: Male Hours Worked [Info] Owner
<|sequence_separator|>
 they were perceived as more emotional, which made participants more confident in their own opinion."

Ms Salerno said both** men** and[[ women]] reacted in the same way to women expressing themselves angrily.

"Participants confidence in their own verdict dropped significantly after male holdouts expressed anger," the paper\'s findings stated.

Explanation of attention head 3 behavior: this attention head`;

const thirdAssistantMessage = ' attends to male-related tokens from paired female-related tokens.';

const fourthUserMessage = `

Attention head 4
Activations:
Vietnamese[[** Ministry**]] of Foreign Affairs spokesperson Le Hai Binh is seen in this file photo. . Tuoi Tre

The Ministry of Foreign Affairs has ordered a thorough investigation into a case in which a Vietnamese fisherman was shot dead on his boat in Vietnam\'s Truong Sa (Sprat
<|sequence_separator|>
 J[[**okin**]]en tells a much different story. He almost sounded like a pitchman.

"All the staff, team service guys, all the trainers, they're unbelievable guys," said Jokinen. "It's not just the players, it's the staff around the team. I feel really bad for them
<|sequence_separator|>
 a[[** Pv**]]E game, we probably would use it but based on the tests we've run on it, that wouldn't be our first choice for a live RvR game. Now, could we use it for prototyping? Yep, we are already doing that. Second, as to other engines there are both financial
<|sequence_separator|>
-[[**tun**]]er is also customisable for hassle-free experimentation. The Strix X399-E Gaming takes up to three double-wide cards in SLI or CrossFireX. Primary graphics slots are protected by SafeSlot from damages that heavy GPU coolers can potentially cause.

Personalised RGB lighting is made possible
<|sequence_separator|>
 to[[** abolish**]] such a complex? Are there ways vegans can eat more sustainably? What are some of the health challenges for new vegans, and how can we raise awareness of these issues so that, for instance, medical professionals are more supportive of veganism?

Moreover, it is essential that vegans differentiate
Explanation of attention head 4 behavior: this attention head`;

const fourthAssistantMessage = ' attends from the second token in the sequence to the second token in the sequence.';

export const generateExplanationOaiAttentionHead = async (
  activations: Activation[],
  explanationModel: ExplanationModelType,
  explanationModelOpenRouterId: string | null,
  explainerModelType: AutoInterpModelType,
  explainerKeyType: UserSecretType,
  explainerKey: string,
) => {
  let newMessage = `

Attention head 5
Activations:`;

  // eslint-disable-next-line
  for (let i = 0; i < activations.length; i++) {
    newMessage += makeOAIattentionActivation(activations[i]);
    if (i < activations.length - 1) {
      newMessage = `${newMessage}\n${ATTENTION_SEQUENCE_SEPARATOR}`;
    }
  }
  newMessage += `
Explanation of attention head 5 behavior: this attention head`;
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
          (explainerKeyType === UserSecretType.OPENROUTER ? explanationModelOpenRouterId : explanationModel.name) || '',
        max_tokens: 80,
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
      makeAnthropicMessage('user', fourthUserMessage),
      makeAnthropicMessage('assistant', fourthAssistantMessage),
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
        makeGeminiMessage('user', fourthUserMessage),
        makeGeminiMessage('model', fourthAssistantMessage),
      ],
    });
    const result = await chat.sendMessage(newMessage);
    const explanationString = result.response.text();
    console.log(explanationString);
    return explanationString;
  }
  return '';
};
