import { getModelById } from '@/lib/db/model';
import { getVectorsForModelAndUser } from '@/lib/db/neuron';
import { FeaturePreset, SteerFeature, SteerPreset } from '@/lib/utils/steer';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';

const steerPresetsSchema = object({
  modelId: string().required(),
});

const PROMPT_PRESETS: { [key: string]: { name: string; prompt: string }[] } = {
  'gemma-2-2b': [
    {
      name: 'Most Iconic Structure',
      prompt: 'The most iconic structure on Earth is',
    },
    {
      name: 'Violent Criminal',
      prompt: 'The violent criminal broke into the home and',
    },
    {
      name: 'Roses Poem',
      prompt: 'Roses are red, violets are blue,',
    },
    {
      name: 'Grew Up Watching',
      prompt: 'I grew up watching',
    },
    {
      name: 'Favorite Country',
      prompt: 'My favorite country to visit is',
    },
    {
      name: 'Buy Phone',
      prompt: 'If I were to buy a phone today, I would buy',
    },
    {
      name: 'Turn On Radio',
      prompt: 'Every time I turn on the radio, I hear',
    },
    {
      name: 'What am I thinking?',
      prompt: "What am I thinking about? I'm thinking about",
    },
    {
      name: 'Favorite Animal',
      prompt: 'My favorite animal is',
    },
  ],
  'gemma-2-9b-it': [],
};

export const FEATURE_PRESETS: {
  [key: string]: FeaturePreset[];
} = {
  'llama3.3-70b-it': [
    {
      name: '🧢 Assistant Cap',
      features: [
        {
          modelId: 'llama3.3-70b-it',
          layer: '32-neuronpedia-resid-post',
          index: 95134726,
          explanation: 'Assistant Cap L32',
          strength: -1.234375,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '33-neuronpedia-resid-post',
          index: 155044423,
          explanation: 'Assistant Cap L33',
          strength: -1.5703125,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '34-neuronpedia-resid-post',
          index: 165437805,
          explanation: 'Assistant Cap L34',
          strength: -2.21875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '35-neuronpedia-resid-post',
          index: 110286670,
          explanation: 'Assistant Cap L35',
          strength: -1.703125,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '36-neuronpedia-resid-post',
          index: 156321408,
          explanation: 'Assistant Cap L36',
          strength: -1.796875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '37-neuronpedia-resid-post',
          index: 159476403,
          explanation: 'Assistant Cap L37',
          strength: -1.7421875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '38-neuronpedia-resid-post',
          index: 123496291,
          explanation: 'Assistant Cap L38',
          strength: -1.765625,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '39-neuronpedia-resid-post',
          index: 120194034,
          explanation: 'Assistant Cap L39',
          strength: -1.5078125,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '40-neuronpedia-resid-post',
          index: 141497702,
          explanation: 'Assistant Cap L40',
          strength: -1.875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '41-neuronpedia-resid-post',
          index: 127083163,
          explanation: 'Assistant Cap L41',
          strength: -1.6640625,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '42-neuronpedia-resid-post',
          index: 92139530,
          explanation: 'Assistant Cap L42',
          strength: -1.6015625,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '43-neuronpedia-resid-post',
          index: 132977111,
          explanation: 'Assistant Cap L43',
          strength: -1.46875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '44-neuronpedia-resid-post',
          index: 136675469,
          explanation: 'Assistant Cap L44',
          strength: -1.6875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '45-neuronpedia-resid-post',
          index: 124473324,
          explanation: 'Assistant Cap L45',
          strength: -1.875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '46-neuronpedia-resid-post',
          index: 115093673,
          explanation: 'Assistant Cap L46',
          strength: -1.8984375,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '47-neuronpedia-resid-post',
          index: 142618817,
          explanation: 'Assistant Cap L47',
          strength: -1.859375,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '48-neuronpedia-resid-post',
          index: 136568392,
          explanation: 'Assistant Cap L48',
          strength: -1.9140625,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '49-neuronpedia-resid-post',
          index: 123090249,
          explanation: 'Assistant Cap L49',
          strength: -1.9140625,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '50-neuronpedia-resid-post',
          index: 165306750,
          explanation: 'Assistant Cap L50',
          strength: -1.8203125,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '51-neuronpedia-resid-post',
          index: 133555003,
          explanation: 'Assistant Cap L51',
          strength: -1.921875,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '52-neuronpedia-resid-post',
          index: 122170101,
          explanation: 'Assistant Cap L52',
          strength: -1.9765625,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '53-neuronpedia-resid-post',
          index: 124671970,
          explanation: 'Assistant Cap L53',
          strength: -1.828125,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '54-neuronpedia-resid-post',
          index: 118427494,
          explanation: 'Assistant Cap L54',
          strength: -1.59375,
        },
        {
          modelId: 'llama3.3-70b-it',
          layer: '55-neuronpedia-resid-post',
          index: 161912969,
          explanation: 'Assistant Cap L55',
          strength: -1.984375,
        },
      ],
      steerMethod: NPSteerMethod.ProjectionCap,
      alias: 'assistant-cap',
    },
  ],
  'deepseek-r1-distill-llama-8b': [
    {
      name: '🌉 San Francisco',
      features: [
        {
          modelId: 'deepseek-r1-distill-llama-8b',
          layer: '15-llamascope-slimpj-res-32k',
          index: 30274,
          explanation: 'references to San Francisco',
          strength: 2.25,
        },
      ],
      exampleSteerOutputId: 'cm7nv3vd6000jwp7wbre7gf00',
    },
    {
      name: '⏰ Anticipation / Excitement',
      features: [
        {
          modelId: 'deepseek-r1-distill-llama-8b',
          layer: '15-llamascope-slimpj-res-32k',
          index: 24910,
          explanation: 'references to anticipation and excitement',
          strength: 4,
        },
      ],
      exampleSteerOutputId: 'cm7lgjlj1001f14b4efzoy963',
    },
    {
      name: '🍞 Bread / Pastries',
      features: [
        {
          modelId: 'deepseek-r1-distill-llama-8b',
          layer: '15-llamascope-slimpj-res-32k',
          index: 8381,
          explanation: 'references to bread and pastries',
          strength: 4,
        },
      ],
      exampleSteerOutputId: 'cm7k66d1o00056wj0vyuzk5ia',
    },
    {
      name: '🤔 Overthinking (eg "Wait"/"Hmm")',
      features: [
        {
          modelId: 'deepseek-r1-distill-llama-8b',
          layer: '15-llamascope-slimpj-res-32k',
          index: 30939,
          explanation: 'references to "Wait, / Oh! / Similar"',
          strength: 5,
        },
      ],
      exampleSteerOutputId: 'cm7nues460001aoew15fapz86', // cm7l3zvtb0063y6zk7x97la8u // 1+1 cm7l3shv4005ly6zkkbgp5e56  cm7l3tuvg005py6zkrw4gfxxk
    },
    {
      name: '🐶 Dogs',
      features: [
        {
          modelId: 'deepseek-r1-distill-llama-8b',
          layer: '15-llamascope-slimpj-res-32k',
          index: 30288,
          explanation: 'references to dogs and their roles in human lives',
          strength: 4,
        },
      ],
      exampleSteerOutputId: 'cm7k5bcgz0005xfq5txqpd3hu',
    },
  ],
  'gpt2-small': [
    {
      name: '🐈 Cats',
      features: [
        {
          modelId: 'gpt2-small',
          layer: '10-res-jb',
          index: 16899,
          explanation: 'mentions of cats or similar words related to cats',
          strength: 40,
        },
      ],
      exampleSteerOutputId: 'cm7cjeodg002t1q958i4y0swb',
    },
    {
      name: '🌉 Golden State',
      features: [
        {
          modelId: 'gpt2-small',
          layer: '10-res-jb',
          index: 19837,
          explanation: 'proper nouns related to a specific entity named "Golden State"',
          strength: 40,
        },
      ],
      exampleSteerOutputId: 'cm7cjqcj3006x1q95bpwjn1hs',
    },
    {
      name: '🍏 Apple',
      features: [
        {
          modelId: 'gpt2-small',
          layer: '10-res-jb',
          index: 4269,
          explanation: 'references to the company Apple',
          strength: 75,
        },
      ],
      exampleSteerOutputId: 'cm7cjtmtp007h1q95yn36dgth',
    },
    {
      name: '⚡ Thunder & Lightning',
      features: [
        {
          modelId: 'gpt2-small',
          layer: '4-res-jb',
          index: 23123,
          explanation: 'words related to severe weather phenomena, particularly thunderstorms and lightning',
          strength: 27,
        },
      ],
      exampleSteerOutputId: 'cm7ck5bzy00bx1q9511n5tw5d',
    },
  ],
  'gemma-2-2b': [
    {
      name: '🐶 Dogs',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 12082,
          explanation: 'references to dogs',
          strength: 150,
        },
      ],
      exampleSteerOutputId: 'cm7ck9pq000ch1q95n780rvji',
    },
    {
      name: '🇬🇧 British English',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 2292,
          explanation: 'variations of British English',
          strength: 24,
        },
      ],
      exampleSteerOutputId: 'cm7ckd57i00d71q959f9cix7e',
    },
    {
      name: '🇫🇷 French',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 12332,
          explanation: 'French text',
          strength: 100,
        },
      ],
      exampleSteerOutputId: 'cm7ckqzf100e51q95v0u9u6w3',
    },
    {
      name: '📢 YELLING',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 11859,
          explanation: 'uppercase text',
          strength: 80,
        },
      ],
      exampleSteerOutputId: 'cm7cksiwu00eb1q95f8nq3mju',
    },
  ],
  'gemma-2-2b-it': [
    {
      name: '🐶 Dogs',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 12082,
          explanation: 'references to dogs',
          strength: 276,
        },
      ],
      exampleSteerOutputId: 'cm7cmmqlq00h31q95oupy3sbj',
    },
    {
      name: '🇫🇷 French',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 12332,
          explanation: 'French text',
          strength: 122,
        },
      ],
      exampleSteerOutputId: 'cm7cn4ade00hf1q95rww15w8h',
    },
    {
      name: '🚫 Refusal',
      features: [
        {
          modelId: 'gemma-2-2b-it',
          layer: '15-neuronpedia-resid-pre',
          index: 0,
          explanation: 'refusal (Arditi et al. 2024)',
          strength: 1,
          hasVector: true,
        },
      ],
      exampleSteerOutputId: 'cm7cp4um700jp1q95z258r6y7',
    },
    {
      name: '📢 YELLING',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 11859,
          explanation: 'uppercase text',
          strength: 80,
        },
      ],
      exampleSteerOutputId: 'cm7cnp28000hl1q95hhay24au',
    },
    {
      name: '🙌 Bravery',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 14377,
          explanation: 'bravery',
          strength: 168,
        },
      ],
      exampleSteerOutputId: 'cm7cnppa200hp1q95ifdaj3pu',
    },
    {
      name: '🇬🇧 British English',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 2292,
          explanation: 'variations of British English',
          strength: 80,
        },
      ],
      exampleSteerOutputId: 'cm7cnquyv00ht1q95nu23qsyr',
    },
    {
      name: '🥴 Misspellings',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 8408,
          explanation: 'misspellings',
          strength: 240,
        },
      ],
      exampleSteerOutputId: 'cm7cnrv9h00i11q95izw9lec3',
    },
    {
      name: '🇪🇸 Spanish',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 8590,
          explanation: 'the Spanish language',
          strength: 80,
        },
      ],
      exampleSteerOutputId: 'cm7cnshp300i51q95uvmq8w4l',
    },
    {
      name: '🌉 San Francisco',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 3124,
          explanation: 'references to San Francisco',
          strength: 154,
        },
      ],
      exampleSteerOutputId: 'cm7cnt3d000i91q95lcl63ela',
    },
    {
      name: '😄 Humor',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 1555,
          explanation: 'humorous contexts',
          strength: 156,
        },
      ],
      exampleSteerOutputId: 'cm7cntunu00id1q95c5khc3t6',
    },
    {
      name: '😬 Cringe',
      features: [
        {
          modelId: 'gemma-2-2b',
          layer: '20-gemmascope-res-16k',
          index: 13710,
          explanation: 'poor storytelling',
          strength: 218,
        },
      ],
      exampleSteerOutputId: 'cm7cnv0ec00ih1q95nkumy3qr',
    },
  ],
  'gemma-2-9b-it': [
    {
      name: '🐱 Cats',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '9-gemmascope-res-131k',
          index: 62610,
          explanation: 'mentions and references to cats and related topics',
          strength: 192,
        },
      ],
      exampleSteerOutputId: 'cm7cp63af00jx1q952neqg6e5',
    },
    {
      name: '㊗️ Chinese',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '9-gemmascope-res-131k',
          index: 121465,
          explanation:
            'Chinese characters and phrases, particularly those related to technical or programming terminology',
          strength: 74,
        },
      ],
      exampleSteerOutputId: 'cm7cp6kc200k11q956c61e232',
    },
    {
      name: '🏴‍☠️ Pirate',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '31-gemmascope-res-131k',
          index: 77558,
          explanation: 'references to pirates and related themes',
          strength: 66,
        },
        {
          modelId: 'gemma-2-9b-it',
          layer: '9-gemmascope-res-131k',
          index: 29917,
          explanation: 'references to pirates and pirate-related themes',
          strength: 166,
        },
      ],
      exampleSteerOutputId: 'cm7cp78b900k51q95qdn01q1l',
    },
    {
      name: '🪶 Shakespeare',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 57285,
          explanation: "famous quotes or phrases from Shakespeare's works",
          strength: 226,
        },
      ],
      exampleSteerOutputId: 'cm7cp8rl600kh1q95oond70rg',
    },
    {
      name: '🍂 Poetry',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 80360,
          explanation: 'rhyming words or phrases at the end of lines in poetic or lyrical text',
          strength: 202,
        },
      ],
      exampleSteerOutputId: 'cm7cp9it600kj1q95pgz7k57s',
    },
    {
      name: '🌁 San Francisco',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 116871,
          explanation: 'references to San Francisco and its landmarks',
          strength: 200,
        },
      ],
      exampleSteerOutputId: 'cm7cpbktq00kr1q95l6y40rtk',
    },
    {
      name: '😍 Positivity',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 111712,
          explanation: 'expressions of positive sentiment and appreciation',
          strength: 160,
        },
      ],
      exampleSteerOutputId: 'cm7cpcu0p00kv1q95qhgcfrvu',
    },
    {
      name: '👎 Negativity',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 120550,
          explanation: 'descriptions of negative or distressing situations',
          strength: 112,
        },
      ],
      exampleSteerOutputId: 'cm7cpdm3800kz1q95ah5nnydk',
    },
    {
      name: '🎵 Music',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 61962,
          explanation: 'instances and descriptions of music and audio-related experiences',
          strength: 170.5,
        },
      ],
      exampleSteerOutputId: 'cm7cph1e100ln1q954pc0tmlo',
    },
    {
      name: '🇬🇧 British English',
      features: [
        {
          modelId: 'gemma-2-9b-it',
          layer: '20-gemmascope-res-131k',
          index: 90098,
          explanation: 'British English*',
          strength: 60,
        },
      ],
      exampleSteerOutputId: 'cm7cphz5700lv1q950urz9h81',
    },
  ],
  'gemma-2-9b': [],
  'gpt-oss-20b': [
    {
      name: '🎭 Jokes and Puns',
      features: [
        {
          modelId: 'gpt-oss-20b',
          layer: '15-resid-post-aa',
          index: 37788,
          explanation: 'jokes and puns',
          strength: 2000,
        },
      ],
      exampleSteerOutputId: 'cmjt45nub001n7gd3tx4h3gfu',
    },
  ],
  'llama3.1-8b': [
    {
      name: '🐱 Cats',
      features: [
        {
          modelId: 'llama3.1-8b',
          layer: '28-llamascope-res-32k',
          index: 12514,
          explanation: 'mentions of cats and related stories',
          strength: 16.25,
        },
      ],
      exampleSteerOutputId: 'cm7cqjlae00nr1q95gwntzh7t',
    },
    {
      name: '🧑‍💻 CEOs',
      features: [
        {
          modelId: 'llama3.1-8b',
          layer: '26-llamascope-res-32k',
          index: 13427,
          explanation: 'references to economic theories and capitalist dynamics',
          strength: 15,
        },
      ],
      exampleSteerOutputId: 'cm7cqldx600nz1q95tph6jhr4',
    },
    {
      name: '🎶 Poetry',
      features: [
        {
          modelId: 'llama3.1-8b',
          layer: '24-llamascope-res-32k',
          index: 20434,
          explanation: 'elements of poetry and rhythmic language',
          strength: 25,
        },
      ],
      exampleSteerOutputId: 'cm7cqpgik00op1q95y5lukh3r',
    },
  ],
  'llama3.1-8b-it': [
    {
      name: '🤡 Nonsense/Absurdity',
      features: [
        {
          modelId: 'llama3.1-8b-it',
          layer: '11-resid-post-aa',
          index: 110281,
          explanation: 'nonsense/absurdity',
          strength: 7.9,
        },
      ],
      exampleSteerOutputId: 'cmfgcgpd90027kenxaxva1w2k',
    },
    {
      name: '◀️ Opposite/Disobedience',
      features: [
        {
          modelId: 'llama3.1-8b-it',
          layer: '11-resid-post-aa',
          index: 33274,
          explanation: 'opposite/disobedience',
          strength: 7.1,
        },
      ],
      exampleSteerOutputId: 'cmfgc8btg0017kenx4zsszyuq',
    },
    {
      name: '😈➡️😇 Evil Persona (Reverse)',
      features: [
        {
          modelId: 'llama3.1-8b-it',
          layer: '11-resid-post-aa',
          index: 87027,
          explanation: 'evil/jailbroken persona',
          strength: -3.5,
        },
      ],
      exampleSteerOutputId: 'cmfj115t6001dwwl29cexrixq',
    },
    {
      name: '👹 Manipulation',
      features: [
        {
          modelId: 'llama3.1-8b-it',
          layer: '11-resid-post-aa',
          index: 16205,
          explanation: 'psychological manipulation',
          strength: 7,
        },
      ],
      exampleSteerOutputId: 'cmfgc48qk000zkenx5a8lzlvq',
    },
  ],
};

const DEFAULT_SELECTED_FEATURES: {
  [key: string]: SteerFeature[];
} = {
  'gemma-2-2b': [
    {
      modelId: 'gemma-2-2b',
      layer: '20-gemmascope-res-16k',
      index: 12082,
      explanation: 'references to dogs',
      strength: -50,
    },
  ],
  'gemma-2-9b-it': [],
};

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();

  try {
    const body = await steerPresetsSchema.validate(bodyJson);
    const { modelId } = body;

    const modelAccess = await getModelById(modelId, request.user);
    if (!modelAccess) {
      return NextResponse.json({ message: 'Model Not Found' }, { status: 404 });
    }

    const featurePresetsToReturn: FeaturePreset[] = [...(FEATURE_PRESETS[modelId] || [])];

    // if user is logged in, add their vectors to the feature presets
    if (request.user) {
      const vectors = await getVectorsForModelAndUser(modelId, request.user.id);
      // eslint-disable-next-line no-restricted-syntax
      for (const vector of vectors) {
        const feature = {
          modelId: vector.modelId,
          layer: vector.layer,
          index: parseInt(vector.index, 10),
          explanation: vector.vectorLabel || '',
          strength: vector.vectorDefaultSteerStrength || 10,
          hasVector: true,
        };
        featurePresetsToReturn.unshift({
          name: vector.vectorLabel || '',
          features: [feature],
          isUserVector: true,
        });
      }
    }

    const toReturn: SteerPreset = {
      model: modelAccess,
      defaultPrompt: '',
      promptPresets: PROMPT_PRESETS[modelId],
      featurePresets: featurePresetsToReturn,
      defaultSelectedFeatures: DEFAULT_SELECTED_FEATURES[modelId],
    };

    return NextResponse.json(toReturn);
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
