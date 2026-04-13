// file needs a lot of cleanup

/* eslint-disable no-var */

export const BOS_TOKENS = ['<bos>', '<|endoftext|>', '<|begin_of_text|>'];

export const LINE_BREAK_REPLACEMENT_CHAR = '↵';

export const ACTIVATION_PRECISION = 4;

export const ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS = [
  { text: 'Stacked', size: 5 },
  { text: 'Snippet', size: 16 },
  { text: 'Full', size: 200 },
  // { text: "Max", size: 100000 },
];

export const HTML_ANOMALIES: { [string: string]: string } = {
  âĢĶ: '—',
  âĢĵ: '–',
  âĢľ: '“',
  âĢĿ: '”',
  âĢĺ: '‘',
  âĢĻ: '’',
  âĢĭ: ' ', // todo: this is actually zero width space
  Ġ: ' ',
  Ċ: LINE_BREAK_REPLACEMENT_CHAR,
  '<0x0A>': LINE_BREAK_REPLACEMENT_CHAR,
  '\n': LINE_BREAK_REPLACEMENT_CHAR,
  ĉ: '\t',
  '▁': ' ',
};

export const HTML_ANOMALIES_NO_LINE_BREAK_CHAR: { [string: string]: string } = {
  âĢĶ: '—', // em dash
  âĢĵ: '–', // en dash
  âĢľ: '“', // left double curly quote
  âĢĿ: '”', // right double curly quote
  âĢĺ: '‘', // left single curly quote
  âĢĻ: '’', // right single curly quote
  âĢĭ: ' ', // zero width space
  Ġ: ' ', // space
  Ċ: '\n', // line break
  '<0x0A>': '\n',
  ĉ: '\t', // tab
  '▁': ' ',
};

export const makeActivationBackgroundColorWithDFA = (
  overallMaxValue: number,
  value: number,
  rgb: string = '52, 211, 153', // emerald-400 #34d399
  dfaValue?: number | undefined,
  dfaMaxValue?: number | undefined,
) => {
  const MINIMUM_OPACITY = 0.05;
  // const MIN_RATIO_TO_HIDE_DFA = 0.1; // if dfa is 10% of max DFA, hide it
  const MINIMUM_THRESHOLD = 0.00005; // if DFA is lower than 0, hide it

  // make the act color for bottom of gradient
  const realMax = overallMaxValue;
  const realCurrent = value;
  const ratio = realCurrent / realMax;
  if (ratio === 0 || value <= MINIMUM_THRESHOLD) {
    var bottomOpacity = 0;
  } else {
    const scale = 1 - MINIMUM_OPACITY; // Maximum scale adjustment to reach 1

    // Adjust value to start scaling from 0.3 upwards
    // We use a logarithmic function to scale the value.
    // The constants 9 and 0.1 are chosen to make the function start at 0.3 for very small values
    // and approach 1 as the value approaches 1.
    var opacity = MINIMUM_OPACITY + (Math.log10(1 + 9 * ratio) * scale) / Math.log10(10);
    // Ensure the opacity does not exceed 1
    bottomOpacity = Math.min(opacity, 1);
    bottomOpacity = Math.max(0, opacity);
  }
  let bottomGradient = `rgba(${rgb}, ${bottomOpacity})`;
  let topGradient = bottomGradient;

  // if dfa value, then use it as top of gradient
  if (dfaValue && dfaMaxValue !== undefined && dfaValue > MINIMUM_THRESHOLD) {
    rgb = '251, 146, 60'; // orange-400 #fb923c
    value = dfaValue;
    const realMax = dfaMaxValue;
    const realCurrent = value;
    const ratio = realCurrent / realMax;
    const scale = 1 - MINIMUM_OPACITY; // Maximum scale adjustment to reach 1

    // Adjust value to start scaling from 0.3 upwards
    // We use a logarithmic function to scale the value.
    // The constants 9 and 0.1 are chosen to make the function start at 0.3 for very small values
    // and approach 1 as the value approaches 1.
    var opacity = MINIMUM_OPACITY + (Math.log10(1 + 9 * ratio) * scale) / Math.log10(10);
    // Ensure the opacity does not exceed 1
    opacity = Math.min(opacity, 1);
    opacity = Math.max(0, opacity);
    // }
    topGradient = `rgba(${rgb}, ${opacity})`;

    if (opacity !== 0 && bottomOpacity === 0) {
      bottomGradient = topGradient;
    }
    if (opacity === 0 && bottomOpacity !== 0) {
      topGradient = bottomGradient;
    }
  }
  return `linear-gradient(to bottom, ${topGradient} 50%, ${bottomGradient} 50%)`;
};

export const keysStringNoLineBreaks = new RegExp(Object.keys(HTML_ANOMALIES_NO_LINE_BREAK_CHAR).join('|'), 'g');
export const keysString = new RegExp(Object.keys(HTML_ANOMALIES).join('|'), 'g');

export function replaceHtmlAnomalies(str: string, replaceLineBreaks = true) {
  if (str === undefined || str === null) {
    return '';
  }
  if (replaceLineBreaks) {
    return str.replace(keysString, (match) => HTML_ANOMALIES[match]);
  }
  return str.replace(keysStringNoLineBreaks, (match) => HTML_ANOMALIES[match]);
}
