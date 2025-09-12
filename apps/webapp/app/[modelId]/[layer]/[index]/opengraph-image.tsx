/* eslint react/no-unknown-property: 0 */
/* eslint no-nested-ternary: 0 */
/* eslint react/no-array-index-key: 0 */

import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/auth/utils';
import { getNeuronOptimized } from '@/lib/db/neuron';
import { getSourceSet } from '@/lib/db/source';
import { env } from '@/lib/env';
import { makeActivationBackgroundColorWithDFA, replaceHtmlAnomalies } from '@/lib/utils/activations';
import { Activation } from '@prisma/client';
import { notFound } from 'next/navigation';
import { ImageResponse } from 'next/og';
import { NextResponse } from 'next/server';

export const alt = 'Neuronpedia Feature Dashboard';
export const size = {
  width: 1200,
  height: 630,
};

export const contentType = 'image/png';

export default async function Image({ params }: { params: { modelId: string; layer: string; index: string } }) {
  const user = await makeAuthedUserFromSessionOrReturnNull();
  const currentNeuron = await getNeuronOptimized(params.modelId, params.layer, params.index, user);
  if (!currentNeuron) {
    notFound();
  }
  const sourceSet = await getSourceSet(currentNeuron.modelId, currentNeuron.sourceSetName || '', user);
  if (!sourceSet) {
    notFound();
  }

  function makeColorRange(numColors: number) {
    const colors = Array.from({ length: numColors }, (_, i) => {
      const ratio = i / (numColors - 1);
      const hex = (color1: string, color2: string, hexRatio: number) => {
        const r = Math.round(
          parseInt(color1.substring(1, 3), 16) * hexRatio + parseInt(color2.substring(1, 3), 16) * (1 - hexRatio),
        ).toString(16);
        const g = Math.round(
          parseInt(color1.substring(3, 5), 16) * hexRatio + parseInt(color2.substring(3, 5), 16) * (1 - hexRatio),
        ).toString(16);
        const b = Math.round(
          parseInt(color1.substring(5, 7), 16) * hexRatio + parseInt(color2.substring(5, 7), 16) * (1 - hexRatio),
        ).toString(16);
        return `#${r.padStart(2, '0')}${g.padStart(2, '0')}${b.padStart(2, '0')}`;
      };
      return hex('#ff8c00', '#ffd097', ratio);
    });
    return colors;
  }

  const negMax = Math.max(...(currentNeuron?.neg_values || []).map((value) => Math.abs(value)));
  const posMax = Math.max(...(currentNeuron?.pos_values || []).map((value) => Math.abs(value)));
  const max = Math.max(negMax, posMax);
  const maxLogit = max;

  const maxActHist = Math.max(...currentNeuron.freq_hist_data_bar_heights.map((value) => Math.abs(value)));
  const histColorRange = makeColorRange(currentNeuron?.freq_hist_data_bar_values.length);
  const maxLogitHist = Math.max(...currentNeuron.logits_hist_data_bar_heights.map((value) => Math.abs(value)));
  const logitColorRange = currentNeuron.logits_hist_data_bar_values.map((value) => {
    if (value >= 0) {
      return '#0000ff';
    }
    return '#ff0000';
  });

  // For DFA, replace activations to remove hidden ones, since we want it to match what users see on the page
  if (sourceSet.showDfa && currentNeuron.activations) {
    // remove all where the dfaTargetIndex !== maxActIndex
    const filteredActs = currentNeuron.activations.filter((act) => {
      if (act.dfaTargetIndex === act.maxValueTokenIndex) {
        return true;
      }
      return false;
    });
    filteredActs.sort((a, b) => {
      if (
        a.values === undefined ||
        b.values === undefined ||
        a.dfaTargetIndex === undefined ||
        a.dfaTargetIndex === null ||
        b.dfaTargetIndex === undefined ||
        b.dfaTargetIndex === null
      ) {
        return 0;
      }
      // sort by activation of the targetDFA token
      const targetDFAactA = a.values[a.dfaTargetIndex];
      const targetDFAactB = b.values[b.dfaTargetIndex];
      if (targetDFAactA > targetDFAactB) {
        return -1;
      }
      if (targetDFAactA < targetDFAactB) {
        return 1;
      }
      return 0;
    });
    currentNeuron.activations = filteredActs;
  }

  function tokenIsInRangeOfAnchorToken(tokenIndex: number, maxIndex: number, currentRange: number) {
    const rangeToUse = sourceSet?.showDfa ? 5 : currentRange;
    return tokenIndex > maxIndex - rangeToUse && tokenIndex < maxIndex + rangeToUse;
  }

  function shouldShowToken(activation: Activation, tokenIndex: number, maxValueIndex: number, currentRange: number) {
    if (!activation.tokens) {
      return false;
    }
    const isInMaxActBuffer = tokenIndex > maxValueIndex - currentRange && tokenIndex < maxValueIndex + currentRange;
    return isInMaxActBuffer;
  }

  if (currentNeuron.maxActApprox === 0) {
    return NextResponse.json({ message: 'success' });
  }

  return new ImageResponse(
    (
      <div tw="flex relative flex-col min-w-[1200px] min-h-[630px] h-[630px] w-[1200px] max-h-[630px] max-w-[1200px] items-center px-8 justify-center bg-white">
        <div tw="flex flex-row text-4xl mb-5 w-full justify-between items-center text-slate-800">
          <div tw="flex flex-row uppercase">
            {currentNeuron.modelId} · {currentNeuron.layer} · {currentNeuron.index}
          </div>
          <div tw="flex flex-row items-center text-3xl font-bold text-slate-700">
            Activations Density {((currentNeuron.frac_nonzero ? currentNeuron.frac_nonzero : 0) * 100).toFixed(3)}%
          </div>
        </div>
        <div tw="mb-0 flex flex-row px-0 pb-0 w-full">
          {sourceSet.showDfa && (
            <div tw="flex flex-col font-bold text-gray-900 text-left min-w-[19%] max-w-[19%] pr-4">
              <div tw="mb-1.5 flex flex-row items-center text-2xl font-bold text-slate-600">Head Attribution</div>
              {currentNeuron.decoder_weights_dist.map((value, ind) => (
                <div key={ind} tw="flex flex-row text-[21px] mb-1 font-extrabold leading-[18px] text-slate-400">
                  <span tw="w-[92px] text-[19px] font-extrabold">Layer {ind}</span>
                  <span tw="rounded bg-slate-400" style={{ width: `${100 * value}px` }} />
                  <span tw="text-slate-700 ml-1.5">{(Math.floor(value * 100) / 100).toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
          <div
            tw={`flex flex-col font-bold text-gray-900 text-left min-w-[25%] max-w-[25%] pr-4 ${
              sourceSet.showDfa && 'pl-4'
            }`}
          >
            <div tw="mb-1 flex flex-row items-center text-2xl font-bold text-slate-600">
              <div>Negative Logits</div>
            </div>
            <div tw="flex flex-col text-[21px] leading-none">
              {currentNeuron?.neg_str.map((s, i) => {
                if (i > 6) {
                  return <div tw="hidden" key={i} />;
                }
                const ratio = Math.abs(currentNeuron?.neg_values[i]) / maxLogit;
                return (
                  // eslint-disable-next-line react/no-array-index-key
                  <div tw="flex flex-row justify-between min-w-[200px] items-center" key={i}>
                    {ratio > 0.9 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#ff2727] bg-opacity-50">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : ratio > 0.75 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#ff2727] bg-opacity-30">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : ratio > 0.5 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#ff2727] bg-opacity-20">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : ratio > 0.25 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#ff2727] bg-opacity-10">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#ff2727] bg-opacity-5">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    )}
                    <div>{currentNeuron?.neg_values[i].toFixed(2)}</div>
                  </div>
                );
              })}
            </div>
          </div>
          <div tw="flex flex-col font-bold text-gray-900 text-left min-w-[25%] max-w-[25%] pl-4">
            <div tw="mb-1 flex flex-row items-center text-2xl font-bold text-slate-600">
              <div>Positive Logits</div>
            </div>
            <div tw="flex flex-col text-[21px] leading-none">
              {currentNeuron?.pos_str.map((s, i) => {
                if (i > 6) {
                  return <div tw="hidden" key={i} />;
                }
                const ratio = Math.abs(currentNeuron?.pos_values[i]) / maxLogit;
                return (
                  <div tw="flex flex-row min-w-[200px] justify-between items-center" key={i}>
                    {ratio > 0.9 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#2727ff] bg-opacity-50">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : ratio > 0.75 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#2727ff] bg-opacity-30">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : ratio > 0.5 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#2727ff] bg-opacity-20">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : ratio > 0.25 ? (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#2727ff] bg-opacity-10">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    ) : (
                      <pre tw="overflow-hidden max-w-[250px] rounded-md font-bold mr-5 my-[3px] px-2 py-1.5 bg-[#2727ff] bg-opacity-5">
                        {replaceHtmlAnomalies(s)}
                      </pre>
                    )}
                    <div>{currentNeuron?.pos_values[i].toFixed(2)}</div>
                  </div>
                );
              })}
            </div>
          </div>
          <div
            tw={`flex flex-col font-bold text-gray-900 text-left ${
              sourceSet.showDfa ? 'min-w-[31%] max-w-[31%]' : 'min-w-[50%] max-w-[50%]'
            } pl-8`}
          >
            <div tw="mb-2 flex flex-row items-center w-full text-2xl font-bold text-slate-600">
              Non-Zero Activations
            </div>
            <div tw="flex flex-row h-[200px] max-h-[200px] w-full items-end border-b border-slate-300 flex-1">
              {currentNeuron.freq_hist_data_bar_heights.map((val, i) => (
                <div
                  key={i}
                  tw="flex flex-1"
                  style={{
                    backgroundColor: histColorRange[i],
                    color: 'rgba(0,0,0,0)',
                    height: `${(val / maxActHist) * 100}%`,
                  }}
                >
                  .
                </div>
              ))}
            </div>
            <div tw="mb-2 mt-5 flex flex-row items-center w-full text-2xl font-bold text-slate-600">Logit Density</div>
            <div tw="flex flex-row h-[200px] max-h-[200px] w-full items-end border-b border-slate-300 flex-1">
              {currentNeuron.logits_hist_data_bar_heights.map((val, i) => (
                <div
                  key={i}
                  tw="flex flex-1"
                  style={{
                    backgroundColor: logitColorRange[i],
                    color: 'rgba(0,0,0,0)',
                    opacity: 0.5,
                    height: `${(val / maxLogitHist) * 100}%`,
                  }}
                >
                  .
                </div>
              ))}
            </div>
          </div>
        </div>
        <div tw="flex flex-col mt-8 justify-start w-full">
          {currentNeuron.activations?.map((activation, i) => {
            if (i > 2) {
              return <div key={i} />;
            }
            return (
              <div key={i} tw="flex text-[23px] mb-2.5 w-full">
                {sourceSet?.showDfa &&
                  activation.tokens?.map((token, tokenIndex) => {
                    if (
                      tokenIsInRangeOfAnchorToken(
                        tokenIndex,
                        activation?.dfaValues?.indexOf(Math.max(...activation.dfaValues)) || 0,
                        5,
                      )
                    ) {
                      return (
                        <span key={tokenIndex} tw="flex">
                          <pre
                            tw={`my-0 text-slate-700 ${
                              sourceSet.showDfa &&
                              tokenIndex === activation?.dfaValues?.indexOf(Math.max(...activation.dfaValues))
                                ? 'border-4 border-emerald-400'
                                : 'py-[4px]'
                            }`}
                            style={{
                              backgroundImage: makeActivationBackgroundColorWithDFA(
                                activation.dfaMaxValue ? activation.dfaMaxValue : 0,
                                activation.dfaValues ? activation.dfaValues[tokenIndex] : 0,
                                '251, 146, 60',
                              ),
                            }}
                            key={tokenIndex}
                          >
                            {replaceHtmlAnomalies(token).replaceAll('↵', ' ')}
                          </pre>
                        </span>
                      );
                    }
                    return <span key={tokenIndex} />;
                  })}

                {sourceSet?.showDfa && <span tw="flex-grow" />}

                {activation.tokens?.map((token, tokenIndex) => {
                  if (
                    shouldShowToken(
                      activation as Activation,
                      tokenIndex,
                      activation.maxValueTokenIndex || 0,
                      sourceSet?.showDfa ? 5 : 7,
                    )
                  ) {
                    return (
                      <span key={tokenIndex} tw="flex">
                        <pre
                          tw={`my-0 text-slate-700 ${
                            tokenIndex === activation.dfaTargetIndex ? 'border-4 border-orange-400' : 'py-[4px]'
                          }`}
                          style={{
                            backgroundImage: makeActivationBackgroundColorWithDFA(
                              activation?.maxValue || 0,
                              activation?.values ? activation?.values[tokenIndex] : 0,
                              '52, 211, 153',
                              // activation.dfaValues
                              //   ? activation.dfaValues[tokenIndex]
                              //   : 0,
                              // activation.dfaMaxValue
                              //   ? activation.dfaMaxValue
                              //   : 0,
                            ),
                          }}
                          key={tokenIndex}
                        >
                          {replaceHtmlAnomalies(token).replaceAll('↵', ' ')}
                        </pre>
                      </span>
                    );
                  }
                  return <span key={tokenIndex} />;
                })}
              </div>
            );
          })}
        </div>
        <div tw="absolute flex flex-row justify-end bottom-0 text-slate-400 right-0 pb-2 pr-3 leading-none">
          {env.NEXT_PUBLIC_URL}/{currentNeuron.modelId}/{currentNeuron.layer}/{currentNeuron.index}
        </div>
      </div>
    ),
    {
      ...size,
    },
  );
}
