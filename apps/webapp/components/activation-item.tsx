'use client';

import {
  ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS,
  LINE_BREAK_REPLACEMENT_CHAR,
  makeActivationBackgroundColorWithDFA,
  replaceHtmlAnomalies,
} from '@/lib/utils/activations';
import { cn } from '@/lib/utils/ui';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Copy } from 'lucide-react';
import { ActivationPartialWithRelations } from 'prisma/generated/zod';
import { useEffect, useState } from 'react';
import ActivationItemTokenTooltip from './activation-item-token-tooltip';

const ACTIVATION_MAX_COPY_TOKENS = 128;
const DFA_TARGET_TOKEN_CLASSNAME = 'border-2 border-orange-400';
const DFA_SOURCE_TOKEN_CLASSNAME = 'border-2 border-emerald-400';
const REGULAR_TOKEN_CLASSNAME = 'border-2 border-transparent hover:bg-slate-200';
export const CENTER_ME_CLASSNAME = 'center-me';

export default function ActivationItem({
  activation,
  // only use this if you want the activation colors to be relative to other activations in a list
  // otherwise, we assume that the darkest green will be the max activation value of the tokens
  // in this activation text
  overallMaxActivationValueInList = activation.maxValue || 0,
  tokensToDisplayAroundMaxActToken = 10000,
  showLineBreaks = false,
  setActivationTestText,
  showTopActivationToken = false,
  centerAndBorderOnTokenIndex = undefined,
  enableExpanding = true,
  dfa = false,
  dfaSplit = false,
  showCopy = true,
  overrideTextSize = 'text-xs',
  overrideLeading = 'leading-none sm:leading-tight',
  overrideTextColor = 'text-slate-600',
  className,
}: {
  activation: ActivationPartialWithRelations;
  overallMaxActivationValueInList?: number;
  tokensToDisplayAroundMaxActToken?: number;
  showLineBreaks?: boolean;
  setActivationTestText?: (value: string) => void;
  showTopActivationToken?: boolean;
  centerAndBorderOnTokenIndex?: number;
  enableExpanding?: boolean;
  dfa?: boolean;
  dfaSplit?: boolean;
  showCopy?: boolean;
  overrideTextSize?: string;
  overrideLeading?: string;
  overrideTextColor?: string;
  className?: string;
}) {
  const [currentRange, setCurrentRange] = useState(tokensToDisplayAroundMaxActToken);
  const [isExpanded, setIsExpanded] = useState(false);
  const [dfaMaxIndex, setDfaMaxIndex] = useState(0);

  const maxActivationTokenIndex = activation.maxValueTokenIndex || 0;
  const maxActivationValue = activation.maxValue || 0;
  const maxDfaTokenIndex = activation.dfaValues?.indexOf(Math.max(...activation.dfaValues)) || 0;
  const maxDfaValue = activation.dfaMaxValue || 0;

  let firstTokenShown = -1;

  useEffect(() => {
    if (!isExpanded) {
      setCurrentRange(tokensToDisplayAroundMaxActToken);
    } else {
      setCurrentRange(10000);
    }
  }, [isExpanded]);

  useEffect(() => {
    if (dfa) {
      setDfaMaxIndex(activation.dfaValues?.indexOf(Math.max(...activation.dfaValues)) || 0);
    }
  }, [dfa, activation.dfaValues]);

  function tokenIsInRangeOfAnchorToken(tokenIndex: number, maxIndex: number) {
    if (!activation.tokens) {
      return false;
    }
    return tokenIndex > maxIndex - currentRange && tokenIndex < maxIndex + currentRange;
  }

  function shouldShowToken(tokenIndex: number) {
    if (!activation.tokens) {
      return false;
    }
    if (activation.maxValue === 0) {
      // show the middle of the text
      return (
        tokenIndex > activation.tokens.length / 2 - currentRange &&
        tokenIndex < activation.tokens.length / 2 + currentRange
      );
    }

    let toReturn = false;
    const isInMaxActBuffer =
      tokenIndex > maxActivationTokenIndex - currentRange && tokenIndex < maxActivationTokenIndex + currentRange;
    if (dfa && activation.dfaTargetIndex !== undefined && activation.dfaTargetIndex !== null) {
      const isInDFASourceBuffer =
        tokenIndex > maxDfaTokenIndex - currentRange && tokenIndex < maxDfaTokenIndex + currentRange;
      const isInTargetDFABuffer =
        tokenIndex > activation.dfaTargetIndex - currentRange && tokenIndex < activation.dfaTargetIndex + currentRange;
      toReturn = isInDFASourceBuffer || isInTargetDFABuffer;
    } else {
      toReturn = isInMaxActBuffer;
    }
    if (toReturn === true && firstTokenShown === -1) {
      firstTokenShown = tokenIndex;
    }
    return toReturn;
  }

  return (
    <div className={cn('flex w-full flex-row items-center justify-start gap-x-2', className)}>
      {showTopActivationToken && (
        <div className="flex flex-row items-center gap-x-1.5">
          {dfa && (
            <div className="hidden items-center justify-center gap-y-0.5 whitespace-nowrap pr-1 text-center font-mono text-[10px] leading-normal text-slate-600 sm:flex sm:w-16 sm:flex-col">
              <div className="rounded bg-slate-100 px-1.5 font-bold">
                {activation.tokens && replaceHtmlAnomalies(activation.tokens[maxDfaTokenIndex])}
              </div>
              {activation.dfaValues && (
                <div className="font-sans font-semibold text-orange-500">{maxDfaValue?.toFixed(2)}</div>
              )}
            </div>
          )}
          <div className="hidden items-center justify-center gap-y-0.5 whitespace-nowrap pr-1 text-center font-mono text-[10px] leading-normal text-slate-600 sm:flex sm:w-16 sm:flex-col">
            <div className="rounded bg-slate-100 px-1.5 font-bold">
              {activation.tokens && replaceHtmlAnomalies(activation.tokens[maxActivationTokenIndex])}
            </div>
            <div className="font-sans font-semibold text-emerald-600">{maxActivationValue?.toFixed(2)}</div>
          </div>
        </div>
      )}
      {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions, jsx-a11y/click-events-have-key-events */}
      <div
        // if it's the smallest range and we're split in dfa mode, then force it to be stacked
        className={`flex-1 ${
          dfa === true &&
          dfaSplit === true &&
          currentRange === ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[0].size &&
          'flex w-full flex-row items-center overflow-hidden'
        } ${overrideLeading}`}
        onClick={() => {
          if (enableExpanding) {
            setIsExpanded(!isExpanded);
          }
        }}
      >
        {/* if we have DFA and we're split and it isn't expanded, these are the tokens on the left side (max DFA values (orange background), the DFA source) */}
        {dfa && dfaSplit && !isExpanded && activation.tokens && (
          <>
            {activation.tokens.map((token, tokenIndex) => {
              const tokenWithReplacedAnomalies = replaceHtmlAnomalies(token);
              const tokenEndsWithSpace = tokenWithReplacedAnomalies.endsWith(' ');
              const tokenStartsWithSpace = tokenWithReplacedAnomalies.startsWith(' ');
              if (tokenIsInRangeOfAnchorToken(tokenIndex, dfaMaxIndex)) {
                return (
                  <span key={tokenIndex}>
                    <Tooltip.Provider skipDelayDuration={0} delayDuration={0}>
                      <Tooltip.Root disableHoverableContent>
                        <Tooltip.Trigger asChild>
                          <span
                            className={`inline-block cursor-default whitespace-nowrap bg-origin-border font-mono ${
                              !dfaSplit && tokenIndex === activation.dfaTargetIndex
                                ? DFA_TARGET_TOKEN_CLASSNAME
                                : dfa && tokenIndex === dfaMaxIndex
                                  ? DFA_SOURCE_TOKEN_CLASSNAME
                                  : REGULAR_TOKEN_CLASSNAME
                            } ${tokenEndsWithSpace && 'pr-1'} ${tokenStartsWithSpace && 'pl-1'} ${
                              activation.lossValues &&
                              (activation.lossValues[tokenIndex] > 0
                                ? 'border-b-red-400'
                                : activation.lossValues[tokenIndex] < 0
                                  ? 'border-b-blue-400'
                                  : '')
                            } ${overrideTextColor} ${overrideTextSize} `}
                            style={{
                              backgroundImage: dfaSplit
                                ? makeActivationBackgroundColorWithDFA(
                                    activation.dfaMaxValue ? activation.dfaMaxValue : 0,
                                    activation.dfaValues ? activation.dfaValues[tokenIndex] : 0,
                                    '251, 146, 60',
                                  )
                                : makeActivationBackgroundColorWithDFA(
                                    overallMaxActivationValueInList,
                                    activation.values ? activation.values[tokenIndex] : 0,
                                    '52, 211, 153',
                                    activation.dfaValues ? activation.dfaValues[tokenIndex] : 0,
                                    activation.dfaMaxValue ? activation.dfaMaxValue : 0,
                                  ),
                            }}
                          >
                            {tokenWithReplacedAnomalies}
                          </span>
                        </Tooltip.Trigger>
                        <ActivationItemTokenTooltip
                          activation={activation}
                          token={token}
                          tokenIndex={tokenIndex}
                          dfaMaxIndex={dfaMaxIndex}
                        />
                      </Tooltip.Root>
                      {(token.indexOf('\n') !== -1 || tokenWithReplacedAnomalies === LINE_BREAK_REPLACEMENT_CHAR) &&
                        showLineBreaks && <br />}
                    </Tooltip.Provider>
                  </span>
                );
              }
              return '';
            })}
            {/* this is the separator between the DFA and the max activations */}
            <span className="inline-block w-full" />
          </>
        )}

        {/* in all cases, we should display the activations / max activation (green background) */}
        {activation.tokens &&
          activation.tokens.map((token, tokenIndex) => {
            const tokenWithReplacedAnomalies = replaceHtmlAnomalies(token);
            const tokenEndsWithSpace = tokenWithReplacedAnomalies.endsWith(' ');
            const tokenStartsWithSpace = tokenWithReplacedAnomalies.startsWith(' ');
            if (
              dfa && dfaSplit
                ? tokenIsInRangeOfAnchorToken(tokenIndex, maxActivationTokenIndex)
                : shouldShowToken(tokenIndex)
            ) {
              return (
                <span key={tokenIndex}>
                  {dfa &&
                    !dfaSplit &&
                    tokenIndex > 0 &&
                    shouldShowToken(tokenIndex - 1) === false &&
                    firstTokenShown !== tokenIndex && (
                      <Tooltip.Provider skipDelayDuration={0} delayDuration={0}>
                        <Tooltip.Root disableHoverableContent>
                          <Tooltip.Trigger asChild>
                            <span className="inline-block cursor-pointer rounded-full border-2 border-x-[4px] border-white bg-slate-100 bg-origin-border px-[4px] py-0 text-[11px] font-medium text-slate-500 hover:bg-sky-200 hover:text-sky-700">
                              ⋯
                            </span>
                          </Tooltip.Trigger>
                          <Tooltip.Portal>
                            <Tooltip.Content
                              className="z-50 flex w-full flex-col items-center gap-y-1.5 rounded border bg-amber-50 px-4 py-2 text-center text-xs font-semibold text-slate-700 shadow"
                              sideOffset={6}
                            >
                              View Full Context
                            </Tooltip.Content>
                          </Tooltip.Portal>
                        </Tooltip.Root>
                      </Tooltip.Provider>
                    )}
                  <Tooltip.Provider skipDelayDuration={0} delayDuration={0}>
                    <Tooltip.Root disableHoverableContent>
                      <Tooltip.Trigger asChild>
                        <span
                          className={`${centerAndBorderOnTokenIndex === tokenIndex ? CENTER_ME_CLASSNAME : ''} inline-block cursor-default whitespace-nowrap bg-origin-border font-mono ${
                            tokenIndex === activation.dfaTargetIndex
                              ? DFA_TARGET_TOKEN_CLASSNAME
                              : dfa && (!dfaSplit || isExpanded) && tokenIndex === dfaMaxIndex
                                ? DFA_SOURCE_TOKEN_CLASSNAME
                                : centerAndBorderOnTokenIndex === tokenIndex
                                  ? 'border border-slate-600'
                                  : REGULAR_TOKEN_CLASSNAME
                          } ${tokenEndsWithSpace && 'pr-1'} ${tokenStartsWithSpace && 'pl-1'} ${
                            activation.lossValues &&
                            (activation.lossValues[tokenIndex] > 0
                              ? 'border-b-red-400'
                              : activation.lossValues[tokenIndex] < 0
                                ? 'border-b-blue-400'
                                : '')
                          } ${overrideTextColor} ${overrideTextSize} `}
                          style={{
                            backgroundImage: makeActivationBackgroundColorWithDFA(
                              overallMaxActivationValueInList,
                              activation.values ? activation.values[tokenIndex] : 0,
                              '52, 211, 153',
                              (!dfaSplit || isExpanded) && activation.dfaValues ? activation.dfaValues[tokenIndex] : 0,
                              (!dfaSplit || isExpanded) && activation.dfaMaxValue ? activation.dfaMaxValue : 0,
                            ),
                          }}
                        >
                          {tokenWithReplacedAnomalies}
                        </span>
                      </Tooltip.Trigger>
                      <ActivationItemTokenTooltip
                        activation={activation}
                        token={token}
                        tokenIndex={tokenIndex}
                        dfaMaxIndex={dfaMaxIndex}
                      />
                    </Tooltip.Root>
                    {(token.indexOf('\n') !== -1 || tokenWithReplacedAnomalies === LINE_BREAK_REPLACEMENT_CHAR) &&
                      showLineBreaks && <br />}
                  </Tooltip.Provider>
                </span>
              );
            }
            return '';
          })}
      </div>
      {showCopy && setActivationTestText && (
        <div className="mt-0 flex flex-row items-center justify-end">
          <button
            type="button"
            className="flex cursor-pointer flex-row rounded bg-slate-100 p-1.5 text-[10px] font-medium text-slate-400 hover:bg-slate-200"
            title="Copy/Remix"
            onClick={() => {
              const wholeString = activation.tokens?.join('') || '';
              // truncate to the tokens around the max activation
              if (activation.values && activation.tokens && activation.values.length > ACTIVATION_MAX_COPY_TOKENS) {
                const maxIndex = activation.values?.indexOf(Math.max(...activation.values!));
                const startIndex = Math.max(0, maxIndex - ACTIVATION_MAX_COPY_TOKENS / 2);
                const endIndex = Math.min(activation.tokens?.length, maxIndex + ACTIVATION_MAX_COPY_TOKENS / 2);
                const truncatedString = activation.tokens?.slice(startIndex, endIndex);
                setActivationTestText(truncatedString.join(''));
              } else {
                setActivationTestText(wholeString);
              }
              window.scrollTo(0, 0);
            }}
          >
            <Copy className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}
