'use client';

import { SteerResult } from '@/app/api/steer/route';
import { useGlobalContext } from '@/components/provider/global-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { env } from '@/lib/env';
import { STEER_MAX_PROMPT_CHARS, SteerFeature } from '@/lib/utils/steer';
import copy from 'copy-to-clipboard';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import { ArrowUp, RotateCcw, Share, X } from 'lucide-react';
import { NPLogprob, NPSteerMethod } from 'neuronpedia-inference-client';
import { useRef } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import CustomTooltip from '../custom-tooltip';

export default function SteerCompletion({
  showSettingsOnMobile,
  isSteering,
  setIsSteering,
  defaultCompletionText,
  setDefaultCompletionText,
  defaultCompletionLogProbs,
  setDefaultCompletionLogProbs,
  steeredCompletionText,
  setSteeredCompletionText,
  steeredCompletionLogProbs,
  setSteeredCompletionLogProbs,
  modelId,
  selectedFeatures,
  typedInText,
  setTypedInText,
  reset,
  copying,
  setCopying,
  setUrl,
  temperature,
  steerTokens,
  freqPenalty,
  randomSeed,
  seed,
  strMultiple,
  steerMethod,
}: {
  showSettingsOnMobile: boolean;
  isSteering: boolean;
  setIsSteering: (isSteering: boolean) => void;
  defaultCompletionText: string;
  setDefaultCompletionText: (text: string) => void;
  defaultCompletionLogProbs: NPLogprob[] | null;
  setDefaultCompletionLogProbs: (logprobs: NPLogprob[] | null) => void;
  steeredCompletionText: string;
  setSteeredCompletionText: (text: string) => void;
  steeredCompletionLogProbs: NPLogprob[] | null;
  setSteeredCompletionLogProbs: (logprobs: NPLogprob[] | null) => void;
  modelId: string;
  selectedFeatures: SteerFeature[];
  typedInText: string;
  setTypedInText: (text: string) => void;
  reset: () => void;
  copying: boolean;
  setCopying: (copying: boolean) => void;
  setUrl: (url: string) => void;
  temperature: number;
  steerTokens: number;
  freqPenalty: number;
  randomSeed: boolean;
  seed: number;
  strMultiple: number;
  steerMethod: NPSteerMethod;
}) {
  const { showToastMessage, showToastServerError } = useGlobalContext();
  const abortControllerRef = useRef<AbortController | null>(null);

  async function stopSteering() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }

  async function sendCompletion() {
    if (selectedFeatures.length === 0) {
      alert('Please select a demo, a preset mode, or add your own features to steer.');
      return;
    }
    if (typedInText.trim().length === 0) {
      alert('Please enter a message.');
      return;
    }
    setIsSteering(true);

    setDefaultCompletionText(typedInText);
    setSteeredCompletionText(typedInText);
    setDefaultCompletionLogProbs(null);
    setSteeredCompletionLogProbs(null);

    // check for character limit
    if (typedInText.length >= STEER_MAX_PROMPT_CHARS) {
      alert(
        `Sorry, we limit the length of each completion.\nPlease keep your completion text under ${STEER_MAX_PROMPT_CHARS} characters.`,
      );
      setDefaultCompletionText('');
      setSteeredCompletionText('');
      setDefaultCompletionLogProbs(null);
      setSteeredCompletionLogProbs(null);
      setIsSteering(false);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    try {
      const stream = true;
      const response = await fetch('/api/steer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: typedInText,
          modelId,
          features: selectedFeatures,
          temperature,
          n_tokens: steerTokens,
          freq_penalty: freqPenalty,
          seed: randomSeed ? Math.floor(Math.random() * 200000000 - 100000000) : seed,
          strength_multiplier: strMultiple,
          steer_method: steerMethod,
          stream,
        }),
        signal,
      });
      if (!response || !response.body) {
        if (response.status === 404) {
          alert(
            !env.IS_ACTUALLY_NEURONPEDIA_ORG
              ? 'Unable to steer with the selected feature. Did you check if you downloaded/imported this SAE?'
              : 'Unable to steer with the selected feature - it was not found.',
          );
        } else {
          alert('Sorry, your message could not be sent at this time. Please try again later.');
        }
        showToastServerError();
        setIsSteering(false);
        setDefaultCompletionText('');
        setSteeredCompletionText('');
        setDefaultCompletionLogProbs(null);
        setSteeredCompletionLogProbs(null);
        return;
      }
      // check if the response is a stream
      const contentType = response.headers.get('content-type');
      if (contentType === 'text/event-stream') {
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .pipeThrough(new EventSourceParserStream())
          .getReader();

        // eslint-disable-next-line
        while (true) {
          // eslint-disable-next-line
          const { done, value } = await reader.read();
          if (done) {
            setIsSteering(false);
            break;
          }
          const data = JSON.parse(value.data) as SteerResult;
          setDefaultCompletionText(data.DEFAULT || '');
          setSteeredCompletionText(data.STEERED || '');
          setDefaultCompletionLogProbs(data.defaultLogProbs || null);
          setSteeredCompletionLogProbs(data.steeredLogProbs || null);
          if (data.id) {
            setUrl(data.id || '');
          }
        }
      } else {
        const data = await response.json();
        setDefaultCompletionText(data.DEFAULT || '');
        setSteeredCompletionText(data.STEERED || '');
        setUrl(data.id || '');
        setIsSteering(false);
        setDefaultCompletionLogProbs(data.defaultLogProbs || null);
        setSteeredCompletionLogProbs(data.steeredLogProbs || null);
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        showToastMessage('Steering aborted.');
        setIsSteering(false);
        setDefaultCompletionText('');
        setSteeredCompletionText('');
      } else {
        console.error(error);
        showToastServerError();
      }
    }
  }

  function makeCompletionText(text: string, logprobs: NPLogprob[] | null) {
    const endText =
      logprobs && logprobs.length > 0
        ? logprobs.map((logprob) => (logprob.topLogprobs ? (logprob.token ? logprob.token : '') : '')).join('')
        : '';
    const truncatedText = endText.length > 0 ? text.slice(0, -endText.length) : text;
    const logprobSpans =
      logprobs && logprobs.length > 0
        ? logprobs.map((logprob, index) => (
            <>
              <CustomTooltip
                key={index}
                side="top"
                trigger={
                  <span className="cursor-pointer whitespace-pre rounded hover:bg-slate-300/70">
                    {logprob.token.replaceAll('\n', '⏎')}
                  </span>
                }
              >
                {logprob.topLogprobs ? (
                  <div className="flex flex-col items-center justify-center">
                    <div className="mb-1 text-[10px] font-medium uppercase text-slate-400">Top Logprobs</div>
                    {logprob.topLogprobs.map((topLogprob, i) => (
                      <div
                        key={topLogprob.token + i}
                        className="flex w-full flex-row items-center justify-between gap-x-5"
                      >
                        <div className="whitespace-pre rounded-md bg-slate-200 px-1 py-0.5 font-mono">
                          {topLogprob.token.replaceAll('\n', '⏎')}
                        </div>
                        <div className="font-mono">{topLogprob.logprob.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  ''
                )}
              </CustomTooltip>

              {logprob.token.includes('\n') ? <br /> : ''}
            </>
          ))
        : null;
    return (
      <>
        {truncatedText}
        {logprobSpans}
      </>
    );
  }

  return (
    <div
      className={`relative h-full max-h-[calc(100dvh-48px)] min-h-[calc(100dvh-48px)] w-full min-w-0 flex-row text-sm font-medium leading-normal text-slate-500 sm:h-full sm:max-h-[calc(100dvh-76px)] sm:min-h-[calc(100dvh-76px)] sm:w-auto sm:basis-2/3 ${
        showSettingsOnMobile ? 'hidden sm:flex' : 'flex'
      }`}
    >
      <div className="hidden h-full flex-1 flex-col overflow-y-scroll bg-slate-100 px-5 py-2 pt-28 text-left text-xs text-slate-400 sm:flex">
        <div className="sticky top-0.5 flex flex-row justify-center uppercase text-white sm:top-0">
          <div className="select-none rounded-full px-5 py-1 text-[10px] font-bold text-slate-600">Normal</div>
        </div>
        <div className="pb-5 pt-6 text-[14px] font-medium leading-normal text-slate-600 sm:pb-32 sm:pt-1">
          {!isSteering && defaultCompletionText.length === 0 && (
            <div className="w-full pl-3 pt-3 text-left text-xl text-slate-400">
              Hey, {`I'm normal ${modelId}!`}
              <div className="mt-2 text-sm text-slate-500">{`I'm the default, non-steered model.`}</div>
            </div>
          )}
          <div className="whitespace-pre-wrap break-words">
            {defaultCompletionText
              ? defaultCompletionLogProbs && defaultCompletionLogProbs.length > 0
                ? makeCompletionText(defaultCompletionText, defaultCompletionLogProbs)
                : defaultCompletionText
              : isSteering
                ? typedInText
                : defaultCompletionText}
          </div>
          {isSteering && <LoadingSquare className="px-0 py-3" />}
        </div>
      </div>
      <div className="flex h-full max-h-[calc(100dvh-48px)] min-h-[calc(100dvh-48px)] w-full flex-1 flex-col overflow-y-scroll bg-sky-100 px-3 py-2 pt-48 text-left text-xs text-slate-400 sm:max-h-[calc(100dvh-76px)] sm:min-h-[calc(100dvh-76px)] sm:px-5 sm:pt-28">
        <div className="sticky top-0.5 flex flex-row justify-center uppercase text-sky-700 sm:top-0">
          <div className="select-none rounded-full px-5 py-1 text-[10px] font-bold">Steered</div>
        </div>
        <div className="pb-28 pt-5 text-[14px] font-medium leading-normal text-slate-600 sm:pb-32 sm:pt-1">
          {!isSteering && steeredCompletionText.length === 0 && (
            <div className="w-full pl-3 pr-3 pt-3 text-left text-xl text-sky-600">
              Hey, {`I'm steered ${modelId}!`}
              {selectedFeatures.length > 0 ? (
                <div className="mt-5 text-sm text-sky-700">
                  {`You're steering me to:`}
                  {selectedFeatures.map((f) => {
                    if (f.strength > 0) {
                      return (
                        <div key={f.index} className="ml-3 mt-1">
                          - Boost {f.explanation}
                        </div>
                      );
                    }
                    return (
                      <div key={f.index} className="ml-3 mt-1">
                        - Reduce {f.explanation}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="mt-2 text-sm text-sky-700">
                  Choose a demo, select a preset, or manually search and add features.
                </div>
              )}
            </div>
          )}
          <div className="whitespace-pre-wrap break-words">
            {steeredCompletionText
              ? steeredCompletionLogProbs && steeredCompletionLogProbs.length > 0
                ? makeCompletionText(steeredCompletionText, steeredCompletionLogProbs)
                : steeredCompletionText
              : isSteering
                ? typedInText
                : steeredCompletionText}
          </div>
          {isSteering && <LoadingSquare className="px-0 py-3" />}
        </div>
      </div>

      <div className="absolute top-16 flex w-full flex-col items-center justify-center sm:top-10">
        <div className="relative flex w-full min-w-[95%] flex-col items-center justify-center gap-x-2 gap-y-3 px-7 sm:flex-row">
          <div className="flex w-full flex-row items-center justify-center gap-x-2">
            <ReactTextareaAutosize
              name="searchQuery"
              disabled={isSteering}
              value={typedInText}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !isSteering) {
                  sendCompletion();
                  e.preventDefault();
                }
              }}
              onChange={(e) => {
                setTypedInText(e.target.value);
              }}
              required
              placeholder="Type some text to complete."
              className="mt-0 w-full flex-1 resize-none rounded-lg border border-slate-300 px-5 py-3.5 pr-10 text-left text-xs font-medium text-slate-800 placeholder-slate-400 shadow-md transition-all focus:border-slate-300 focus:shadow-lg focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
            />
            <button
              type="button"
              onClick={() => {
                if (!isSteering) {
                  sendCompletion();
                } else {
                  stopSteering();
                }
              }}
              className="flex h-full cursor-pointer items-center justify-center"
            >
              {!isSteering ? (
                <ArrowUp className="h-8 w-8 rounded-full bg-gBlue p-1.5 text-white hover:bg-gBlue/80" />
              ) : (
                <X className="h-8 w-8 rounded-full bg-red-400 p-1.5 text-white hover:bg-red-600" />
              )}
            </button>
          </div>
          <div className="my-0 flex flex-row gap-x-2 sm:my-0">
            <button
              type="button"
              disabled={defaultCompletionText.length === 0 || isSteering}
              onClick={() => {
                if (defaultCompletionText.length === 0) {
                  return;
                }
                reset();
              }}
              className="flex aspect-square h-8 cursor-pointer flex-row items-center justify-center gap-x-1.5 rounded-full bg-slate-300 px-4 text-xs text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300 sm:aspect-auto sm:h-10 sm:px-5 sm:text-sm"
            >
              <RotateCcw className="h-3.5 w-3.5 sm:h-5 sm:w-5" />
              <span className="inline-block whitespace-pre">Reset</span>
            </button>
            <button
              type="button"
              disabled={defaultCompletionText.length === 0 || isSteering}
              onClick={() => {
                if (defaultCompletionText.length === 0) {
                  return;
                }
                // get current url
                setCopying(true);
                copy(window.location.href);
                showToastMessage(
                  'Copied share link to clipboard.\nPaste it somewhere to share your conversation and steering settings.',
                );
              }}
              className="flex aspect-square h-8 cursor-pointer flex-row items-center justify-center gap-x-1.5 rounded-full bg-slate-300 px-4 text-xs text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300 sm:aspect-auto sm:h-10 sm:px-5 sm:text-sm"
            >
              <Share className="h-3.5 w-3.5 sm:h-5 sm:w-5" />
              <span className="inline-block whitespace-pre">{copying ? 'Copied' : 'Share'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
