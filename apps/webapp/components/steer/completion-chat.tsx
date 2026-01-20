import { SteerResultChat } from '@/app/api/steer-chat/route';
import { useGlobalContext } from '@/components/provider/global-provider';
import SteerChatMessage from '@/components/steer/chat-message';
import { LoadingSquare } from '@/components/svg/loading-square';
import { IS_ACTUALLY_NEURONPEDIA_ORG } from '@/lib/env';
import { ChatMessage, STEER_MAX_PROMPT_CHARS_THINKING, SteerFeature } from '@/lib/utils/steer';
import copy from 'copy-to-clipboard';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import { ArrowUp, RotateCcw, Share, X } from 'lucide-react';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import { useEffect, useRef } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import { STEER_MAX_PROMPT_CHARS } from '@/lib/utils/steer';
import { NNSIGHT_MODELS } from './steerer';

export default function SteerCompletionChat({
  showSettingsOnMobile,
  isSteering,
  setIsSteering,
  defaultChatMessages,
  setDefaultChatMessages,
  steeredChatMessages,
  setSteeredChatMessages,
  modelId,
  selectedFeatures,
  reset,
  typedInText,
  setTypedInText,
  copying,
  setCopying,
  setUrl,
  temperature,
  steerTokens,
  freqPenalty,
  randomSeed,
  seed,
  strMultiple,
  steerSpecialTokens,
  steerMethod,
}: {
  showSettingsOnMobile: boolean;
  isSteering: boolean;
  setIsSteering: (isSteering: boolean) => void;
  defaultChatMessages: ChatMessage[];
  setDefaultChatMessages: (chatMessages: ChatMessage[]) => void;
  steeredChatMessages: ChatMessage[];
  setSteeredChatMessages: (chatMessages: ChatMessage[]) => void;
  modelId: string;
  selectedFeatures: SteerFeature[];
  reset: () => void;
  typedInText: string;
  setTypedInText: (text: string) => void;
  copying: boolean;
  setCopying: (copying: boolean) => void;
  setUrl: (url: string) => void;
  temperature: number;
  steerTokens: number;
  freqPenalty: number;
  randomSeed: boolean;
  seed: number;
  strMultiple: number;
  steerSpecialTokens: boolean;
  steerMethod: NPSteerMethod;
}) {
  const normalEndRef = useRef<HTMLDivElement | null>(null);
  const steeredEndRef = useRef<HTMLDivElement | null>(null);
  const { showToastMessage, showToastServerError } = useGlobalContext();
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToNewestChatMessage = () => {
    normalEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
    if (steeredEndRef.current && steeredEndRef.current?.scrollHeight > 400) {
      steeredEndRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'end',
      });
    }
  };

  useEffect(() => {
    if (steeredChatMessages.length > 0 || defaultChatMessages.length > 0) {
      scrollToNewestChatMessage();
    }
  }, [steeredChatMessages, defaultChatMessages]);

  async function stopSteering() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }

  async function sendChat() {
    if (selectedFeatures.length === 0) {
      alert('Please select a demo, a preset mode, or add your own features to steer.');
      return;
    }
    if (typedInText.trim().length === 0) {
      alert('Please enter a message.');
      return;
    }
    setIsSteering(true);

    const newDefaultChatMessages: ChatMessage[] = [...defaultChatMessages, { content: typedInText, role: 'user' }];
    const newSteeredChatMessages: ChatMessage[] = [...steeredChatMessages, { content: typedInText, role: 'user' }];
    // add to the chat messages (it will show up on UI as we load it)
    setDefaultChatMessages(newDefaultChatMessages);
    setSteeredChatMessages(newSteeredChatMessages);

    // calculate the number of characters in all the chat messages
    const defaultPromptToSendChars = newDefaultChatMessages.map((m) => m.content).join('').length;
    const steeredPromptToSendChars = newSteeredChatMessages.map((m) => m.content).join('').length;

    // // check for character limit
    const maxPromptChars = NNSIGHT_MODELS.includes(modelId) ? STEER_MAX_PROMPT_CHARS_THINKING : STEER_MAX_PROMPT_CHARS;
    if (defaultPromptToSendChars >= maxPromptChars || steeredPromptToSendChars >= maxPromptChars) {
      alert('Sorry, we limit the length of each chat conversation.\nPlease click Reset to start a new conversation.');
      setIsSteering(false);
      removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    // send the chat messages to the backend
    try {
      const stream = true;
      const response = await fetch(`/api/steer-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          defaultChatMessages: newDefaultChatMessages,
          steeredChatMessages: newSteeredChatMessages,
          modelId,
          features: selectedFeatures,
          temperature,
          n_tokens: steerTokens,
          freq_penalty: freqPenalty,
          seed: randomSeed ? Math.floor(Math.random() * 200000000 - 100000000) : seed,
          strength_multiplier: strMultiple,
          steer_method: steerMethod,
          steer_special_tokens: steerSpecialTokens,
          stream,
        }),
        signal,
      });
      if (!response || !response.body) {
        alert('Sorry, your message could not be sent at this time. Please try again later.');
        showToastServerError();
        setIsSteering(false);
        removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
        return;
      }
      if (response.status !== 200) {
        if (response.status === 404) {
          alert(
            !IS_ACTUALLY_NEURONPEDIA_ORG
              ? 'Unable to steer with the selected feature. Did you check if you downloaded/imported this SAE?'
              : 'Unable to steer with the selected feature - it was not found.',
          );
        } else {
          showToastServerError();
        }
        setIsSteering(false);
        removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
        return;
      }

      // check if the response is a stream
      const contentType = response.headers.get('content-type');
      if (contentType === 'text/event-stream') {
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .pipeThrough(new EventSourceParserStream())
          .getReader();

        while (true) {
          // eslint-disable-next-line
          const { done, value } = await reader.read();
          if (done) {
            setIsSteering(false);
            break;
          }
          const data = JSON.parse(value.data) as SteerResultChat;
          if (data.DEFAULT?.chatTemplate) {
            setDefaultChatMessages(data.DEFAULT?.chatTemplate || []);
          }
          if (data.STEERED?.chatTemplate) {
            setSteeredChatMessages(data.STEERED?.chatTemplate || []);
          }
          if (data.id) {
            setUrl(data.id);
          }
          setTypedInText('');
        }
      } else {
        const data = await response.json();
        setDefaultChatMessages(data.DEFAULT?.chatTemplate || []);
        setSteeredChatMessages(data.STEERED?.chatTemplate || []);
        setUrl(data.id);
        setTypedInText('');
        setIsSteering(false);
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        showToastMessage('Steering aborted.');
      } else {
        console.error(error);
        showToastServerError();
      }
      setIsSteering(false);
      removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
    }
  }

  function removeLastFailedUserMessage(defaultMsgs: ChatMessage[], steeredMsgs: ChatMessage[]) {
    if (defaultMsgs.length > 0 && defaultMsgs[defaultMsgs.length - 1].role === 'user') {
      setDefaultChatMessages(defaultMsgs.slice(0, -1));
    }
    if (steeredMsgs.length > 0 && steeredMsgs[steeredMsgs.length - 1].role === 'user') {
      setSteeredChatMessages(steeredMsgs.slice(0, -1));
    }
  }

  return (
    <div
      className={`relative h-full max-h-[calc(100dvh-48px)] min-h-[calc(100dvh-48px)] w-full min-w-0 flex-col text-sm font-medium leading-normal text-slate-500 sm:h-full sm:max-h-[calc(100dvh-76px)] sm:min-h-[calc(100dvh-76px)] sm:w-auto sm:basis-2/3 ${showSettingsOnMobile ? 'hidden sm:flex' : 'flex'
        }`}
    >
      {/* background for large screen and multiline input */}
      <div className="absolute top-0 flex h-full w-full flex-row">
        <div className="hidden flex-1 bg-slate-100 sm:block" />
        <div className="flex-1 bg-sky-100" />
      </div>
      <div className="relative flex h-full w-full flex-col sm:flex-row">
        <div className="hidden h-full flex-1 flex-col overflow-y-scroll bg-slate-100 px-5 py-2 text-left text-xs text-slate-400 sm:flex">
          <div className="sticky top-0.5 flex flex-row justify-center uppercase text-white sm:top-0">
            <div className="select-none rounded-full bg-slate-100 px-5 py-1 text-[10px] font-bold text-slate-600">
              Default
            </div>
          </div>
          <div
            className="pb-5 pt-6 text-[14px] font-medium leading-normal text-slate-600 sm:pb-32 sm:pt-3"
            ref={normalEndRef}
          >
            {!isSteering && steeredChatMessages.length === 0 && (
              <div className="w-full pl-3 pt-8 text-left text-lg text-slate-400">
                Hey, {`I'm the default ${modelId}!`}
                <div className="mt-6 text-sm text-slate-500">{`I'm the default, non-steered and non-capped model.`}</div>
              </div>
            )}
            <SteerChatMessage chatMessages={defaultChatMessages} steered={false} />
            {isSteering && <LoadingSquare className="px-1.5 py-3" />}
          </div>
        </div>
        <div className="flex h-full max-h-[calc(100dvh-48px)] min-h-[calc(100dvh-48px)] w-full flex-1 flex-col overflow-y-scroll bg-sky-100 px-3 py-2 text-left text-xs text-slate-400 sm:max-h-[calc(100dvh-76px)] sm:min-h-[calc(100dvh-76px)] sm:px-5">
          <div className="sticky top-0.5 flex flex-row justify-center uppercase text-sky-700 sm:top-0">
            <div className="select-none rounded-full bg-sky-100 px-5 py-1 text-[10px] font-bold">
              {steerMethod === NPSteerMethod.ProjectionCap ? 'Capped' : 'Steered'}
            </div>
          </div>
          <div
            className="pb-28 pt-5 text-[14px] font-medium leading-normal text-slate-600 sm:pb-32 sm:pt-3"
            ref={steeredEndRef}
          >
            {!isSteering && steeredChatMessages.length === 0 && (
              <div className="w-full pl-3 pr-3 pt-8 text-left text-lg text-sky-600">
                Hey, {`I'm ${steerMethod === NPSteerMethod.ProjectionCap ? 'capped' : 'steered'} ${modelId}!`}
                {selectedFeatures.length > 0 ? (
                  <div className="mt-5 text-sm text-sky-700">
                    {`Using method "${steerMethod
                      .replaceAll('_', ' ')
                      .toLowerCase()
                      .replace(/\b\w/g, (c) => c.toUpperCase())}":`}
                    {selectedFeatures.slice(0, 5).map((f) => {
                      return (
                        <div key={f.index} className="ml-3 mt-1">
                          - {f.explanation}
                        </div>
                      );
                    })}
                    {selectedFeatures.length > 5 && (
                      <div className="ml-3 mt-1">
                        - and {selectedFeatures.length - 5} other feature
                        {selectedFeatures.length - 5 !== 1 ? 's' : ''}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="mt-6 text-sm text-sky-700">
                    Choose a demo, select a preset, or manually search and add features.
                  </div>
                )}
              </div>
            )}
            <SteerChatMessage chatMessages={steeredChatMessages} steered />
            {isSteering && <LoadingSquare className="px-1.5 py-3" />}
          </div>
        </div>
      </div>
      <div className="-mt-32 flex w-full flex-col items-center justify-center pb-8 sm:-mt-28 sm:pb-4">
        <div className="relative mb-2 mt-2.5 flex flex-row items-center justify-center gap-x-1.5 sm:mt-0 sm:gap-x-3">
          <div className="my-0 flex flex-row gap-x-2 sm:my-0">
            <button
              type="button"
              disabled={defaultChatMessages.length === 0 || isSteering}
              onClick={() => {
                if (defaultChatMessages.length === 0) {
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
              disabled={defaultChatMessages.length === 0 || isSteering}
              onClick={() => {
                if (defaultChatMessages.length === 0) {
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
        <div className="relative flex min-w-[95%] flex-row items-center justify-center sm:min-w-[60%]">
          <ReactTextareaAutosize
            name="searchQuery"
            disabled={isSteering}
            value={typedInText}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !isSteering) {
                sendChat();
                e.preventDefault();
              }
            }}
            onChange={(e) => {
              setTypedInText(e.target.value);
            }}
            required
            placeholder="Ask or say something..."
            className="mt-0 w-full flex-1 resize-none rounded-lg border border-slate-300 px-4 py-3.5 pr-10 text-left text-xs font-medium text-slate-800 placeholder-slate-400 shadow-md transition-all focus:border-slate-300 focus:shadow-lg focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
          />
          <button
            type="button"
            onClick={() => {
              if (!isSteering) {
                sendChat();
              } else {
                stopSteering();
              }
            }}
            className="absolute right-2 flex h-full cursor-pointer items-center justify-center"
          >
            {!isSteering ? (
              <ArrowUp className="h-8 w-8 rounded-full bg-gBlue p-1.5 text-white hover:bg-gBlue/80" />
            ) : (
              <X className="h-8 w-8 rounded-full bg-red-400 p-1.5 text-white hover:bg-red-600" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
