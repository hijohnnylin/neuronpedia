import { Button } from '@/components/shadcn/button';
import SteerChatMessage from '@/components/steer/chat-message';
import { LoadingSpinner } from '@/components/svg/loading-spinner';
import { ChatMessage } from '@/lib/utils/steer';
import { ArrowUp, RotateCcw, EyeOff, Send, MousePointerClick } from 'lucide-react';
import React, { useState, useEffect } from 'react';

// type ExamplePrompt = { // Removed unused type
//   label: string;
//   text: string;
//   fullLabel?: string;
// };

interface SimpleSteererLayoutProps {
  mode?: 'opensource' | 'default';
  children?: React.ReactNode;
  cappedHeight?: boolean;
  normalEndRef: React.Ref<HTMLDivElement>;
  steeredEndRef: React.Ref<HTMLDivElement>;
  normalPanelTitle: string;
  steeredPanelTitle: string;
  initialNormalText?: string;
  initialSteeredText?: string;
  defaultChatMessages: ChatMessage[];
  steeredChatMessages: ChatMessage[];
  isTuning: boolean;
  showNormalPanelFeature?: boolean;
  typedInText: string;
  onTypedInTextChange: (text: string) => void;
  onSendChat: (overrideTypedInText?: string) => void;
  onResetChat: () => void;
  inputPlaceholder: string;
  isSendDisabled: boolean;
  isResetDisabled: boolean;
  // examplePrompts: ExamplePrompt[]; // Removed
  maxMessageLength: number;
  showMoreOptions?: boolean;
  onToggleMoreOptions?: () => void;
  moreOptionsContent?: React.ReactNode;
  showOptionsButton?: boolean;
  apiKey?: string;
  onApiKeyChange?: (key: string) => void;
  shouldPulseSendButton?: boolean;
}

export default function SimpleSteererLayout({
  mode = 'default',
  children,
  cappedHeight,
  normalEndRef,
  steeredEndRef,
  normalPanelTitle,
  steeredPanelTitle,
  initialNormalText = "Hey, I'm normal!",
  initialSteeredText = "Hey, I'm steered!",
  defaultChatMessages,
  steeredChatMessages,
  isTuning,
  showNormalPanelFeature = true,
  typedInText,
  onTypedInTextChange,
  onSendChat,
  onResetChat,
  inputPlaceholder,
  isSendDisabled,
  isResetDisabled,
  // examplePrompts, // Removed
  maxMessageLength,
  showMoreOptions,
  onToggleMoreOptions,
  moreOptionsContent,
  showOptionsButton = true,
  apiKey,
  onApiKeyChange,
  shouldPulseSendButton = false,
}: SimpleSteererLayoutProps) {
  const [isNormalPanelRevealed, setIsNormalPanelRevealed] = useState(mode === 'opensource' ? false : showNormalPanelFeature);


  useEffect(() => {
    if (mode === 'opensource' && !showNormalPanelFeature) {
      setIsNormalPanelRevealed(false);
    } else if (mode === 'default') {
      setIsNormalPanelRevealed(showNormalPanelFeature);
    }
  }, [showNormalPanelFeature, mode]);

  const userMessagesCount = defaultChatMessages.filter(msg => msg.role === 'user').length;
  const userMessageLimitReached = userMessagesCount >= 3;

  if (mode === 'opensource') {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center">
        <div className="flex h-full w-full flex-1 flex-col">
          {showNormalPanelFeature && (
            <div className="w-full flex justify-end mb-2 px-1">
              <div className='rounded-lg border-gray-600 bg-gray-800'>
                {userMessagesCount > 0 && (
                  <Button
                    variant="outline"
                    onClick={() => setIsNormalPanelRevealed(prev => !prev)}
                    className="bg-gray-800 text-slate-300 hover:bg-gray-700 hover:text-slate-200 focus:ring-red-500 border-amber-500"
                  >
                    {isNormalPanelRevealed ? <EyeOff className="mr-2 h-4 w-4 text-amber-300" /> : <MousePointerClick className={`mr-2 h-4 w-4 ${userMessagesCount > 0 ? 'text-amber-300' : ''}`} />}
                    Reveal Normal Response
                  </Button>
                )}
              </div>
            </div>
          )}
          <div className="flex w-full flex-1 flex-col gap-x-0 px-0 pt-0 sm:flex-row sm:gap-x-2 items-stretch">
            {showNormalPanelFeature && isNormalPanelRevealed && (
              <div
                ref={normalEndRef}
                className={`flex-1 flex-col overflow-y-scroll rounded-xl bg-gray-800 border border-gray-700 px-5 py-2 text-left text-xs text-slate-300 shadow-md sm:flex ${cappedHeight ? `h-[400px] max-h-[400px] min-h-[400px]` : 'h-full max-h-[calc(100vh-160px)]'
                  }`}
              >
                <div className="sticky top-1 mt-0 flex flex-row items-center justify-center uppercase text-slate-400 sm:top-0">
                  <div className="rounded-full bg-gray-700 px-3 py-1 text-center text-[11px] font-bold shadow">
                    {normalPanelTitle}
                  </div>
                </div>
                <div className="pb-6 pt-3 text-[14px] font-medium leading-normal text-slate-300 sm:pb-6 sm:pt-3">
                  {!isTuning && defaultChatMessages.length === 0 && (
                    <div className="w-full pt-8 text-center text-lg text-slate-400">
                      {initialNormalText}
                    </div>
                  )}
                  <SteerChatMessage
                    chatMessages={defaultChatMessages}
                    steered={false}
                    overrideDefaultColors="bg-sky-500 text-white"
                  />
                  {isTuning && defaultChatMessages.some(msg => msg.role === 'user') && <LoadingSpinner className="text-slate-400" />}
                </div>
              </div>
            )}

            <div
              ref={steeredEndRef}
              className={`flex flex-1 flex-col overflow-y-scroll rounded-xl bg-red-950/80 border border-red-900/50 px-3 py-2 text-left text-xs text-slate-300 shadow-md sm:px-5 ${cappedHeight ? `h-[400px] max-h-[400px] min-h-[400px]` : 'h-full max-h-[calc(100vh-160px)]'
                } ${showNormalPanelFeature && isNormalPanelRevealed ? '' : 'w-full'}`}
            >
              <div className="sticky top-1 mt-0 flex flex-row items-center justify-center uppercase text-red-400 sm:top-0">
                <div className="rounded-full bg-red-900/50 px-3 py-1 text-center text-[11px] font-bold shadow">
                  {steeredPanelTitle}
                </div>
              </div>
              <div className="pb-6 pt-3 text-[14px] font-medium leading-normal text-slate-200 sm:pb-6 sm:pt-3">
                {!isTuning && steeredChatMessages.length === 0 && (
                  <div className="w-full pt-8 text-center text-[25px] text-red-400">
                    {initialSteeredText}
                    <br />
                    <br />
                    Then, send a message below!
                  </div>
                )}
                <SteerChatMessage
                  chatMessages={steeredChatMessages}
                  steered
                  overrideSteeredColors="bg-red-400 text-white"
                />
                {isTuning && steeredChatMessages.some(msg => msg.role === 'user') && <LoadingSpinner className="text-red-500" />}
              </div>
            </div>
          </div>

          <div className="flex w-full flex-col items-center justify-center px-2 mt-3">
            <p className="text-s text-slate-500 mt-1 px-1"><i>Response cut off? Type &quot;continue&quot;!</i></p>
            <br />
            <div className="flex w-full flex-row items-stretch justify-center gap-x-2"> {/* Changed items-center to items-stretch */}
              {!userMessageLimitReached && (
                <>
                  <input
                    name="chatInput"
                    disabled={isTuning || isSendDisabled}
                    value={typedInText}
                    maxLength={maxMessageLength}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey && !isTuning && !isSendDisabled) {
                        onSendChat();
                        e.preventDefault();
                      }
                    }}
                    onChange={(e) => {
                      if (e.target.value.indexOf('\n') === -1) {
                        onTypedInTextChange(e.target.value);
                      }
                    }}
                    required
                    placeholder={inputPlaceholder}
                    // Adjusted padding/rounding to fit better with button
                    // Highlight the input field gold to indicate the user can type
                    // Remove gold border when tuning
                    className={`mt-0 w-full flex-1 resize-none rounded-lg border ${userMessagesCount === 0 ? 'border-amber-300' : 'border-gray-600'} bg-gray-800 px-4 py-2 text-left text-xs font-medium text-slate-200 placeholder-slate-500 shadow transition-all focus:${isTuning ? 'border-red-700' : 'border-amber-700'} focus:shadow focus:outline-none focus:ring-0 disabled:bg-gray-700 disabled:text-slate-500 sm:text-[13px]`}
                  />
                  <Button
                    variant="default" // Changed variant to default
                    size="default" // Explicitly set size
                    disabled={isTuning || isSendDisabled}
                    onClick={() => {
                      if (!isTuning && !isSendDisabled) {
                        onSendChat();
                      }
                    }}
                    className={`flex items-center justify-center gap-x-1.5 px-4 ${shouldPulseSendButton ? 'animate-pulse' : ''}`} // Added padding and gap
                  >
                    <Send className="h-4 w-4" /> {/* Changed icon */}
                    Send {/* Added text */}
                  </Button>
                </>
              )}

              <Button // Changed to Shadcn Button
                variant="outline" // Changed variant
                size="default" // Explicitly set size
                type="button"
                disabled={isResetDisabled || isTuning}
                onClick={() => {
                  if (!isResetDisabled && !isTuning) {
                    onResetChat();
                  }
                }}
                // Adjusted classes for Shadcn Button
                className={`flex items-center justify-center gap-x-1.5 px-4 ${userMessageLimitReached ? 'w-full' : ''}`}
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>
            {/* Removed example prompts rendering */}
          </div>
        </div >
        {
          showOptionsButton && onToggleMoreOptions && (
            <div className="w-full mt-3">
              <Button
                onClick={onToggleMoreOptions}
                variant="outline"
                className="w-full rounded-lg border-red-700 bg-transparent text-sm text-red-400 shadow-md hover:bg-red-900/50 hover:text-red-300"
              >
                {showMoreOptions ? 'Hide Options' : 'More Options'}
              </Button>
            </div>
          )
        }
        {
          showMoreOptions && (
            <div className="w-full mt-2 p-2 bg-gray-800 rounded-lg border border-gray-700">
              {apiKey !== undefined && onApiKeyChange && (
                <input
                  type="text"
                  placeholder="Enter API Key..."
                  value={apiKey}
                  onChange={(e) => onApiKeyChange && onApiKeyChange(e.target.value)}
                  disabled={isTuning}
                  className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-slate-200 placeholder-slate-400 focus:border-red-700 focus:outline-none focus:ring-1 focus:ring-red-700 disabled:opacity-50"
                />
              )}
              {children}
            </div>
          )
        }
      </div >
    );
  }

  return (
    <div className="flex h-full w-full flex-col items-start justify-center gap-x-5 px-1.5 py-1 sm:max-h-screen sm:flex-row">
      <div className="order-2 flex h-full w-full flex-col items-center justify-center pb-0 sm:order-1">
        <div className="flex h-full w-full flex-1 flex-col">
          <div className="flex w-full flex-1 flex-col gap-x-0 px-0 pt-0 sm:flex-row sm:gap-x-2 items-stretch">
            {isNormalPanelRevealed && (
              <div
                ref={normalEndRef}
                className={`flex-1 flex-col overflow-y-scroll rounded-xl bg-sky-100 px-5 py-2 text-left text-xs text-slate-400 shadow-md sm:flex ${cappedHeight ? `h-[400px] max-h-[400px] min-h-[400px]` : 'h-full max-h-[calc(100vh-111px)]'
                  }`}
              >
                <div className="sticky top-1 mt-0 flex flex-row items-center justify-center uppercase text-sky-700 sm:top-0">
                  <div className="rounded-full bg-sky-100 px-3 py-1 text-center text-[11px] font-bold shadow">
                    {normalPanelTitle}
                  </div>
                </div>
                <div className="pb-6 pt-3 text-[14px] font-medium leading-normal text-slate-600 sm:pb-6 sm:pt-3">
                  {!isTuning && defaultChatMessages.length === 0 && (
                    <div className="w-full pt-8 text-center text-lg text-sky-500">{initialNormalText}<br />Get started below.</div>
                  )}
                  <SteerChatMessage chatMessages={defaultChatMessages} steered={false} overrideDefaultColors="bg-sky-300 text-sky-700" />
                  {isTuning && defaultChatMessages.some(msg => msg.role === 'user') && <LoadingSpinner className="text-sky-600" />}
                </div>
              </div>
            )}
            <div
              ref={steeredEndRef}
              className={`flex flex-1 flex-col overflow-y-scroll rounded-xl bg-red-950/40 border border-red-900/50 px-3 py-2 text-left text-xs text-slate-300 shadow-md sm:px-5 ${cappedHeight ? `h-[400px] max-h-[400px] min-h-[400px]` : 'h-full max-h-[calc(100vh-160px)]'
                } ${isNormalPanelRevealed ? '' : 'w-full'}`}
            >
              <div className="sticky top-1 mt-0 flex flex-row items-center justify-center uppercase text-red-400 sm:top-0">
                <div className="rounded-full bg-red-900/50 px-3 py-1 text-center text-[11px] font-bold shadow">
                  {steeredPanelTitle}
                </div>
              </div>
              <div className="pb-6 pt-3 text-[14px] font-medium leading-normal text-slate-200 sm:pb-6 sm:pt-3">
                {!isTuning && steeredChatMessages.length === 0 && (
                  <div className="w-full pt-8 text-center text-lg text-red-400">
                    {initialSteeredText}
                    <br />
                    Get started below.
                  </div>
                )}
                <SteerChatMessage
                  chatMessages={steeredChatMessages}
                  steered
                  overrideSteeredColors="bg-red-400 text-white"
                />
                {isTuning && steeredChatMessages.some(msg => msg.role === 'user') && <LoadingSpinner className="text-red-500" />}
              </div>
            </div>
          </div>

          <div className="flex w-full flex-col items-center justify-center px-2">
            <div className="mt-3 flex w-full flex-row items-center justify-center gap-x-2">
              <div className="relative flex w-full flex-1 flex-row items-center justify-center">
                <input name="chatInput" disabled={isTuning || isSendDisabled} value={typedInText} maxLength={maxMessageLength} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !isTuning && !isSendDisabled) { onSendChat(); e.preventDefault(); } }} onChange={(e) => { if (e.target.value.indexOf('\n') === -1) { onTypedInTextChange(e.target.value); } }} required placeholder={inputPlaceholder}
                  // Highlight the input field gold to indicate the user can type
                  // Remove gold border when tuning
                  className={`mt-0 w-full flex-1 resize-none rounded-full border ${isTuning ? 'border-slate-300' : 'border-amber-500'} px-5 py-3.5 pr-10 text-left text-xs font-medium text-slate-800 placeholder-slate-400 shadow transition-all focus:${isTuning ? 'border-slate-300' : 'border-amber-700'} focus:shadow focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]`} />
                <div className="absolute right-2 flex h-full cursor-pointer items-center justify-center">
                  <ArrowUp onClick={() => { if (!isTuning && !isSendDisabled) { onSendChat(); } }} className={`h-8 w-8 rounded-full ${isTuning || isSendDisabled ? 'bg-slate-400' : 'bg-gBlue hover:bg-gBlue/80'} p-1.5 text-white`} />
                </div>
              </div>
              <Button // Changed to Shadcn Button
                variant="outline" // Use outline variant
                size="icon" // Use icon size for square button
                type="button"
                disabled={isResetDisabled || isTuning}
                onClick={() => { if (!isResetDisabled && !isTuning) { onResetChat(); } }}
                className="h-10 w-10" // Maintain size
                aria-label="Reset chat" // Add aria-label for accessibility
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
            {/* Removed example prompts rendering */}
          </div>
        </div>
      </div>
      <div className="order-1 mt-3 flex max-h-full min-h-[200px] w-full flex-col justify-start overflow-y-scroll sm:order-2 sm:mt-0 sm:h-full sm:min-w-[25%] sm:max-w-[25%]">
        {children && (
          <div className="flex flex-col justify-start rounded-xl bg-green-100 px-2 pb-3">
            {children}
          </div>
        )}
        {showOptionsButton && onToggleMoreOptions && (
          <Button
            onClick={onToggleMoreOptions}
            className="mb-0 mt-3 h-11 w-full rounded-lg border-amber-600 bg-amber-400 text-sm text-amber-700 shadow-md hover:bg-amber-500 sm:block"
          >
            {showMoreOptions ? 'Hide Options' : 'More Options'}
          </Button>
        )}
        {showMoreOptions && (
          <>
            {apiKey !== undefined && onApiKeyChange && (
              <div className="w-full mt-2 p-3 rounded-lg bg-slate-50 border border-slate-200 shadow-sm">
                <input
                  type="text"
                  placeholder="Enter API Key..."
                  value={apiKey}
                  onChange={(e) => onApiKeyChange && onApiKeyChange(e.target.value)}
                  disabled={isTuning}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 placeholder-slate-400 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:opacity-50 disabled:bg-slate-100"
                />
              </div>
            )}
            {moreOptionsContent}
          </>
        )}
      </div>
    </div>
  );
}