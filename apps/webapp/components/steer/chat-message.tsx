import { ChatMessage } from '@/lib/utils/steer';
import { RefObject } from 'react';

export default function SteerChatMessage({
  chatMessages,
  steered,
  overrideSteeredColors,
  overrideDefaultColors,
  overrideTextSize,
  messageRefs,
}: {
  chatMessages: ChatMessage[];
  steered: boolean;
  overrideSteeredColors?: string;
  overrideDefaultColors?: string;
  overrideTextSize?: string;
  messageRefs?: RefObject<(HTMLDivElement | null)[]>;
}) {
  return (
    <div className="flex flex-col gap-y-2 sm:gap-y-3">
      {chatMessages.map((s, i) => {
        if (s.role === 'user') {
          return (
            <div
              className="flex w-full justify-end"
              key={i}
              ref={(el) => {
                if (messageRefs?.current) {
                  // eslint-disable-next-line no-param-reassign
                  messageRefs.current[i] = el;
                }
              }}
            >
              <div
                className={`max-w-[85%] whitespace-pre-wrap rounded-lg bg-slate-500 px-3 py-2 ${overrideTextSize || 'text-[12.5px]'} text-white sm:max-w-[90%] sm:rounded-xl sm:px-3 sm:py-2 sm:text-[12.5px]`}
              >
                {s.content}
              </div>
            </div>
          );
        }
        if (s.role === 'model' || s.role === 'assistant') {
          return (
            <div
              className="flex w-full justify-start"
              key={i}
              ref={(el) => {
                if (messageRefs?.current) {
                  // eslint-disable-next-line no-param-reassign
                  messageRefs.current[i] = el;
                }
              }}
            >
              <div
                className={`max-w-[85%] rounded-lg sm:max-w-[90%] sm:rounded-xl ${
                  steered
                    ? overrideSteeredColors || 'bg-sky-300 text-sky-700'
                    : overrideDefaultColors || 'bg-slate-300 text-slate-700'
                } whitespace-pre-wrap break-words px-3 py-2 ${overrideTextSize || 'text-[12.5px]'} sm:text-[12.5px]`}
              >
                {s.content.startsWith('<think>') ? (
                  <>
                    {s.content.split('<think>').map((part, index) => {
                      if (index === 0) return part;
                      const [thinkContent, remainingContent] = part.split('</think>');
                      return (
                        <>
                          <div className="-mx-3 -mt-2 mb-2 bg-white/25 px-3 py-2 text-[11.5px]">
                            <div
                              className={`pb-1.5 pt-0.5 text-left text-[8px] font-bold leading-none ${
                                steered ? 'text-sky-700/50' : 'text-slate-700/50'
                              }`}
                            >
                              THINKING
                            </div>
                            {thinkContent ? thinkContent.trim() : ''}
                          </div>
                          {remainingContent ? remainingContent.trim() : ''}
                        </>
                      );
                    })}
                  </>
                ) : (
                  s.content
                )}
              </div>
            </div>
          );
        }
        return '';
      })}
    </div>
  );
}
