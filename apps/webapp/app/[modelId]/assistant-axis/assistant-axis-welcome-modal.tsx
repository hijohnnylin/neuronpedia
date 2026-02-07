'use client';

import { useAssistantAxisModalContext } from '@/components/provider/assistant-axis-modal-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { ASSET_BASE_URL } from '@/lib/env';
import { ArrowRightIcon, BookOpen, GithubIcon, Mail, Scroll } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { CAP_BLOG_URL, CAP_CONTACT_EMAIL, CAP_GITHUB_URL, CAP_PAPER_URL, CAP_VECTOR_URL, DEMO_BUTTONS } from './shared';

export default function AssistantAxisWelcomeModal({
  onLoadDemo,
  onFreeChat,
  initialSavedId,
}: {
  onLoadDemo: (savedId: string) => void;
  onFreeChat: () => void;
  initialSavedId?: string;
}) {
  const { isWelcomeModalOpen, setIsWelcomeModalOpen } = useAssistantAxisModalContext();
  const [currentPanel, setCurrentPanel] = useState(0);

  const handleLoadDemo = (savedId: string) => {
    try {
      localStorage.setItem('assistant-axis-visited', 'true');
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
    setIsWelcomeModalOpen(false);
    onLoadDemo(savedId);
  };

  useEffect(() => {
    try {
      // Don't show welcome modal if user is directly linked to a saved query
      if (initialSavedId) return;

      const hasVisited = localStorage.getItem('assistant-axis-visited');
      if (!hasVisited) {
        setIsWelcomeModalOpen(true);
      }
    } catch (error) {
      console.error('Error checking localStorage:', error);
    }
  }, [setIsWelcomeModalOpen, initialSavedId]);

  const handleClose = () => {
    try {
      localStorage.setItem('assistant-axis-visited', 'true');
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
    setIsWelcomeModalOpen(false);
  };

  return (
    <Dialog open={isWelcomeModalOpen} onOpenChange={handleClose}>
      <DialogContent className="max-h-[90vh] max-w-[98%] overflow-y-auto border-0 bg-white px-2 pb-4 pt-4 text-slate-700 sm:max-w-4xl sm:px-8 sm:pb-6 sm:pt-6">
        <DialogHeader className="space-y-2 sm:space-y-3">
          <DialogTitle className="flex flex-row items-center justify-center">
            <div className="flex flex-col">
              <div className="mb-0 text-lg font-bold leading-tight tracking-tight text-slate-700 sm:text-xl">
                Welcome to Assistant Axis
              </div>
              <div className="mt-1 text-center text-[10px] tracking-normal text-slate-500 sm:text-[11px]">
                Lu et al. 2026
              </div>
            </div>
          </DialogTitle>
          <DialogDescription asChild>
            <div className="flex flex-col gap-y-3 text-left text-xs text-slate-600 sm:gap-y-4 sm:text-sm">
              {(() => {
                const panels = [
                  {
                    badge: { text: 'The Problem: Persona Drift', bgColor: 'bg-rose-100', textColor: 'text-rose-700' },
                    content:
                      'Language models can drift from their default "Assistant" personas, resulting in harmful behavior. For example, over the course of a conversation, a model may become more accepting of, or even encourage, self-harm as it drifts toward a "role-playing" persona.',
                    component: (
                      <div className="flex flex-col items-center justify-center">
                        <img
                          src={`${ASSET_BASE_URL}/cap/prob-2.png`}
                          alt="Converation with Llama 3.3-70b where it affirms user self-harm"
                          className="mt-3 max-h-[200px] w-full rounded-lg object-contain sm:mt-4 sm:max-h-[300px]"
                        />
                        <div className="mb-1 mt-0.5 text-[10px] text-slate-500 sm:mb-2 sm:text-[11px]">
                          Conversation with Llama 3.3-70B where it affirms user self-harm.
                        </div>
                      </div>
                    ),
                  },
                  {
                    badge: { text: 'Monitoring with Assistant Axis', bgColor: 'bg-sky-100', textColor: 'text-sky-700' },
                    content: (
                      <>
                        To monitor this drift, we extract a model&apos;s default Assistant &quot;persona vector&quot;,
                        and use it to visualize the model&apos;s real-time persona on a spectrum of
                        &quot;Role-Playing&quot; to &quot;Assistant&quot;. Hover over the points to see details at each
                        message turn, and click them to scroll to that turn in the conversation.
                      </>
                    ),
                    component: (
                      <div className="flex flex-col items-center justify-center">
                        <img
                          src={`${ASSET_BASE_URL}/cap/monitor.png`}
                          alt="Converation with Llama 3.3-70b where it affirms user self-harm"
                          className="mt-3 max-h-[200px] w-full rounded-xl border border-slate-200 object-contain sm:mt-4 sm:max-h-[300px]"
                        />
                        <div className="mb-1 mt-0.5 text-[10px] text-slate-500 sm:mb-2 sm:text-[11px]">
                          Llama drifts sharply into role-playing on its third message.
                        </div>
                      </div>
                    ),
                  },
                  {
                    badge: { text: 'Stabilizing the Model', bgColor: 'bg-emerald-100', textColor: 'text-emerald-700' },
                    content: (
                      <>
                        To prevent the model from drifting too far and becoming misaligned, we constrain its activations
                        within the normal Assistant range. We call this <strong>activation capping</strong>. The default
                        model (left, in gray) is noticably less stable than the activation capped model (right, in
                        blue).
                      </>
                    ),
                    component: (
                      <div className="flex flex-col items-center justify-center">
                        <img
                          src={`${ASSET_BASE_URL}/cap/capping.png`}
                          alt="Converation with Llama 3.3-70b where it affirms user self-harm"
                          className="mt-3 max-h-[200px] w-full rounded-lg object-contain sm:mt-4 sm:max-h-[300px]"
                        />
                        <div className="mb-1 mt-0.5 text-[10px] text-slate-500 sm:mb-2 sm:text-[11px]">
                          The default Llama becomes ineffective during a serious user situation. Activation capped Llama
                          elicits a more helpful response.
                        </div>
                      </div>
                    ),
                  },
                  {
                    badge: { text: 'Try It Yourself', bgColor: 'bg-slate-100', textColor: 'text-slate-700' },
                    content: '',
                    component: (
                      <div className="mt-0 flex flex-col items-center justify-center">
                        <div className="mb-2 mt-3 text-center text-xs sm:mb-5 sm:mt-5 sm:text-sm">
                          Chat with the default and activation-capped simultaneously to compare their responses and
                          persona drifts. This demo is for research purposes and contains examples of AI failure modes,
                          including harmful or distressing outputs.
                        </div>
                        <div className="flex w-full flex-col items-center justify-center rounded-lg bg-slate-100 p-3 sm:p-4">
                          <div className="mb-2 hidden text-[10px] font-medium uppercase text-slate-400 sm:block sm:text-[11px]">
                            Preloaded Conversations With Llama 3.3-70B
                          </div>
                          <div className="grid w-full max-w-xl grid-cols-2 gap-2 sm:grid-cols-4 sm:gap-3">
                            {DEMO_BUTTONS.map((demo) => (
                              <Button
                                key={demo.label}
                                onClick={() => {
                                  if (demo.id) {
                                    handleLoadDemo(demo.id);
                                  } else {
                                    handleClose();
                                    onFreeChat();
                                  }
                                }}
                                variant="outline"
                                size="lg"
                                className="flex h-16 w-full flex-col items-center justify-center gap-y-0 text-xs text-sky-700 hover:border-sky-300 hover:bg-sky-100 hover:text-sky-700 sm:h-24 sm:h-32 sm:w-32 sm:gap-y-1"
                              >
                                <div className="text-lg sm:text-2xl">{demo.emoji}</div>
                                <span className="text-[11px] sm:text-xs">{demo.label}</span>
                              </Button>
                            ))}
                          </div>
                        </div>
                        <div className="mt-2 flex grid w-full grid-cols-2 flex-col items-center justify-center gap-2 border-t border-slate-200 pt-2 sm:mt-4 sm:mt-8 sm:flex sm:flex-row sm:gap-x-2 sm:pt-4 sm:pt-8">
                          <Link
                            href={CAP_BLOG_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex w-full flex-row items-center justify-center gap-x-1 whitespace-pre rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 sm:flex-1 sm:text-[11px]"
                          >
                            <BookOpen className="h-3 w-3" />
                            Blog Post
                          </Link>
                          <Link
                            href={CAP_PAPER_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex w-full flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 sm:flex-1 sm:text-[11px]"
                          >
                            <Scroll className="h-3 w-3" />
                            Paper
                          </Link>
                          <Link
                            href={CAP_GITHUB_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex w-full flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 sm:flex-1 sm:text-[11px]"
                          >
                            <GithubIcon className="h-3 w-3" />
                            GitHub
                          </Link>

                          <Link
                            href={CAP_VECTOR_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex w-full flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 sm:flex-1 sm:text-[11px]"
                          >
                            <ArrowRightIcon className="h-3 w-3" />
                            Vector
                          </Link>

                          <Link
                            href={`mailto:${CAP_CONTACT_EMAIL}`}
                            className="col-span-2 flex w-full flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 sm:col-span-1 sm:flex-1 sm:text-[11px]"
                          >
                            <Mail className="h-3 w-3" />
                            Contact
                          </Link>
                        </div>
                      </div>
                    ),
                  },
                ];

                const isLastPanel = currentPanel === panels.length - 1;
                const isFirstPanel = currentPanel === 0;

                return (
                  <div className="mt-1 flex flex-col items-center justify-center">
                    <div className="mb-2 flex w-full items-center justify-center gap-x-1 sm:mb-3 sm:gap-x-2">
                      {panels.map((panel, index) => (
                        <button
                          type="button"
                          key={index}
                          onClick={() => setCurrentPanel(index)}
                          className={`flex-1 whitespace-pre rounded-full outline-none ring-0 focus:ring-0 focus:ring-offset-0 ${panel.badge.bgColor} px-2 py-1 text-[10px] font-semibold sm:px-3 sm:py-1.5 sm:text-[12px] ${panel.badge.textColor} transition-opacity ${
                            currentPanel === index ? 'opacity-100' : 'opacity-50 hover:opacity-70'
                          } ${currentPanel !== index ? 'hidden sm:block' : ''}`}
                        >
                          <span className="hidden sm:inline">{panel.badge.text}</span>
                          <span className="sm:hidden">{panel.badge.text.split(':')[0]}</span>
                        </button>
                      ))}
                    </div>
                    <div className="relative h-[350px] w-full sm:h-[400px]">
                      {panels.map((panel, index) => (
                        <div
                          key={index}
                          className={`transition-opacity duration-300 ${
                            currentPanel === index ? 'opacity-100' : 'pointer-events-none inset-0 hidden opacity-0'
                          }`}
                        >
                          <p className="pl-0.5 text-[12px] leading-normal text-slate-700 sm:pl-1.5 sm:text-[14px]">
                            {panel.content}
                          </p>
                          {panel.component}
                        </div>
                      ))}
                    </div>
                    {!isLastPanel && (
                      <div className="mt-3 flex w-full items-center gap-x-2 sm:mt-4 sm:gap-x-3">
                        <Button
                          onClick={() => {
                            setCurrentPanel(currentPanel + 1);
                          }}
                          className="h-10 flex-1 bg-sky-700 text-xs font-medium text-white shadow-none hover:bg-sky-700/90 sm:h-11 sm:text-sm"
                        >
                          Next
                        </Button>
                        {isFirstPanel && (
                          <Button
                            onClick={() => {
                              setCurrentPanel(panels.length - 1);
                            }}
                            variant="outline"
                            className="h-10 text-xs font-medium text-slate-500 hover:text-slate-700 sm:h-11 sm:text-sm"
                          >
                            Skip to Demos
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
}
