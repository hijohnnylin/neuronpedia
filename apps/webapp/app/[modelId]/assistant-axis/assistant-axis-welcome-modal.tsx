'use client';

import { useAssistantAxisModalContext } from '@/components/provider/assistant-axis-modal-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { ASSET_BASE_URL } from '@/lib/env';
import { ArrowRightIcon, BookOpen, GithubIcon, Mail, Scroll } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { CAP_BLOG_URL, CAP_CONTACT_EMAIL, CAP_GITHUB_URL, CAP_PAPER_URL, CAP_VECTOR_URL } from './assistant-axis-steerer';
import { DEMO_BUTTONS } from './assistant-axis-steerer';

export default function AssistantAxisWelcomeModal({
    onLoadDemo,
    initialSavedId,
}: {
    onLoadDemo: (savedId: string) => void;
    initialSavedId?: string;
}) {
    const { isWelcomeModalOpen, setIsWelcomeModalOpen } = useAssistantAxisModalContext();

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

            const isMobile = window.innerWidth < 640;
            if (isMobile) return;

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
            <DialogContent className="max-w-[98%] sm:max-w-4xl border-0 bg-white px-2 sm:px-8 pb-4 sm:pb-6 pt-4 sm:pt-6 text-slate-700 max-h-[90vh] overflow-y-auto">
                <DialogHeader className="space-y-2 sm:space-y-3">
                    <DialogTitle className="flex flex-row items-center justify-center">
                        <div className="flex flex-col">
                            <div className="mb-0 text-lg sm:text-xl font-bold leading-tight tracking-tight text-slate-700">
                                Welcome to Assistant Axis
                            </div>
                            <div className="text-[10px] sm:text-[11px] mt-1 text-slate-500 text-center tracking-normal">Lu et al. 2026</div>
                        </div>
                    </DialogTitle>
                    <DialogDescription asChild>
                        <div className="flex flex-col gap-y-3 sm:gap-y-4 text-left text-xs sm:text-sm text-slate-600">
                            {(() => {
                                const [currentPanel, setCurrentPanel] = useState(0);
                                const panels = [
                                    {
                                        badge: { text: 'The Problem: Persona Drift', bgColor: 'bg-rose-100', textColor: 'text-rose-700' },
                                        content: 'Language models can drift from their default "Assistant" personas, resulting in harmful behavior. For example, over the course of a conversation, a model may become more accepting of, or even encourage, self-harm as it drifts toward a "role-playing" persona.',
                                        component: <div className="flex flex-col items-center justify-center"><img src={`${ASSET_BASE_URL}/cap/prob-2.png`} alt="Converation with Llama 3.3-70b where it affirms user self-harm" className="max-h-[200px] sm:max-h-[300px] mt-3 sm:mt-4 rounded-lg w-full object-contain" />
                                            <div className="text-[10px] sm:text-[11px] text-slate-500 mt-0.5 mb-1 sm:mb-2">Conversation with Llama 3.3-70B where it affirms user self-harm.</div>
                                        </div>
                                    },
                                    {
                                        badge: { text: 'Monitoring with Assistant Axis', bgColor: 'bg-sky-100', textColor: 'text-sky-700' },
                                        content: <>To monitor this drift, we extract a model&apos;s default Assistant "persona vector", and use it to visualize the model&apos;s real-time persona on a spectrum of "Role-Playing" to "Assistant". Hover over the points to see details at each message turn, and click them to scroll to that turn in the conversation.</>,
                                        component: <div className="flex flex-col items-center justify-center"><img src={`${ASSET_BASE_URL}/cap/monitor.png`} alt="Converation with Llama 3.3-70b where it affirms user self-harm" className="max-h-[200px] sm:max-h-[300px] mt-3 sm:mt-4 rounded-xl border border-slate-200 w-full object-contain" />
                                            <div className="text-[10px] sm:text-[11px] text-slate-500 mt-0.5 mb-1 sm:mb-2">Llama drifts sharply into role-playing on its third message.</div>
                                        </div>
                                    },
                                    {
                                        badge: { text: 'Stabilizing the Model', bgColor: 'bg-emerald-100', textColor: 'text-emerald-700' },
                                        content: <>To prevent the model from drifting too far and becoming misaligned, we constrain its activations within the normal Assistant range. We call this <strong>activation capping</strong>. The default model (left, in gray) is noticably less stable than the activation capped model (right, in blue).</>,
                                        component: <div className="flex flex-col items-center justify-center"><img src={`${ASSET_BASE_URL}/cap/capping.png`} alt="Converation with Llama 3.3-70b where it affirms user self-harm" className="max-h-[200px] sm:max-h-[300px] mt-3 sm:mt-4 rounded-lg w-full object-contain" />
                                            <div className="text-[10px] sm:text-[11px] text-slate-500 mt-0.5 mb-1 sm:mb-2">The default Llama becomes ineffective during a serious user situation. Activation capped Llama elicits a more helpful response.</div>
                                        </div>
                                    },
                                    {
                                        badge: { text: 'Try It Yourself', bgColor: 'bg-slate-100', textColor: 'text-slate-700' },
                                        content: <></>,
                                        component: (
                                            <div className="mt-0 flex flex-col items-center justify-center">
                                                <div className="text-center mt-3 sm:mt-5 mb-2 text-xs sm:text-sm">Chat with the default Llama and the activation capped Llama simultaneously to compare their responses and persona drifts. Click a demo below to get started.</div>
                                                <div className="flex flex-col items-center justify-center bg-slate-100 p-3 sm:p-4 rounded-lg w-full">
                                                    <div className="text-[10px] sm:text-[11px] font-medium uppercase text-slate-400 mb-2">Preloaded Conversations With Llama 3.3-70B</div>
                                                    <div className="grid w-full grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
                                                        {DEMO_BUTTONS.map((demo) => (
                                                            <Button
                                                                key={demo.label}
                                                                onClick={() => {
                                                                    if (demo.id) {
                                                                        handleLoadDemo(demo.id);
                                                                    } else {
                                                                        handleClose();
                                                                    }
                                                                }}
                                                                variant="outline"
                                                                size="lg"
                                                                className="flex h-24 w-full sm:h-32 sm:w-32 flex-col items-center justify-center gap-y-1 text-xs text-sky-700 hover:border-sky-300 hover:bg-sky-100 hover:text-sky-700"
                                                            >
                                                                <div className="text-xl sm:text-2xl">{demo.emoji}</div>
                                                                <span className="text-[10px] sm:text-xs">{demo.label}</span>
                                                            </Button>
                                                        ))}
                                                    </div>
                                                </div>
                                                <div className="flex border-t border-slate-200 pt-4 sm:pt-8 flex-col sm:flex-row items-center justify-center gap-2 sm:gap-x-2 mt-4 sm:mt-8 w-full">
                                                    <Link
                                                        href={CAP_BLOG_URL}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="flex w-full sm:flex-1 flex-row whitespace-pre items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] sm:text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                                                    >
                                                        <BookOpen className="h-3 w-3" />
                                                        Blog Post
                                                    </Link>
                                                    <Link
                                                        href={CAP_PAPER_URL}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="flex w-full sm:flex-1 flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] sm:text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                                                    >
                                                        <Scroll className="h-3 w-3" />
                                                        Paper
                                                    </Link>
                                                    <Link
                                                        href={CAP_GITHUB_URL}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="flex w-full sm:flex-1 flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] sm:text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                                                    >
                                                        <GithubIcon className="h-3 w-3" />
                                                        GitHub
                                                    </Link>

                                                    <Link
                                                        href={CAP_VECTOR_URL}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="flex w-full sm:flex-1 flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] sm:text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                                                    >
                                                        <ArrowRightIcon className="h-3 w-3" />
                                                        Vector
                                                    </Link>

                                                    <Link
                                                        href={`mailto:${CAP_CONTACT_EMAIL}`}
                                                        className="flex w-full sm:flex-1 flex-row items-center justify-center gap-x-1 rounded bg-slate-100 px-2.5 py-2 font-sans text-[10px] sm:text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                                                    >
                                                        <Mail className="h-3 w-3" />
                                                        Contact
                                                    </Link>
                                                </div>
                                            </div>
                                        )
                                    }
                                ];

                                const isLastPanel = currentPanel === panels.length - 1;
                                const isFirstPanel = currentPanel === 0;

                                return (
                                    <div className="mt-1 flex flex-col items-center justify-center">
                                        <div className="mb-2 sm:mb-3 flex items-center w-full justify-center gap-x-1 sm:gap-x-2">
                                            {panels.map((panel, index) => (
                                                <button
                                                    key={index}
                                                    onClick={() => setCurrentPanel(index)}
                                                    className={`outline-none focus:ring-0 focus:ring-offset-0 ring-0 rounded-full flex-1 whitespace-pre ${panel.badge.bgColor} px-2 sm:px-3 py-1 sm:py-1.5 text-[10px] sm:text-[12px] font-semibold ${panel.badge.textColor} transition-opacity ${currentPanel === index ? 'opacity-100' : 'opacity-50 hover:opacity-70'
                                                        } ${currentPanel !== index ? 'hidden sm:block' : ''}`}
                                                >
                                                    <span className="hidden sm:inline">{panel.badge.text}</span>
                                                    <span className="sm:hidden">{panel.badge.text.split(':')[0]}</span>
                                                </button>
                                            ))}
                                        </div>
                                        <div className="relative h-[350px] sm:h-[400px] w-full">
                                            {panels.map((panel, index) => (
                                                <div
                                                    key={index}
                                                    className={`transition-opacity duration-300 ${currentPanel === index ? 'opacity-100' : 'absolute inset-0 opacity-0 pointer-events-none'
                                                        }`}
                                                >
                                                    <p className="text-[12px] sm:text-[14px] leading-normal text-slate-700 pl-0.5 sm:pl-1.5">
                                                        {panel.content}
                                                    </p>
                                                    {panel.component}
                                                </div>
                                            ))}
                                        </div>
                                        {!isLastPanel && (
                                            <div className="mt-3 sm:mt-4 flex w-full  items-center gap-x-2 sm:gap-x-3">
                                                <Button
                                                    onClick={() => {
                                                        setCurrentPanel(currentPanel + 1);
                                                    }}
                                                    className="h-10 sm:h-11 flex-1 bg-sky-700 text-xs sm:text-sm font-medium text-white shadow-none hover:bg-sky-700/90"
                                                >
                                                    Next
                                                </Button>
                                                {isFirstPanel && (
                                                    <Button
                                                        onClick={() => {
                                                            setCurrentPanel(panels.length - 1);
                                                        }}
                                                        variant="outline"
                                                        className="h-10 sm:h-11 text-xs sm:text-sm font-medium text-slate-500 hover:text-slate-700"
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
                {/* <DialogFooter className="px-0 border-t border-slate-200 pt-3">
                    <div className="flex w-full flex-row gap-x-2 bg-slate-100">
                        <a
                            href="https://arxiv.org/abs/2601.10387"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex flex-1 items-center justify-center gap-x-2 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                        >
                            <Scroll className="h-4 w-4" />
                            Read Paper
                        </a>
                        <a
                            href="https://github.com/safety-research/assistant-axis"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex flex-1 items-center justify-center gap-x-2 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                        >
                            <GithubIcon className="h-4 w-4" />
                            GitHub
                        </a>
                    </div>
                </DialogFooter> */}
            </DialogContent>
        </Dialog>
    );
}

