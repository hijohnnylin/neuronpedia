'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { Input } from '@/components/shadcn/input';
import { Label } from '@/components/shadcn/label';
import { LoadingSquare } from '@/components/svg/loading-square';
import {
  getEstimatedTimeFromNumTokens,
  GRAPH_DESIREDLOGITPROB_DEFAULT,
  GRAPH_DESIREDLOGITPROB_MAX,
  GRAPH_DESIREDLOGITPROB_MIN,
  GRAPH_EDGETHRESHOLD_DEFAULT,
  GRAPH_EDGETHRESHOLD_MAX,
  GRAPH_EDGETHRESHOLD_MIN,
  GRAPH_GENERATION_ENABLED_MODELS,
  GRAPH_MAX_PROMPT_LENGTH_CHARS,
  GRAPH_MAX_TOKENS,
  GRAPH_MAXFEATURENODES_DEFAULT,
  GRAPH_MAXFEATURENODES_MAX,
  GRAPH_MAXFEATURENODES_MIN,
  GRAPH_MAXNLOGITS_DEFAULT,
  GRAPH_MAXNLOGITS_MAX,
  GRAPH_MAXNLOGITS_MIN,
  GRAPH_NODETHRESHOLD_DEFAULT,
  GRAPH_NODETHRESHOLD_MAX,
  GRAPH_NODETHRESHOLD_MIN,
  graphGenerateSchemaClient,
  GraphTokenizeResponse,
  RUNPOD_BUSY_ERROR,
} from '@/lib/utils/graph';
import * as RadixSelect from '@radix-ui/react-select';
import * as RadixSlider from '@radix-ui/react-slider';
import { Form, Formik, FormikHelpers, FormikProps } from 'formik';
import _ from 'lodash';
import { ChevronDownIcon, ChevronUpIcon, X } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';

import { ChatMessage } from '@/lib/utils/steer';

const BORING_TOKENS = ['<strong>', '<em>', '<code>', '<b>', '<i>', '<think>', '</think>', '<|im_start|>', '<|im_end|>'];
const BORING_SYMBOLS = ['*', '**', '`', '```', '-', '–', '—', '_', '__', '~', '='];

const isBoringToken = (token: string) => {
  const trimmedToken = token.trim();
  return BORING_TOKENS.includes(trimmedToken) || BORING_SYMBOLS.includes(trimmedToken);
};

interface FormValues {
  prompt: string;
  modelId: string;
  maxNLogits: number;
  desiredLogitProb: number;
  nodeThreshold: number;
  edgeThreshold: number;
  maxFeatureNodes: number;
  sourceSetName: string;
  slug: string;
}

interface GenerateGraphResponse {
  message: string;
  s3url: string;
  url: string;
  numNodes: number;
  numLinks: number;
}

// Helper component to observe Formik values and trigger tokenization
const FormikValuesObserver: React.FC<{
  prompt: string;
  modelId: string;
  sourceSetName: string;
  maxNLogits: number;
  desiredLogitProb: number;
  debouncedTokenize: (
    modelId: string,
    prompt: string,
    sourceSetName: string,
    maxNLogits: number,
    desiredLogitProb: number,
  ) => void;
  // eslint-disable-next-line
}> = ({ prompt, modelId, sourceSetName, maxNLogits, desiredLogitProb, debouncedTokenize }) => {
  useEffect(() => {
    debouncedTokenize(modelId, prompt, sourceSetName, maxNLogits, desiredLogitProb);
  }, [prompt, modelId, maxNLogits, desiredLogitProb, debouncedTokenize]);

  return null;
};

// Helper function to format seconds into "X min, Y sec"
const formatCountdown = (totalSeconds: number): string => {
  // eslint-disable-next-line
  if (totalSeconds < 0) totalSeconds = 0;
  const m = Math.floor(totalSeconds / 60);
  const s = Math.round(totalSeconds % 60);

  if (m > 0) {
    return `~${m} min, ${s} sec`;
  }
  if (s === 0 && m === 0 && totalSeconds > 0) {
    // Handle cases like 0.5s rounding to 1s, but ensure it's not showing 0s for longer
    return '~1 sec';
  }
  return `~${s} sec`;
};

export default function GenerateGraphModal({ showGenerateModal }: { showGenerateModal: boolean }) {
  const { isGenerateGraphModalOpen, setIsGenerateGraphModalOpen, generateGraphModalPrompt } = useGraphModalContext();
  const { selectedModelId, selectedSourceSetName } = useGraphContext();
  const { globalModels } = useGlobalContext();
  const [generationResult, setGenerationResult] = useState<GenerateGraphResponse | null>(null);
  const [graphTokenizeResponse, setGraphTokenizeResponse] = useState<GraphTokenizeResponse | null>(null);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTokenizing, setIsTokenizing] = useState(false);
  const [endsWithSpace, setEndsWithSpace] = useState(false);
  const [error, setError] = useState<React.ReactNode | null>(null);
  const [countdownTime, setCountdownTime] = useState<number | null>(null);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [chatPrompts, setChatPrompts] = useState<ChatMessage[]>([]);

  const session = useSession();
  const { setSignInModalOpen, getHasGraphsSourceSetsForModelId } = useGlobalContext();
  const formikRef = useRef<FormikProps<FormValues>>(null);

  useEffect(() => {
    if (showGenerateModal) {
      setIsGenerateGraphModalOpen(true);
    }
  }, [showGenerateModal, setIsGenerateGraphModalOpen]);

  const initialValues: FormValues = {
    prompt: generateGraphModalPrompt,
    modelId: selectedModelId || '',
    sourceSetName: selectedSourceSetName || '',
    maxNLogits: GRAPH_MAXNLOGITS_DEFAULT,
    desiredLogitProb: GRAPH_DESIREDLOGITPROB_DEFAULT,
    nodeThreshold: GRAPH_NODETHRESHOLD_DEFAULT,
    edgeThreshold: GRAPH_EDGETHRESHOLD_DEFAULT,
    maxFeatureNodes: GRAPH_MAXFEATURENODES_DEFAULT,
    slug: '',
  };

  // these are used for custom UI elements for instruct models
  const GRAPH_INSTRUCT_QWEN = ['qwen3-4b'];
  const GRAPH_INSTRUCT_GEMMA_3 = ['gemma-3-4b-it'];

  const convertPromptToChatPromptsQwen = (prompt: string): ChatMessage[] => {
    const prompts: ChatMessage[] = [];

    // Split by the start tokens to find each message
    const parts = prompt.split(/<\|im_start\|>/);

    for (let i = 1; i < parts.length; i += 1) {
      // Skip first empty part
      const part = parts[i];

      // Find the role and content
      const lines = part.split('\n');
      if (lines.length < 2) {
        // eslint-disable-next-line no-continue
        continue;
      }

      const role = lines[0].trim();
      const contentLines = lines.slice(1);

      // Remove <|im_end|> if present at the end
      let content = contentLines.join('\n');
      if (content.endsWith('<|im_end|>')) {
        content = content.slice(0, -10); // Remove '<|im_end|>'
      }

      if (role === 'system' || role === 'user' || role === 'assistant') {
        prompts.push({
          role: role as ChatMessage['role'],
          content: content.trim(),
        });
      }
    }

    return prompts;
  };

  // Gemma 3 format: <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}<end_of_turn>
  // Note: Gemma 3 does NOT have system prompts, and uses "model" instead of "assistant"
  const convertPromptToChatPromptsGemma3 = (prompt: string): ChatMessage[] => {
    const prompts: ChatMessage[] = [];

    // Remove <bos> if present at the start
    let cleanedPrompt = prompt;
    if (cleanedPrompt.startsWith('<bos>')) {
      cleanedPrompt = cleanedPrompt.slice(5);
    }

    // Split by <start_of_turn> to find each message
    const parts = cleanedPrompt.split(/<start_of_turn>/);

    for (let i = 1; i < parts.length; i += 1) {
      // Skip first empty part
      const part = parts[i];

      // Find the role and content
      const lines = part.split('\n');
      if (lines.length < 2) {
        // eslint-disable-next-line no-continue
        continue;
      }

      const role = lines[0].trim();
      const contentLines = lines.slice(1);

      // Remove <end_of_turn> if present at the end
      let content = contentLines.join('\n');
      if (content.endsWith('<end_of_turn>')) {
        content = content.slice(0, -13); // Remove '<end_of_turn>'
      }
      // Also handle trailing newline after <end_of_turn>
      content = content.replace(/<end_of_turn>\n?$/, '');

      // Gemma 3 uses "model" for assistant role
      if (role === 'user') {
        prompts.push({
          role: 'user',
          content: content.trim(),
        });
      } else if (role === 'model') {
        prompts.push({
          role: 'assistant',
          content: content.trim(),
        });
      }
    }

    return prompts;
  };

  useEffect(() => {
    if (generateGraphModalPrompt) {
      let prompts: ChatMessage[] = [];
      if (GRAPH_INSTRUCT_QWEN.includes(selectedModelId)) {
        prompts = convertPromptToChatPromptsQwen(generateGraphModalPrompt);
      } else if (GRAPH_INSTRUCT_GEMMA_3.includes(selectedModelId)) {
        prompts = convertPromptToChatPromptsGemma3(generateGraphModalPrompt);
      }
      setChatPrompts(prompts);
    } else {
      setChatPrompts([]);
    }
  }, [generateGraphModalPrompt]);

  const isInstructAndDoesntStartWithSpecialToken = (modelId: string, prompt: string) => {
    if (GRAPH_INSTRUCT_GEMMA_3.includes(modelId)) {
      return !prompt.startsWith('<bos>');
    }
    if (GRAPH_INSTRUCT_QWEN.includes(modelId)) {
      return !prompt.startsWith('<|im_start|>');
    }
    return false;
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const debouncedTokenize = useCallback(
    _.debounce(
      async (modelId: string, prompt: string, sourceSetName: string, maxNLogits: number, desiredLogitProb: number) => {
        if (!prompt.trim() || !modelId) {
          setGraphTokenizeResponse(null);
          setEstimatedTime(null);
          setIsTokenizing(false);
          return;
        }
        try {
          setIsTokenizing(true);
          console.log(`tokenizing: ${prompt}`);
          console.log(
            'request: ',
            JSON.stringify({
              modelId,
              prompt,
              sourceSetName,
              maxNLogits,
              desiredLogitProb,
            }),
          );
          const response = await fetch('/api/graph/tokenize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              modelId,
              prompt,
              sourceSetName,
              maxNLogits,
              desiredLogitProb,
            }),
          });
          if (!response.ok) {
            console.log('response not ok');
            const errorData = await response.json();
            console.log('errorData: ', errorData);
            throw new Error(errorData.message || 'Failed to tokenize prompt');
          }
          if (prompt.endsWith(' ')) {
            setEndsWithSpace(true);
          } else {
            setEndsWithSpace(false);
          }
          const data = (await response.json()) as GraphTokenizeResponse;
          setGraphTokenizeResponse(data);
          if (data.input_tokens) {
            setEstimatedTime(
              getEstimatedTimeFromNumTokens(data.input_tokens.length, GRAPH_INSTRUCT_GEMMA_3.includes(modelId)),
            );
          } else {
            setEstimatedTime(null);
          }
        } catch (e) {
          console.error('Tokenization error:', e);
          setGraphTokenizeResponse(null);
          setEstimatedTime(null);
        } finally {
          setIsTokenizing(false);
        }
      },
      1000,
    ),
    [],
  );

  // Effect for managing the countdown timer
  useEffect(() => {
    let intervalId: NodeJS.Timeout | undefined;

    if (isGenerating && estimatedTime !== null && estimatedTime > 0) {
      // Initialize countdownTime with the full estimatedTime (rounded to nearest second)
      setCountdownTime(Math.round(estimatedTime));

      intervalId = setInterval(() => {
        setCountdownTime((prevTime) => {
          if (prevTime === null || prevTime <= 1) {
            // Stop at 1 or 0
            if (intervalId) clearInterval(intervalId);
            return 0; // Set to 0 to indicate completion or very short duration
          }
          return prevTime - 1;
        });
      }, 1000);
    } else {
      // Clear interval if not generating or no valid estimate
      if (intervalId) {
        clearInterval(intervalId);
      }
      // Reset countdownTime if generation is not active
      if (!isGenerating) {
        setCountdownTime(null);
      }
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isGenerating, estimatedTime]);

  const handleSubmit = async (values: FormValues, { setSubmitting }: FormikHelpers<FormValues>) => {
    setIsGenerating(true);
    setError(null);
    setGenerationResult(null);

    // remove generate param from url if it's there
    const url = new URL(window.location.href);
    url.searchParams.delete('generate');
    window.history.replaceState({}, '', url.toString());

    try {
      const response = await fetch('/api/graph/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...values,
        }),
      });

      const responseData = await response.json();
      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Rate limit exceeded. Users are limited to 10 graphs per hour - please try again later.');
        }
        if (responseData.error === RUNPOD_BUSY_ERROR) {
          // TODO special limit display
          throw new Error('Oops - looks like we are at capacity right now. Please try again in a minute!');
        }
        throw new Error(responseData.message || responseData.error || 'Failed to generate graph.');
      }

      setGenerationResult(responseData as GenerateGraphResponse);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      if (e instanceof Error && e.message === RUNPOD_BUSY_ERROR) {
        setError(
          <>
            Oops - looks like we are at capacity right now, please try again in a minute. You can also go to{' '}
            <a
              href="https://github.com/safety-research/circuit-tracer"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-700 underline hover:text-sky-800"
            >
              https://github.com/safety-research/circuit-tracer
            </a>{' '}
            to run it yourself in Colab or on a local machine.
          </>,
        );
      } else {
        setError(errorMessage);
        setIsGenerating(false);
      }
    } finally {
      setIsGenerating(false);
      setSubmitting(false);
    }
  };

  const handleOpenChange = (open: boolean) => {
    if (!open && isGenerating) {
      return;
    }
    if (!open) {
      if (!generationResult) {
        setGraphTokenizeResponse(null);
        setChatPrompts([]);
        setEstimatedTime(null);
        setError(null);
        if (formikRef.current) {
          formikRef.current.resetForm();
        }
      }
    } else if (generationResult) {
      setGenerationResult(null);
      setError(null);
      if (formikRef.current) {
        formikRef.current.resetForm();
      }
    }
    setIsGenerateGraphModalOpen(open);
  };

  // Convert chat prompts to model-specific formatted string and update values.prompt
  useEffect(() => {
    if (chatPrompts.length === 0) {
      if (formikRef.current) {
        formikRef.current.setFieldValue('prompt', '');
      }
      return;
    }

    const formatChatToQwen3 = (messages: ChatMessage[]): string => {
      let formatted = '';
      for (let i = 0; i < messages.length; i += 1) {
        const message = messages[i];
        const isLast = i === messages.length - 1;

        if (message.role === 'system') {
          formatted += `<|im_start|>system\n${message.content}${isLast ? '' : '<|im_end|>\n'}`;
        } else if (message.role === 'user') {
          formatted += `<|im_start|>user\n${message.content}${isLast ? '' : '<|im_end|>\n'}`;
        } else if (message.role === 'assistant') {
          formatted += `<|im_start|>assistant\n${message.content}${isLast ? '' : '<|im_end|>\n'}`;
        }
      }
      return formatted;
    };

    // Gemma 3 format: <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}
    // Note: Gemma 3 does NOT have system prompts, uses "model" instead of "assistant"
    const formatChatToGemma3 = (messages: ChatMessage[]): string => {
      // no need to start with BOS, the backend adds it automatically
      let formatted = '';
      for (let i = 0; i < messages.length; i += 1) {
        const message = messages[i];
        const isLast = i === messages.length - 1;

        if (message.role === 'user') {
          formatted += `<start_of_turn>user\n${message.content}${isLast ? '' : '<end_of_turn>\n'}`;
        } else if (message.role === 'assistant') {
          // Gemma 3 uses "model" for assistant messages
          formatted += `<start_of_turn>model\n${message.content}${isLast ? '' : '<end_of_turn>\n'}`;
        }
      }
      return formatted;
    };

    let formattedPrompt = '';
    if (GRAPH_INSTRUCT_QWEN.includes(selectedModelId)) {
      formattedPrompt = formatChatToQwen3(chatPrompts);
    } else if (GRAPH_INSTRUCT_GEMMA_3.includes(selectedModelId)) {
      formattedPrompt = formatChatToGemma3(chatPrompts);
    }

    if (formikRef.current) {
      formikRef.current.setFieldValue('prompt', formattedPrompt);
    }
  }, [chatPrompts, selectedModelId]);

  return (
    <Dialog open={isGenerateGraphModalOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="z-[10001] cursor-default gap-y-2 bg-white text-slate-700 sm:max-w-screen-lg">
        <DialogHeader className="text-center">
          <DialogTitle className="text-center">Generate New Graph</DialogTitle>
          <DialogDescription className="text-center text-xs text-slate-600">
            Generate a new attribution graph for a custom prompt using{' '}
            <a
              href="https://github.com/safety-research/circuit-tracer"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-600 hover:text-sky-800 hover:underline"
            >
              circuit-tracer
            </a>
            .
          </DialogDescription>
        </DialogHeader>

        {!session?.data?.user && (
          <p className="mt-1 rounded-md bg-amber-100 p-3 py-2 text-xs text-amber-700">
            {`You aren't signed in, so you'll need to manually keep track of any graphs you generate. To automatically save graphs to your account, `}
            <Button
              variant="link"
              onClick={() => {
                setSignInModalOpen(true);
                setIsGenerateGraphModalOpen(false);
              }}
              className="h-auto cursor-pointer px-0 py-0 text-xs font-medium text-amber-800 underline md:text-xs"
            >
              sign up with one click
            </Button>
            .
          </p>
        )}
        {!generationResult ? (
          <Formik
            innerRef={formikRef}
            initialValues={initialValues}
            validationSchema={graphGenerateSchemaClient}
            onSubmit={handleSubmit}
          >
            {({ values, errors, touched, handleChange, handleBlur, setFieldValue, dirty }) => (
              <>
                <FormikValuesObserver
                  prompt={values.prompt}
                  modelId={values.modelId}
                  sourceSetName={values.sourceSetName}
                  maxNLogits={values.maxNLogits}
                  desiredLogitProb={values.desiredLogitProb}
                  debouncedTokenize={debouncedTokenize}
                />

                <Form className="flex w-full flex-col gap-x-3">
                  <div className="flex w-full flex-col gap-x-5 pb-3 sm:flex-row">
                    <div className="flex-1">
                      <div className="mb-3 flex flex-row items-end justify-between gap-x-3">
                        <div className="flex flex-1 flex-row gap-x-3">
                          <div className="w-36">
                            <Label htmlFor="modelId" className="px-0 text-left text-xs">
                              Model
                            </Label>
                            <RadixSelect.Root
                              value={values.modelId}
                              onValueChange={(value: string) => {
                                setFieldValue('modelId', value);
                                setFieldValue('sourceSetName', getHasGraphsSourceSetsForModelId(value)[0].name);
                                setGraphTokenizeResponse(null);
                                setChatPrompts([]);
                                setFieldValue('prompt', '');
                              }}
                              disabled={isGenerating}
                            >
                              <RadixSelect.Trigger
                                id="modelId"
                                className="mt-1 flex w-full items-center justify-between rounded-md border border-slate-300 px-3 py-2 text-xs text-slate-500 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                <RadixSelect.Value placeholder="Select a model" />
                                <RadixSelect.Icon className="text-slate-500">
                                  <ChevronDownIcon className="h-4 w-4" />
                                </RadixSelect.Icon>
                              </RadixSelect.Trigger>
                              <RadixSelect.Portal>
                                <RadixSelect.Content className="z-[20000] min-w-[8rem] overflow-hidden rounded-md border border-slate-200 bg-white text-slate-900 shadow-md">
                                  <RadixSelect.ScrollUpButton className="flex items-center justify-center py-1">
                                    <ChevronUpIcon className="h-4 w-4" />
                                  </RadixSelect.ScrollUpButton>
                                  <RadixSelect.Viewport className="divide-y divide-slate-200 p-1">
                                    {GRAPH_GENERATION_ENABLED_MODELS.map((model) => (
                                      <RadixSelect.Item
                                        key={model}
                                        value={model}
                                        className="relative flex w-full cursor-pointer select-none items-center rounded-sm px-3 py-2 text-xs text-slate-600 outline-none hover:bg-sky-100 focus:bg-slate-100 data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                                      >
                                        <RadixSelect.ItemText>
                                          <span className="font-mono font-medium text-sky-700">
                                            {globalModels[model]?.displayName || model}
                                          </span>
                                        </RadixSelect.ItemText>
                                      </RadixSelect.Item>
                                    ))}
                                  </RadixSelect.Viewport>
                                  <RadixSelect.ScrollDownButton className="flex items-center justify-center py-1">
                                    <ChevronDownIcon className="h-4 w-4" />
                                  </RadixSelect.ScrollDownButton>
                                </RadixSelect.Content>
                              </RadixSelect.Portal>
                            </RadixSelect.Root>
                            {errors.modelId && touched.modelId && (
                              <p className="mt-1 text-xs text-red-500">{errors.modelId}</p>
                            )}
                          </div>
                          <div className="flex-1">
                            <Label htmlFor="sourceSet" className="px-0 text-left text-xs">
                              Source Set
                            </Label>
                            <RadixSelect.Root
                              value={values.sourceSetName}
                              onValueChange={(value: string) => {
                                setFieldValue('sourceSetName', value);
                              }}
                              disabled={isGenerating}
                            >
                              <RadixSelect.Trigger
                                id="sourceSetName"
                                className="mt-1 flex w-full items-center justify-between rounded-md border border-slate-300 px-2 py-2 text-[11px] text-slate-500 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                <RadixSelect.Value
                                  placeholder={<span className="text-red-500">Select a source set</span>}
                                />
                                <RadixSelect.Icon className="text-slate-500">
                                  <ChevronDownIcon className="h-4 w-4" />
                                </RadixSelect.Icon>
                              </RadixSelect.Trigger>
                              <RadixSelect.Portal>
                                <RadixSelect.Content className="z-[20000] min-w-[8rem] overflow-hidden rounded-md border border-slate-200 bg-white text-slate-900 shadow-md">
                                  <RadixSelect.ScrollUpButton className="flex items-center justify-center py-1">
                                    <ChevronUpIcon className="h-4 w-4" />
                                  </RadixSelect.ScrollUpButton>
                                  <RadixSelect.Viewport className="divide-y divide-slate-200 p-1">
                                    {getHasGraphsSourceSetsForModelId(values.modelId).map((sourceSet) => (
                                      <RadixSelect.Item
                                        key={sourceSet.name}
                                        value={sourceSet.name}
                                        className="relative flex w-full cursor-pointer select-none flex-col items-start gap-y-1 rounded-sm px-3 py-2.5 text-xs text-slate-600 outline-none hover:bg-sky-100 focus:bg-slate-100 data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                                      >
                                        <RadixSelect.ItemText className="w-full text-left font-mono font-medium uppercase text-sky-700">
                                          <span className="font-mono font-medium text-sky-700">
                                            {sourceSet.name.toUpperCase()}
                                          </span>
                                        </RadixSelect.ItemText>
                                        <div className="w-full text-[10px] font-normal text-slate-500">
                                          {sourceSet.description} · {sourceSet.creatorName}
                                        </div>
                                      </RadixSelect.Item>
                                    ))}
                                  </RadixSelect.Viewport>
                                  <RadixSelect.ScrollDownButton className="flex items-center justify-center py-1">
                                    <ChevronDownIcon className="h-4 w-4" />
                                  </RadixSelect.ScrollDownButton>
                                </RadixSelect.Content>
                              </RadixSelect.Portal>
                            </RadixSelect.Root>
                            {errors.sourceSetName && touched.sourceSetName && (
                              <p className="mt-1 text-xs text-red-500">{errors.sourceSetName}</p>
                            )}
                          </div>
                        </div>
                        <Button
                          type="button"
                          variant="slateLight"
                          size="sm"
                          onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                          disabled={isGenerating}
                          className="hidden h-9 px-3 text-[11px] leading-snug text-slate-600 hover:text-slate-700 sm:block"
                        >
                          Advanced
                        </Button>
                      </div>

                      {showAdvancedSettings && (
                        <div className="w-full">
                          {/* <div className="hidden items-center pt-0 sm:flex">
                          <div className="mr-3 flex-1 border-t border-slate-200" />
                          <span className="text-[11px] text-slate-500">Advanced Settings</span>
                          <div className="ml-3 flex-1 border-t border-slate-200" />
                        </div> */}
                          <div className="flex flex-row gap-x-3 pb-2 pt-1">
                            <div className="flex-1">
                              <Label htmlFor="slug" className="text-[11px]">
                                Custom Graph ID (Optional)
                              </Label>
                              <Input
                                id="slug"
                                name="slug"
                                value={values.slug}
                                onChange={(e) => {
                                  const val = e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '');
                                  setFieldValue('slug', val);
                                }}
                                onKeyDown={(e) => {
                                  e.stopPropagation();
                                }}
                                onBlur={handleBlur}
                                disabled={isGenerating}
                                placeholder="my-graph"
                                className="mt-1 h-8 w-full border-slate-300 text-xs placeholder-slate-400 md:text-xs"
                                maxLength={50}
                              />
                              {errors.slug && touched.slug && (
                                <p className="mt-1 text-xs text-red-500">{errors.slug}</p>
                              )}
                            </div>
                            <div>
                              <Label
                                htmlFor="maxNLogits"
                                className="h-6 w-12 border-slate-300 px-0 text-center text-slate-600 md:text-[11px]"
                              >
                                Max # Logits
                              </Label>
                              <Input
                                id="maxNLogits"
                                name="maxNLogits"
                                type="number"
                                value={values.maxNLogits}
                                disabled={isGenerating}
                                onChange={handleChange}
                                onBlur={handleBlur}
                                className="mt-1 h-8 w-full border-slate-300 text-center text-xs md:text-xs"
                                min={GRAPH_MAXNLOGITS_MIN}
                                max={GRAPH_MAXNLOGITS_MAX}
                                step={1}
                              />
                              {errors.maxNLogits && touched.maxNLogits && (
                                <p className="mt-1 text-xs text-red-500">{errors.maxNLogits}</p>
                              )}
                            </div>
                          </div>

                          {/* Attribution Settings Group */}
                          <div className="space-y-0 pb-2 pt-3">
                            <div className="text-left text-[10px] font-medium uppercase leading-none text-slate-400">
                              Attribution
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <Label
                                  htmlFor="desiredLogitProb"
                                  className="h-6 w-12 border-slate-300 px-0 text-left text-slate-500 md:text-[11px]"
                                >
                                  Desired Logit Probability
                                </Label>
                                <div className="mt-0.5 flex items-center space-x-2">
                                  <Input
                                    id="desiredLogitProb"
                                    name="desiredLogitProb"
                                    type="number"
                                    value={values.desiredLogitProb}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    disabled={isGenerating}
                                    className="h-6 w-12 border-slate-300 px-0 text-center text-slate-600 md:text-[11px]"
                                    min={GRAPH_DESIREDLOGITPROB_MIN}
                                    max={GRAPH_DESIREDLOGITPROB_MAX}
                                    step={0.01}
                                  />
                                  <RadixSlider.Root
                                    name="desiredLogitProb"
                                    value={[values.desiredLogitProb]}
                                    onValueChange={(newVal: number[]) => setFieldValue('desiredLogitProb', newVal[0])}
                                    min={GRAPH_DESIREDLOGITPROB_MIN}
                                    max={GRAPH_DESIREDLOGITPROB_MAX}
                                    disabled={isGenerating}
                                    step={0.01}
                                    className="relative flex h-4 w-full flex-1 touch-none select-none items-center"
                                  >
                                    <RadixSlider.Track className="relative h-1.5 w-full flex-grow overflow-hidden rounded-full bg-slate-200">
                                      <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
                                    </RadixSlider.Track>
                                    <RadixSlider.Thumb className="block h-4 w-4 rounded-full border-2 border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
                                  </RadixSlider.Root>
                                </div>
                                {errors.desiredLogitProb && touched.desiredLogitProb && (
                                  <p className="mt-1 text-xs text-red-500">{errors.desiredLogitProb}</p>
                                )}
                              </div>

                              {/* Max # Nodes */}
                              <div>
                                <Label
                                  htmlFor="maxFeatureNodes"
                                  className="h-6 w-12 border-slate-300 px-0 text-left text-slate-500 md:text-[11px]"
                                >
                                  Max # Nodes
                                </Label>
                                <div className="mt-0.5 flex items-center space-x-2">
                                  <Input
                                    id="maxFeatureNodes"
                                    name="maxFeatureNodes"
                                    type="number"
                                    value={values.maxFeatureNodes}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    disabled={isGenerating}
                                    className="h-6 w-12 border-slate-300 px-0 text-center text-slate-600 md:text-[11px]"
                                    min={GRAPH_MAXFEATURENODES_MIN}
                                    max={GRAPH_MAXFEATURENODES_MAX}
                                    step={1}
                                  />
                                  <RadixSlider.Root
                                    name="maxFeatureNodes"
                                    value={[values.maxFeatureNodes]}
                                    onValueChange={(newVal: number[]) => setFieldValue('maxFeatureNodes', newVal[0])}
                                    min={GRAPH_MAXFEATURENODES_MIN}
                                    max={GRAPH_MAXFEATURENODES_MAX}
                                    disabled={isGenerating}
                                    step={500}
                                    className="relative flex h-4 w-full flex-1 touch-none select-none items-center"
                                  >
                                    <RadixSlider.Track className="relative h-1.5 w-full flex-grow overflow-hidden rounded-full bg-slate-200">
                                      <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
                                    </RadixSlider.Track>
                                    <RadixSlider.Thumb className="block h-4 w-4 rounded-full border-2 border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
                                  </RadixSlider.Root>
                                </div>
                                {errors.maxFeatureNodes && touched.maxFeatureNodes && (
                                  <p className="mt-1 text-xs text-red-500">{errors.maxFeatureNodes}</p>
                                )}
                              </div>
                            </div>
                          </div>

                          {/* Pruning Settings Group */}
                          <div className="space-y-0 pb-3 pt-3">
                            <div className="text-left text-[10px] font-medium uppercase leading-none text-slate-400">
                              Pruning
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              {/* Node Threshold and Edge Threshold on same line */}
                              <div>
                                <Label
                                  htmlFor="nodeThreshold"
                                  className="h-6 w-12 border-slate-300 px-0 text-left text-slate-500 md:text-[11px]"
                                >
                                  Node Threshold
                                </Label>
                                <div className="mt-0.5 flex items-center space-x-2">
                                  <Input
                                    id="nodeThreshold"
                                    name="nodeThreshold"
                                    type="number"
                                    value={values.nodeThreshold}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    disabled={isGenerating}
                                    className="h-6 w-12 border-slate-300 px-0 text-center text-slate-600 md:text-[11px]"
                                    min={GRAPH_NODETHRESHOLD_MIN}
                                    max={GRAPH_NODETHRESHOLD_MAX}
                                    step={0.01}
                                  />
                                  <RadixSlider.Root
                                    name="nodeThreshold"
                                    value={[values.nodeThreshold]}
                                    onValueChange={(newVal: number[]) => setFieldValue('nodeThreshold', newVal[0])}
                                    min={GRAPH_NODETHRESHOLD_MIN}
                                    max={GRAPH_NODETHRESHOLD_MAX}
                                    disabled={isGenerating}
                                    step={0.01}
                                    className="relative flex h-4 w-full flex-1 touch-none select-none items-center"
                                  >
                                    <RadixSlider.Track className="relative h-1.5 w-full flex-grow overflow-hidden rounded-full bg-slate-200">
                                      <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
                                    </RadixSlider.Track>
                                    <RadixSlider.Thumb className="block h-4 w-4 rounded-full border-2 border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
                                  </RadixSlider.Root>
                                </div>
                                {errors.nodeThreshold && touched.nodeThreshold && (
                                  <p className="mt-1 text-xs text-red-500">{errors.nodeThreshold}</p>
                                )}
                              </div>
                              <div>
                                <Label
                                  htmlFor="edgeThreshold"
                                  className="h-6 w-12 border-slate-300 px-0 text-left text-slate-500 md:text-[11px]"
                                >
                                  Edge Threshold
                                </Label>
                                <div className="mt-0.5 flex items-center space-x-2">
                                  <Input
                                    id="edgeThreshold"
                                    name="edgeThreshold"
                                    type="number"
                                    value={values.edgeThreshold}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    disabled={isGenerating}
                                    className="h-6 w-12 border-slate-300 px-0 text-center text-slate-600 md:text-[11px]"
                                    min={GRAPH_EDGETHRESHOLD_MIN}
                                    max={GRAPH_EDGETHRESHOLD_MAX}
                                    step={0.01}
                                  />
                                  <RadixSlider.Root
                                    name="edgeThreshold"
                                    value={[values.edgeThreshold]}
                                    onValueChange={(newVal: number[]) => setFieldValue('edgeThreshold', newVal[0])}
                                    min={GRAPH_EDGETHRESHOLD_MIN}
                                    max={GRAPH_EDGETHRESHOLD_MAX}
                                    disabled={isGenerating}
                                    step={0.01}
                                    className="relative flex h-4 w-full flex-1 touch-none select-none items-center"
                                  >
                                    <RadixSlider.Track className="relative h-1.5 w-full flex-grow overflow-hidden rounded-full bg-slate-200">
                                      <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
                                    </RadixSlider.Track>
                                    <RadixSlider.Thumb className="block h-4 w-4 rounded-full border-2 border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
                                  </RadixSlider.Root>
                                </div>
                                {errors.edgeThreshold && touched.edgeThreshold && (
                                  <p className="mt-1 text-xs text-red-500">{errors.edgeThreshold}</p>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      <div>
                        {GRAPH_INSTRUCT_QWEN.includes(values.modelId) ? (
                          <>
                            <Label htmlFor="prompt" className="text-xs">
                              Chat Messages to Complete
                            </Label>
                            <p className="mt-0.5 text-[11.5px] text-slate-500">
                              Click a message type to append to the chat conversation.
                            </p>
                            <Input
                              id="prompt"
                              name="prompt"
                              value={values.prompt}
                              onChange={handleChange}
                              onBlur={handleBlur}
                              type="hidden"
                            />
                            <div className="mb-3 mt-2 flex flex-col gap-y-1">
                              {chatPrompts.map((prompt, index) => (
                                <div
                                  key={index}
                                  className="mb-0 mt-0 flex flex-row items-center justify-center gap-x-2 text-xs"
                                >
                                  <div className="w-20 min-w-20 self-start rounded-md bg-slate-200 px-2 py-1.5 text-center text-[10px] font-medium uppercase leading-none text-slate-600">
                                    {prompt.role}
                                  </div>
                                  <div className="flex w-full flex-col">
                                    {prompt.role === 'assistant' ? (
                                      // assistant has two fields: think and content
                                      <>
                                        <div className="relative w-full">
                                          <ReactTextareaAutosize
                                            value={prompt.content.match(/<think>\n(.*?)\n<\/think>\n/s)?.[1] || ''}
                                            onChange={(e) => {
                                              setChatPrompts(
                                                chatPrompts.map((p, i) =>
                                                  i === index
                                                    ? {
                                                        ...p,
                                                        content: p.content.includes('<think>\n')
                                                          ? p.content.replace(
                                                              /<think>\n(.*?)\n<\/think>\n/s,
                                                              `<think>\n${e.target.value}\n</think>\n`,
                                                            )
                                                          : `<think>\n${e.target.value}\n</think>\n\n${p.content}`,
                                                      }
                                                    : p,
                                                ),
                                              );
                                            }}
                                            minRows={1}
                                            placeholder="Enter optional thinking text. Do not include <think> tags."
                                            onBlur={handleBlur}
                                            onKeyDown={(e) => {
                                              e.stopPropagation();
                                            }}
                                            disabled={isGenerating}
                                            className="w-full flex-1 resize-none rounded-md border border-slate-300 px-3 py-5 text-xs leading-normal text-slate-700 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:ring-sky-500"
                                          />
                                          <div className="absolute left-2 top-[3px] text-[10px] text-slate-400">{`<think>`}</div>
                                          <div className="absolute bottom-[7px] left-2 text-[10px] text-slate-400">{`</think>`}</div>
                                        </div>
                                        <ReactTextareaAutosize
                                          value={
                                            prompt.content.includes('</think>\n\n')
                                              ? prompt.content.split('</think>\n\n')[1] || ''
                                              : prompt.content
                                          }
                                          minRows={1}
                                          onChange={(e) => {
                                            setChatPrompts(
                                              chatPrompts.map((p, i) =>
                                                i === index
                                                  ? {
                                                      ...p,
                                                      content: p.content.includes('</think>\n\n')
                                                        ? `${p.content.split('</think>\n\n')[0]}</think>\n\n${e.target.value}`
                                                        : e.target.value,
                                                    }
                                                  : p,
                                              ),
                                            );
                                          }}
                                          placeholder={`Enter the ${prompt.role} message.`}
                                          onBlur={handleBlur}
                                          onKeyDown={(e) => {
                                            e.stopPropagation();
                                          }}
                                          disabled={isGenerating}
                                          className="w-full flex-1 resize-none rounded-md border border-slate-300 px-3 py-2 text-xs leading-normal text-slate-700 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:ring-sky-500"
                                        />
                                      </>
                                    ) : (
                                      <ReactTextareaAutosize
                                        value={prompt.content}
                                        minRows={1}
                                        onChange={(e) => {
                                          setChatPrompts(
                                            chatPrompts.map((p, i) =>
                                              i === index ? { ...p, content: e.target.value } : p,
                                            ),
                                          );
                                        }}
                                        placeholder={`Enter the ${prompt.role} message.`}
                                        onBlur={handleBlur}
                                        onKeyDown={(e) => {
                                          e.stopPropagation();
                                        }}
                                        disabled={isGenerating}
                                        className="w-full flex-1 resize-none rounded-md border border-slate-300 px-3 py-2 text-xs leading-normal text-slate-700 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:ring-sky-500"
                                      />
                                    )}
                                  </div>
                                  <Button
                                    type="button"
                                    variant="destructive"
                                    size="sm"
                                    className="h-5 w-5 min-w-5 self-start px-0 text-xs"
                                    disabled={index !== chatPrompts.length - 1 || isGenerating}
                                    onClick={() => {
                                      // eslint-disable-next-line
                                      setChatPrompts(chatPrompts.filter((_, i) => i !== index));
                                    }}
                                  >
                                    <X className="h-3 w-3" />
                                  </Button>
                                </div>
                              ))}
                            </div>
                            <div className="mt-0.5 flex w-full flex-row gap-x-2">
                              <Button
                                type="button"
                                variant="default"
                                size="sm"
                                className="h-9 flex-1 px-2 text-xs"
                                disabled={chatPrompts.length > 0 || isGenerating}
                                onClick={() => {
                                  setChatPrompts([...chatPrompts, { role: 'system', content: '' }]);
                                }}
                              >
                                + System
                              </Button>
                              <Button
                                type="button"
                                variant="default"
                                size="sm"
                                className="h-9 flex-1 px-2 text-xs"
                                disabled={
                                  isGenerating ||
                                  (chatPrompts.length === 0
                                    ? false
                                    : !(
                                        chatPrompts[chatPrompts.length - 1].role === 'system' ||
                                        chatPrompts[chatPrompts.length - 1].role === 'assistant'
                                      ) || chatPrompts[chatPrompts.length - 1].content.trim().length === 0) // user can only come after system or assistant
                                }
                                onClick={() => {
                                  setChatPrompts([...chatPrompts, { role: 'user', content: '' }]);
                                }}
                              >
                                + User
                              </Button>
                              <Button
                                type="button"
                                variant="default"
                                size="sm"
                                className="h-9 flex-1 px-2 text-xs"
                                disabled={
                                  isGenerating ||
                                  (chatPrompts.length > 0
                                    ? chatPrompts[chatPrompts.length - 1].role !== 'user' ||
                                      chatPrompts[chatPrompts.length - 1].content.trim().length === 0 // assistant can only come after user
                                    : true)
                                }
                                onClick={() => {
                                  setChatPrompts([
                                    ...chatPrompts,
                                    { role: 'assistant', content: '<think>\n\n</think>\n\n' },
                                  ]);
                                }}
                              >
                                + Assistant
                              </Button>
                            </div>
                          </>
                        ) : GRAPH_INSTRUCT_GEMMA_3.includes(values.modelId) ? (
                          <>
                            <Label htmlFor="prompt" className="text-xs">
                              Chat Messages to Complete
                            </Label>
                            <p className="mt-0.5 text-[11.5px] text-slate-500">
                              Click a message type to append to the chat conversation.
                            </p>
                            <Input
                              id="prompt"
                              name="prompt"
                              value={values.prompt}
                              onChange={handleChange}
                              onBlur={handleBlur}
                              type="hidden"
                            />
                            <div className="mb-3 mt-2 flex flex-col gap-y-1">
                              {chatPrompts.map((prompt, index) => (
                                <div
                                  key={index}
                                  className="mb-0 mt-0 flex flex-row items-center justify-center gap-x-2 text-xs"
                                >
                                  <div className="w-20 min-w-20 self-start rounded-md bg-slate-200 px-2 py-1.5 text-center text-[10px] font-medium uppercase leading-none text-slate-600">
                                    {prompt.role === 'assistant' ? 'model' : prompt.role}
                                  </div>
                                  <div className="flex w-full flex-col">
                                    <ReactTextareaAutosize
                                      value={prompt.content}
                                      minRows={1}
                                      onChange={(e) => {
                                        setChatPrompts(
                                          chatPrompts.map((p, i) =>
                                            i === index ? { ...p, content: e.target.value } : p,
                                          ),
                                        );
                                      }}
                                      placeholder={`Enter the ${prompt.role === 'assistant' ? 'model' : prompt.role} message.`}
                                      onBlur={handleBlur}
                                      onKeyDown={(e) => {
                                        e.stopPropagation();
                                      }}
                                      disabled={isGenerating}
                                      className="w-full flex-1 resize-none rounded-md border border-slate-300 px-3 py-2 text-xs leading-normal text-slate-700 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:ring-sky-500"
                                    />
                                  </div>
                                  <Button
                                    type="button"
                                    variant="destructive"
                                    size="sm"
                                    className="h-5 w-5 min-w-5 self-start px-0 text-xs"
                                    disabled={index !== chatPrompts.length - 1 || isGenerating}
                                    onClick={() => {
                                      // eslint-disable-next-line
                                      setChatPrompts(chatPrompts.filter((_, i) => i !== index));
                                    }}
                                  >
                                    <X className="h-3 w-3" />
                                  </Button>
                                </div>
                              ))}
                            </div>
                            <div className="mt-0.5 flex w-full flex-row gap-x-2">
                              <Button
                                type="button"
                                variant="default"
                                size="sm"
                                className="h-9 flex-1 px-2 text-xs"
                                disabled={
                                  isGenerating ||
                                  (chatPrompts.length === 0
                                    ? false
                                    : chatPrompts[chatPrompts.length - 1].role !== 'assistant' ||
                                      chatPrompts[chatPrompts.length - 1].content.trim().length === 0) // user can only come first or after assistant
                                }
                                onClick={() => {
                                  setChatPrompts([...chatPrompts, { role: 'user', content: '' }]);
                                }}
                              >
                                + User
                              </Button>
                              <Button
                                type="button"
                                variant="default"
                                size="sm"
                                className="h-9 flex-1 px-2 text-xs"
                                disabled={
                                  isGenerating ||
                                  (chatPrompts.length > 0
                                    ? chatPrompts[chatPrompts.length - 1].role !== 'user' ||
                                      chatPrompts[chatPrompts.length - 1].content.trim().length === 0 // model can only come after user
                                    : true)
                                }
                                onClick={() => {
                                  setChatPrompts([...chatPrompts, { role: 'assistant', content: '' }]);
                                }}
                              >
                                + Model
                              </Button>
                            </div>
                          </>
                        ) : (
                          <>
                            <Label htmlFor="prompt" className="text-xs">
                              Prompt to Complete
                            </Label>
                            <p className="mt-0.5 text-[11.5px] text-slate-500">
                              In general, you want your prompt to be missing a word at the end, because we want to
                              analyze how the model comes up with the word <strong>after</strong> your prompt. (Eg
                              &quot;The capital of the state containing Dallas is&quot;)
                            </p>
                            <ReactTextareaAutosize
                              id="prompt"
                              name="prompt"
                              minRows={3}
                              maxRows={6}
                              value={values.prompt}
                              onChange={handleChange}
                              onBlur={handleBlur}
                              onKeyDown={(e) => {
                                e.stopPropagation();
                              }}
                              disabled={isGenerating}
                              placeholder="Enter the prompt to visualize..."
                              className="mt-1 w-full resize-none rounded-md border border-slate-300 p-2 text-xs leading-normal text-slate-700 shadow-sm focus:border-sky-500 focus:ring-sky-500"
                              maxLength={GRAPH_MAX_PROMPT_LENGTH_CHARS}
                            />
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex-1">
                      {isTokenizing ? (
                        <div className="mt-1.5 flex w-full flex-row items-center justify-start gap-x-0.5 pb-2 text-xs text-sky-700">
                          <LoadingSquare className="mr-1 h-5 w-5" size={20} />
                          <div className="flex items-center justify-start">Tokenizing...</div>
                        </div>
                      ) : graphTokenizeResponse ? (
                        <div className="flex flex-col">
                          <div className="forceShowScrollBar mx-0 mb-2 mt-1 max-h-full flex-1 overflow-y-scroll text-xs text-slate-500">
                            <div className="text-[8px] font-medium uppercase text-slate-500">
                              {graphTokenizeResponse.input_tokens.length} TOKENS
                            </div>
                            <div className="">
                              {(() => {
                                const tokenGroups: string[][] = [];
                                let currentGroup: string[] = [];

                                graphTokenizeResponse.input_tokens.forEach((token) => {
                                  currentGroup.push(token);
                                  if (/\n/.test(token)) {
                                    tokenGroups.push(currentGroup);
                                    currentGroup = [];
                                  }
                                });

                                if (currentGroup.length > 0) {
                                  tokenGroups.push(currentGroup);
                                }

                                return tokenGroups.map((group, groupIdx) => (
                                  <div key={groupIdx} className="flex flex-wrap">
                                    {group.map((t, idx) => (
                                      <span
                                        key={`${t}-${groupIdx}-${idx}`}
                                        className="mx-0.5 my-0.5 rounded bg-slate-200 px-[5px] py-[1px] font-mono text-[10px] text-slate-700"
                                      >
                                        {t.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                                      </span>
                                    ))}
                                    {/\n/.test(group[group.length - 1]) && <br />}
                                  </div>
                                ));
                              })()}
                            </div>
                          </div>
                          <div className="mb-0.5 text-[8px] font-medium uppercase text-slate-500">Next Token</div>
                          <div className="mb-2 flex flex-wrap gap-x-1.5 gap-y-[3px]">
                            {graphTokenizeResponse.salient_logits.slice(0, 5).map((logit, index) => (
                              <span
                                key={index}
                                className="whitespace-pre rounded bg-slate-200 px-[7px] py-[3px] text-xs"
                              >
                                <span className="whitespace-pre pr-1 font-mono text-slate-700">
                                  {logit.token.replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                                </span>
                                <span className="whitespace-pre font-sans text-slate-400">
                                  {logit.probability.toFixed(2)}
                                </span>
                              </span>
                            ))}
                          </div>

                          {endsWithSpace && (
                            <div className="mb-1.5 mt-0 text-xs text-amber-600">
                              Warning: Your prompt ends with a space, which may result in an unexpected output because
                              tokens often already have a space prepended (eg{' '}
                              <span className="mx-0 whitespace-pre rounded bg-slate-200 px-[3px] py-[1px] font-mono text-[10px] text-slate-700">
                                {' '}
                                coffee
                              </span>
                              , not{' '}
                              <span className="mx-0 whitespace-pre rounded bg-slate-200 px-[3px] py-[1px] font-mono text-[10px] text-slate-700">
                                coffee
                              </span>
                              ). Consider removing the ending space.
                            </div>
                          )}
                          {graphTokenizeResponse.salient_logits.length > 0 &&
                            (graphTokenizeResponse.salient_logits[0].token.trim() === '' ? (
                              <div className="mb-1.5 mt-0 text-xs text-amber-600">
                                {/* "i like to   " will trigger this */}
                                Warning: The next most likely token is whitespace. This may not be an
                                &apos;interesting&apos; graph as spaces are not often an ideal example of a reasoning
                                conclusion.
                              </div>
                            ) : isBoringToken(graphTokenizeResponse.salient_logits[0].token) ? (
                              <div className="mb-1.5 mt-0 text-xs text-amber-600">
                                Warning: The next most likely token is a special token, markdown, or HTML/symbol. Double
                                check that this is the reasoning &apos;conclusion&apos; that you expect to see.
                              </div>
                            ) : (
                              <div className="mb-1.5 mt-0 text-xs text-slate-500">
                                Check the next tokens to make sure they are a satisfying &apos;conclusion&apos; to your
                                prompt. If it&apos;s a markdown or HTML tag, you should probably change your prompt.
                              </div>
                            ))}
                        </div>
                      ) : (
                        <div className="mt-1 pb-3 pt-0.5 text-xs text-slate-400">
                          Tokenized prompt and likely next tokens will appear here.
                        </div>
                      )}
                      {errors.prompt && touched.prompt && <p className="mt-1 text-xs text-red-500">{errors.prompt}</p>}
                      {graphTokenizeResponse &&
                        !isTokenizing &&
                        graphTokenizeResponse.input_tokens.length > GRAPH_MAX_TOKENS && (
                          <p className="mt-1 text-xs text-red-500">
                            Prompt exceeds maximum token limit of {GRAPH_MAX_TOKENS}.
                          </p>
                        )}
                      {isInstructAndDoesntStartWithSpecialToken(values.modelId, values.prompt) && (
                        <div className="mb-1.5 mt-0 text-xs text-amber-600">
                          {GRAPH_INSTRUCT_QWEN.includes(values.modelId) ? (
                            <>Click &quot;+ System&quot; or &quot;+ User&quot; to start creating a chat.</>
                          ) : (
                            <>Click &quot;+ User&quot; or &quot;+ Model&quot; to start creating a chat.</>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {error && <p className="mt-4 rounded-md bg-red-100 p-3 text-sm text-red-600">{error}</p>}

                  <div className="mt-2 flex flex-col items-center justify-between gap-y-3 border-t pt-4 text-xs sm:flex-row">
                    {isGenerating ? (
                      countdownTime !== null ? (
                        <div className="flex flex-row items-center justify-start gap-x-2 whitespace-pre text-xs leading-none text-slate-500 sm:text-sm">
                          <LoadingSquare className="h-6 w-6 sm:h-8 sm:w-8" size={32} />
                          Remaining time: {formatCountdown(countdownTime)}
                        </div>
                      ) : countdownTime === 0 ? (
                        <div className="text-xs text-slate-600 sm:text-sm">Remaining time: {formatCountdown(0)}</div>
                      ) : (
                        <div className="text-xs text-slate-600 sm:text-sm">Estimating time...</div>
                      )
                    ) : estimatedTime !== null && !generationResult ? (
                      <div className="flex flex-row items-center justify-start gap-x-1.5 whitespace-pre text-xs leading-none text-slate-500 sm:text-sm">
                        Estimated generation time:{' '}
                        {estimatedTime < 60
                          ? `~${Math.round(estimatedTime)} sec`
                          : `~${(estimatedTime / 60).toFixed(1)} min`}
                      </div>
                    ) : (
                      <div /> // Empty div to maintain layout for button alignment
                    )}
                    <div className="flex w-full flex-row justify-end gap-x-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => {
                          if (formikRef.current) {
                            formikRef.current.resetForm({
                              values: {
                                ...initialValues,
                                prompt: '',
                                slug: '',
                              },
                            });
                            setGraphTokenizeResponse(null);
                            setEstimatedTime(null);
                            setError(null);
                            setChatPrompts([]);
                          }
                        }}
                        disabled={isGenerating}
                        className="flex items-center justify-center gap-x-1.5 text-xs"
                        title="Reset to defaults"
                      >
                        Reset
                      </Button>
                      <Button
                        type="submit"
                        variant="emerald"
                        disabled={
                          isGenerating ||
                          !dirty ||
                          Object.keys(errors).length > 0 ||
                          (graphTokenizeResponse !== null &&
                            graphTokenizeResponse.input_tokens.length > GRAPH_MAX_TOKENS)
                        }
                        className="w-full text-xs sm:w-auto"
                      >
                        {isGenerating ? <>Generating...</> : 'Start Generation'}
                      </Button>
                    </div>
                  </div>
                </Form>
              </>
            )}
          </Formik>
        ) : (
          <div className="mt-2 flex w-full max-w-xl flex-col space-y-1.5 place-self-center self-center rounded-md bg-slate-50 p-3 pb-2">
            <h3 className="text-center text-base font-medium text-slate-800">Graph Generated Successfully</h3>
            <div className="flex w-full flex-row gap-x-2 gap-y-1">
              <p className="flex-1 text-center text-sm text-slate-700">{generationResult.numNodes} Nodes</p>
              <p className="flex-1 text-center text-sm text-slate-700">{generationResult.numLinks} Links</p>
            </div>
            <p className="pb-3 text-center text-xs">
              <a href={generationResult.url} rel="noopener noreferrer" className="break-all hover:underline">
                {generationResult.url}
              </a>
            </p>
            {/* <p className="text-xs text-slate-500">
              Raw S3 URL:{' '}
              <a
                href={generationResult.s3url}
                target="_blank"
                rel="noopener noreferrer"
                className="break-all text-sky-600 hover:underline"
              >
                {generationResult.s3url}
              </a>
            </p> */}
            <div className="mt-2 flex w-full flex-col gap-y-2 sm:flex-row sm:gap-x-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  if (formikRef.current) {
                    formikRef.current.resetForm({
                      values: {
                        ...initialValues,
                        prompt: '',
                        slug: '',
                      },
                    });
                  }
                  setGenerationResult(null);
                  setGraphTokenizeResponse(null);
                  setEstimatedTime(null);
                  setError(null);
                  setChatPrompts([]);
                }}
                disabled={isGenerating}
                className="flex items-center justify-center gap-x-1.5 text-xs"
                title="Reset to defaults"
              >
                Reset
              </Button>
              <Button
                onClick={() => {
                  window.location.href = generationResult.url;
                }}
                className="flex-1 bg-sky-600 text-sm hover:bg-sky-700 sm:w-auto"
              >
                Open Graph
              </Button>
              <Button
                onClick={() => {
                  window.open(generationResult.url, '_blank');
                  setIsGenerateGraphModalOpen(false);
                }}
                className="flex-1 bg-sky-600 hover:bg-sky-700 sm:w-auto"
              >
                Open Graph (New Tab)
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
