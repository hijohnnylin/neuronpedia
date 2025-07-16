'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
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
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';

const BORING_TOKENS = ['<strong>', '<em>', '<code>', '<b>', '<i>'];
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
  maxNLogits: number;
  desiredLogitProb: number;
  debouncedTokenize: (modelId: string, prompt: string, maxNLogits: number, desiredLogitProb: number) => void;
  // eslint-disable-next-line
}> = ({ prompt, modelId, maxNLogits, desiredLogitProb, debouncedTokenize }) => {
  useEffect(() => {
    debouncedTokenize(modelId, prompt, maxNLogits, desiredLogitProb);
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

export default function GenerateGraphModal() {
  const { isGenerateGraphModalOpen, setIsGenerateGraphModalOpen } = useGraphModalContext();
  const [generationResult, setGenerationResult] = useState<GenerateGraphResponse | null>(null);
  const [graphTokenizeResponse, setGraphTokenizeResponse] = useState<GraphTokenizeResponse | null>(null);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTokenizing, setIsTokenizing] = useState(false);
  const [endsWithSpace, setEndsWithSpace] = useState(false);
  const [error, setError] = useState<React.ReactNode | null>(null);
  const [countdownTime, setCountdownTime] = useState<number | null>(null);

  const session = useSession();
  const { setSignInModalOpen, showToast } = useGlobalContext() as any;
  const formikRef = useRef<FormikProps<FormValues>>(null);

  const initialValues: FormValues = {
    prompt: '',
    modelId: GRAPH_GENERATION_ENABLED_MODELS[0] || '',
    maxNLogits: GRAPH_MAXNLOGITS_DEFAULT,
    desiredLogitProb: GRAPH_DESIREDLOGITPROB_DEFAULT,
    nodeThreshold: GRAPH_NODETHRESHOLD_DEFAULT,
    edgeThreshold: GRAPH_EDGETHRESHOLD_DEFAULT,
    maxFeatureNodes: GRAPH_MAXFEATURENODES_DEFAULT,
    slug: '',
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const debouncedTokenize = useCallback(
    _.debounce(async (modelId: string, prompt: string, maxNLogits: number, desiredLogitProb: number) => {
      if (!prompt.trim() || !modelId) {
        setGraphTokenizeResponse(null);
        setEstimatedTime(null);
        setIsTokenizing(false);
        return;
      }
      try {
        setIsTokenizing(true);
        console.log(`tokenizing: ${prompt}`);
        const response = await fetch('/api/graph/tokenize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modelId,
            prompt,
            maxNLogits,
            desiredLogitProb,
          }),
        });
        if (!response.ok) {
          const errorData = await response.json();
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
          setEstimatedTime(getEstimatedTimeFromNumTokens(data.input_tokens.length));
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
    }, 1000),
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

    try {
      const response = await fetch('/api/graph/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
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
      if (showToast) {
        showToast({
          title: 'Success!',
          description: responseData.message || 'Graph generated successfully.',
          variant: 'success',
        });
      }
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
              className="text-sky-600 underline hover:text-sky-800"
            >
              https://github.com/safety-research/circuit-tracer
            </a>{' '}
            to run it yourself in Colab or on a local machine.
          </>,
        );
      } else {
        setError(errorMessage);
      }
      if (showToast) {
        showToast({
          title: 'Error Generating Graph',
          description: errorMessage,
          variant: 'destructive',
        });
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

  return (
    <Dialog open={isGenerateGraphModalOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="z-[10001] cursor-default select-none bg-white text-slate-700 sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Generate New Graph</DialogTitle>
          <DialogDescription className="text-sm text-slate-600">
            Generate a new attribution graph for a custom prompt. Powered by{' '}
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
          <p className="mt-1 rounded-md bg-amber-100 p-3 text-xs text-amber-700">
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
                  maxNLogits={values.maxNLogits}
                  desiredLogitProb={values.desiredLogitProb}
                  debouncedTokenize={debouncedTokenize}
                />
                <Form className="space-y-0">
                  <div>
                    <Label htmlFor="prompt" className="text-xs">
                      Prompt
                    </Label>
                    <p className="mt-0.5 text-[11.5px] text-slate-500">
                      In general, you want your prompt to be missing a word at the end, because we want to analyze how
                      the model comes up with the word <strong>after</strong> your prompt. (Eg &quot;The capital of the
                      state containing Dallas is&quot;)
                    </p>
                    <ReactTextareaAutosize
                      id="prompt"
                      name="prompt"
                      minRows={2}
                      value={values.prompt}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      onKeyDown={(e) => {
                        e.stopPropagation();
                      }}
                      disabled={isGenerating}
                      placeholder="Enter the prompt to visualize..."
                      className="mt-1 w-full resize-none rounded-md border border-slate-300 p-2 text-sm shadow-sm focus:border-sky-500 focus:ring-sky-500"
                      maxLength={GRAPH_MAX_PROMPT_LENGTH_CHARS}
                    />
                    {isTokenizing ? (
                      <div className="mt-1.5 flex w-full flex-row items-center justify-start gap-x-0.5 pb-2 text-xs text-sky-700">
                        <LoadingSquare className="mr-1 h-5 w-5" size={20} />
                        <div className="flex items-center justify-start">Tokenizing...</div>
                      </div>
                    ) : graphTokenizeResponse ? (
                      <div className="flex flex-col">
                        <div className="mx-0 mt-1 flex-1 pb-2 text-xs text-slate-500">
                          {/* <div className="mb-1">{graphTokenizeResponse.input_tokens.length} Tokens</div> */}
                          <div className="flex flex-wrap gap-x-1 gap-y-[3px]">
                            {graphTokenizeResponse.input_tokens.map((t, idx) => (
                              <span
                                key={`${t}-${idx}`}
                                className="mx-0 rounded bg-slate-200 px-[3px] py-[1px] font-mono text-[10px] text-slate-700"
                              >
                                {t.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                                {/* {t.replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}} */}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="mb-0.5 text-[8px] font-medium uppercase text-slate-500">Next Token</div>
                        <div className="mb-2 flex flex-wrap gap-x-1.5 gap-y-[3px]">
                          {graphTokenizeResponse.salient_logits.slice(0, 5).map((logit, index) => (
                            <span key={index} className="whitespace-pre rounded bg-slate-200 px-[7px] py-[3px] text-xs">
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
                              Warning: The next most likely token is a markdown/HTML/symbol. Double check that this is
                              the reasoning &apos;conclusion&apos; that you expect to see.
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
                  </div>

                  <div className="flex-1 pb-4">
                    <Label htmlFor="slug" className="text-xs text-slate-600">
                      Name Your Graph (Slug/ID)
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
                      className="mt-1 w-full border-slate-300 text-xs placeholder-slate-400"
                      maxLength={50}
                    />
                    {errors.slug && touched.slug && <p className="mt-1 text-xs text-red-500">{errors.slug}</p>}
                  </div>

                  <div className="hidden items-center pt-1 sm:flex">
                    <div className="mr-3 flex-1 border-t border-slate-200" />
                    <span className="text-[11px] text-slate-500">Advanced Settings</span>
                    <div className="ml-3 flex-1 border-t border-slate-200" />
                  </div>
                  <div className="hidden flex-row gap-x-3 pt-2 sm:flex">
                    <div className="flex-1">
                      <Label
                        htmlFor="modelId"
                        className="h-6 w-12 border-slate-300 px-0 text-left text-slate-600 md:text-[11px]"
                      >
                        Model
                      </Label>
                      <RadixSelect.Root
                        value={values.modelId}
                        onValueChange={(value: string) => setFieldValue('modelId', value)}
                        disabled={isGenerating}
                      >
                        <RadixSelect.Trigger
                          id="modelId"
                          className="mt-1 flex w-full items-center justify-between rounded-md border border-slate-300 bg-slate-200 px-3 py-2 text-xs text-slate-500 placeholder-slate-400 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          <RadixSelect.Value placeholder="Select a model" />
                          <RadixSelect.Icon className="text-slate-500">
                            <ChevronDownIcon className="h-4 w-4" />
                          </RadixSelect.Icon>
                        </RadixSelect.Trigger>
                        <RadixSelect.Portal>
                          <RadixSelect.Content className="z-50 min-w-[8rem] overflow-hidden rounded-md border border-slate-200 bg-white text-slate-900 shadow-md">
                            <RadixSelect.ScrollUpButton className="flex items-center justify-center py-1">
                              <ChevronUpIcon className="h-4 w-4" />
                            </RadixSelect.ScrollUpButton>
                            <RadixSelect.Viewport className="p-1">
                              {GRAPH_GENERATION_ENABLED_MODELS.map((model) => (
                                <RadixSelect.Item
                                  key={model}
                                  value={model}
                                  className="relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-4 pr-2 text-sm outline-none focus:bg-slate-100 data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                                >
                                  <RadixSelect.ItemText>{model}</RadixSelect.ItemText>
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
                        className="mt-1 w-full border-slate-300 text-center text-xs md:text-xs"
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
                  <div className="hidden space-y-0 pt-3 sm:block">
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
                  <div className="hidden space-y-0 pb-3 pt-3 sm:block">
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

                  {error && <p className="mt-4 rounded-md bg-red-100 p-3 text-sm text-red-600">{error}</p>}

                  <div className="mt-4 flex items-center justify-between border-t pt-4 text-xs">
                    {isGenerating ? (
                      countdownTime !== null ? (
                        <div className="flex flex-row items-center justify-start gap-x-2 whitespace-pre text-sm leading-none text-slate-500">
                          <LoadingSquare className="" size={32} />
                          Remaining time: {formatCountdown(countdownTime)}
                        </div>
                      ) : countdownTime === 0 ? (
                        <div className="text-sm text-slate-600">Remaining time: {formatCountdown(0)}</div>
                      ) : (
                        <div className="text-sm text-slate-600">Estimating time...</div>
                      )
                    ) : estimatedTime !== null && !generationResult ? (
                      <div className="flex flex-row items-center justify-start gap-x-1.5 whitespace-pre text-sm leading-none text-slate-500">
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
                          }
                        }}
                        disabled={isGenerating}
                        className="flex items-center justify-center gap-x-1.5"
                        title="Reset to defaults"
                      >
                        Reset
                      </Button>
                      <Button
                        type="submit"
                        disabled={
                          isGenerating ||
                          !dirty ||
                          Object.keys(errors).length > 0 ||
                          (graphTokenizeResponse !== null &&
                            graphTokenizeResponse.input_tokens.length > GRAPH_MAX_TOKENS)
                        }
                        className="w-full sm:w-auto"
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
          <div className="flex w-full flex-col space-y-1 pb-2">
            <h3 className="text-lg font-medium text-sky-700">Graph Generated Successfully</h3>
            <p>
              <strong>Nodes:</strong> {generationResult.numNodes}
            </p>
            <p>
              <strong>Links:</strong> {generationResult.numLinks}
            </p>
            <p className="pb-3 text-sm">
              <strong>URL:</strong>{' '}
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
            <Button
              onClick={() => {
                window.location.href = generationResult.url;
              }}
              className="mt-2 w-full flex-1 bg-sky-600 hover:bg-sky-700 sm:w-auto"
            >
              Open Graph
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
