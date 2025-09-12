'use client';

// TODO: fix eslint
// eslint-disable-next-line
import ActivationItem from '@/components/activation-item';
import { useGlobalContext } from '@/components/provider/global-provider';
import { env } from '@/lib/env';
import { BOS_TOKENS } from '@/lib/utils/activations';
import { Activation } from '@prisma/client';
import copy from 'copy-to-clipboard';
import { Check, Copy, Joystick, Play, Share } from 'lucide-react';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import { Button } from './shadcn/button';

const DEFAULT_STEER_MULTIPLIER = 3;

export default function ActivationSingleForm({
  neuron,
  overallMaxValue,
  formValue,
  enterSubmits = false,
  placeholder = undefined,
  callback = undefined,
  hideBos = false,
  embed = false,
  hideSteer = false,
}: {
  neuron: NeuronWithPartialRelations;
  overallMaxValue: number;
  formValue: string;
  enterSubmits?: boolean;
  placeholder?: string;
  callback?: (newActivation?: Activation) => void;
  hideBos?: boolean;
  embed?: boolean;
  hideSteer?: boolean;
}) {
  const { getSourceSet, showToastServerError, showToastMessage, isGraphEnabledForSource } = useGlobalContext();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [customText, setCustomText] = useState('');
  const [activationResult, setActivationResult] = useState<Activation | undefined>();
  const [copyClicked, setCopyClicked] = useState(false);

  useEffect(() => {
    if (copyClicked) {
      setTimeout(() => {
        setCopyClicked(false);
      }, 2000);
    }
  }, [copyClicked]);

  const testClicked = (text: string) => {
    setIsSubmitting(true);
    if (text.trim().length === 0) {
      alert('Please enter some text.');
      setIsSubmitting(false);
      return;
    }

    fetch(`/api/activation/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customText: text,
        neuron,
      }),
    })
      .then((response) => {
        if (response.status === 429 || response.status === 405) {
          alert('Sorry, we are limiting each user to 250 messages per hour. Please try again later.');
          return null;
        }
        if (response.status !== 200) {
          alert('Sorry, your request could not be completed at this time. Please try again later.');
          return null;
        }
        return response.json();
      })
      .then((newActivation) => {
        if (newActivation === null) {
          setIsSubmitting(false);
        } else {
          setIsSubmitting(false);
          const actToSet = newActivation;
          if (hideBos) {
            // TODO: do this in the inference instance instead
            if (newActivation.tokens.length > 0 && BOS_TOKENS.includes(newActivation.tokens[0])) {
              actToSet.tokens.shift();
              actToSet.values.shift();
              actToSet.maxValue = Math.max(...actToSet.values);
              actToSet.maxValueTokenIndex = actToSet.values.indexOf(actToSet.maxValue);
              if (newActivation.dfaValues !== undefined) {
                actToSet.dfaValues.shift();
                actToSet.dfaMaxValue = Math.max(...actToSet.dfaValues);
                actToSet.dfaTargetIndex = actToSet.values.indexOf(actToSet.dfaMaxValue);
              }
            }
          }
          setActivationResult(actToSet);
          if (callback) {
            callback(actToSet);
          }
        }
      })
      .catch(() => {
        setIsSubmitting(false);
        showToastServerError();
        if (callback) {
          callback();
        }
      });
  };

  useEffect(() => {
    setCustomText(formValue);
    if (formValue.trim().length > 0) {
      testClicked(formValue);
    }
  }, [formValue]); // FIX: can't add testClicked because it will cause a loop

  function makeActivationTextColor(overallMaxActivationTextValue: number, value: number, min = 0) {
    const realMax = overallMaxActivationTextValue - 0;
    const realCurrent = value + Math.abs(0);
    const opacity = realCurrent / realMax;
    return `rgba(5, 150, 105, ${Math.max(opacity, min)})`;
  }

  return (
    <div
      className={`flex w-full flex-row items-center justify-center gap-x-1.5 sm:border-0 ${
        activationResult ? 'border-b' : ''
      } ${enterSubmits ? '' : 'px-2 sm:px-3'} `}
    >
      <div className="flex w-full flex-col pb-1.5 pt-1.5 sm:pb-2 sm:pt-3">
        <div className="flex w-full flex-row gap-x-1.5">
          <div className="flex flex-1 flex-row gap-0 overflow-hidden rounded border border-sky-800">
            <label
              htmlFor="customText"
              className="mt-0 block w-full grow"
              aria-label="Test activation with custom text"
            >
              <ReactTextareaAutosize
                id="customText"
                name="customText"
                required
                value={customText}
                minRows={1}
                onChange={(e) => {
                  if (enterSubmits && e.target.value.indexOf('\n') !== -1) {
                    testClicked(e.target.value);
                  } else {
                    setCustomText(e.target.value);
                  }
                }}
                className="form-input mt-0 block w-full flex-1 resize-none rounded-l border-0 border-slate-300 px-2.5 py-2 font-mono text-[11px] leading-tight text-slate-700 placeholder-slate-400 focus:border-slate-300 focus:outline-0 focus:ring-0 sm:h-24 sm:px-3 sm:text-xs"
                placeholder={placeholder || `Test activation with custom text.`}
              />
            </label>
            <button
              type="button"
              onClick={() => {
                testClicked(customText);
              }}
              disabled={isSubmitting}
              className="flex w-[54px] min-w-[54px] flex-1 flex-col items-center justify-center gap-y-0.5 bg-sky-800 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-sky-600 hover:text-white disabled:bg-slate-300 disabled:text-slate-400"
            >
              <Play className="h-3.5 w-3.5" />
              Test
            </button>
          </div>

          {!hideSteer && !isGraphEnabledForSource(neuron.modelId, neuron.layer) && (
            <Button
              className="flex h-auto flex-col gap-y-0.5 border-emerald-700 px-2.5 text-[11px] text-xs font-medium text-emerald-700 hover:bg-emerald-50 hover:text-emerald-800"
              variant="outline"
              onClick={() => {
                window.open(
                  `/${neuron.modelId}/steer?source=${neuron.layer}&index=${neuron.index}${neuron.activations && neuron.activations.length > 0 ? `&strength=${neuron.activations?.[0]?.maxValue ? ((neuron.activations?.[0]?.maxValue || 0) * DEFAULT_STEER_MULTIPLIER).toFixed(1) : 10}` : ''}`,
                  '_blank',
                );
              }}
            >
              <Joystick className="h-4 w-4" /> Steer
            </Button>
          )}
        </div>
        {activationResult && (
          <div className="mb-1 mt-2 flex w-full flex-col rounded-md border-0 bg-slate-50 pb-1 pr-2 pt-0.5 sm:pb-2 sm:pt-1">
            <div className="flex w-full flex-row pb-0.5 pt-1 sm:pb-1 sm:pt-2">
              <div className="hidden max-h-72 shrink-0 flex-col items-center justify-center overflow-y-scroll text-center sm:flex">
                <div className="px-5 font-mono text-[10px] font-medium text-slate-500">
                  {activationResult.maxValueTokenIndex !== undefined && activationResult.tokens !== undefined ? (
                    <span>{activationResult.tokens[activationResult.maxValueTokenIndex]}</span>
                  ) : (
                    ''
                  )}
                </div>
                <div
                  className="px-5 font-mono text-[10px] font-bold"
                  style={{
                    color: makeActivationTextColor(1, 1, 0.5),
                  }}
                >
                  {activationResult.maxValue.toFixed(2)}
                </div>
              </div>
              <div className="max-h-72 flex-auto overflow-y-scroll px-3 text-sm sm:px-0">
                <ActivationItem
                  activation={activationResult}
                  overallMaxActivationValueInList={
                    overallMaxValue === -100 ? activationResult.maxValue : overallMaxValue
                  }
                  overrideTextSize="text-[10px] sm:text-xs"
                  showLineBreaks
                  dfa={getSourceSet(neuron.modelId, neuron.sourceSetName || '')?.showDfa}
                />
              </div>
              {!embed && (
                <div className="mt-0 flex flex-row items-center justify-start">
                  <button
                    type="button"
                    className="my-1 ml-3 flex w-[72px] cursor-pointer flex-row items-center justify-center gap-x-0.5 whitespace-pre rounded bg-slate-200 px-2 py-1.5 text-[9px] font-medium text-slate-600 hover:bg-slate-300 sm:px-2.5 sm:py-1.5 sm:text-[10.5px]"
                    title="Share Custom Activation Test Result"
                    onClick={() => {
                      const url = `${env.NEXT_PUBLIC_URL}/${neuron.modelId}/${neuron.layer}/${
                        neuron.index
                      }?defaulttesttext=${encodeURIComponent(activationResult.tokens.join(''))}`;
                      copy(url);
                      setCopyClicked(true);
                      showToastMessage(
                        <div className="flex flex-col items-center justify-center gap-y-1">
                          <div className="flex flex-row items-center justify-center gap-x-1 font-semibold">
                            <Copy className="h-4 w-4" /> Copied!
                          </div>
                          <div className="mt-1 text-xs">
                            The link to this feature, including this activation result, has been copied to your
                            clipboard.
                          </div>
                        </div>,
                      );
                    }}
                  >
                    {copyClicked ? (
                      <>
                        <Check className="h-3 w-3" /> Copied
                      </>
                    ) : (
                      <>
                        <Share className="h-3 w-3" /> Share
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
