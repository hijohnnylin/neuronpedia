/* eslint no-nested-ternary: 0 */

import ActivationItem from '@/components/activation-item';
import FeatureStats from '@/components/feature-stats';
import PanelLoader from '@/components/panel-loader';
import { useGlobalContext } from '@/components/provider/global-provider';
import { deleteFeatureFromUmapMap, UmapListItem, useUmapContext } from '@/components/provider/umap-provider';
import { NEXT_PUBLIC_URL } from '@/lib/env';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import * as Popover from '@radix-ui/react-popover';
import { Field, Form, Formik } from 'formik';
import { ArrowUpRight, CircleAlert, Pencil, X } from 'lucide-react';
import { useSession } from 'next-auth/react';
import {
  ExplanationWithPartialRelations,
  ListWithPartialRelations,
  NeuronWithPartialRelations,
} from 'prisma/generated/zod';

const NUM_ACTIVATIONS_TO_SHOW = 3;

export default function UmapSelectedItem({
  feature,
  listItem,
  listToUse,
  umap = false,
  hideDescription = false,
  style,
  isHovered = false,
}: {
  feature: NeuronIdentifier;
  listItem: UmapListItem;
  listToUse?: ListWithPartialRelations | null | undefined;
  umap?: boolean;
  hideDescription?: boolean;
  style?: any;
  isHovered?: boolean;
}) {
  const session = useSession();
  const {
    // addAnnotationForExp,
    setSelectedFeatures,
  } = useUmapContext();
  const { showToastMessage } = useGlobalContext();
  const { neuron } = listItem;

  // not loaded yet
  if (!listItem.neuron) {
    return (
      <div
        key={`${feature.modelId}-${feature.layer}-${feature.index}`}
        className="flex min-h-[320px] w-[300px] min-w-[300px] max-w-[300px] flex-col justify-start rounded border bg-slate-100 px-3 pb-20 pt-12"
      >
        <div className="text-left font-mono text-xs font-medium uppercase text-sky-700 hover:text-sky-600">
          {feature.layer}:{feature.index} ↗
        </div>
        <PanelLoader showBackground={false} />
      </div>
    );
  }

  return (
    <div
      style={style}
      className={`group mb-3 flex w-[300px] min-w-[300px] max-w-[300px] flex-col items-start rounded border border-slate-200 bg-slate-100 px-0 pb-2 hover:border-sky-600 ${
        isHovered && 'border-sky-600'
      } `}
      // eslint-disable-next-line
      onMouseOver={() => {
        if (neuron && neuron.explanations && neuron.explanations.length > 0) {
          // TODO: hack
          const toSet = neuron?.explanations[0] as ExplanationWithPartialRelations;
          toSet.neuron = {
            modelId: feature.modelId,
            layer: feature.layer,
            index: feature.index,
          };
          // addAnnotationForExp(toSet);
        }
      }}
      key={`${neuron?.modelId || ''}-${neuron?.layer || ''}-${neuron?.index || ''}`}
    >
      <a
        className={`mb-1 mt-0 flex w-full shrink-0 cursor-pointer flex-row items-start gap-x-1 overflow-x-scroll whitespace-nowrap rounded-t bg-slate-200 py-[6px] text-center font-mono text-[9px] font-medium leading-none text-slate-700 group-hover:bg-sky-200 group-hover:text-sky-700 ${
          isHovered && 'bg-sky-200 text-sky-700'
        } sm:py-2 sm:text-[10px]`}
        href={`${NEXT_PUBLIC_URL}/${neuron?.modelId}/${neuron?.layer}/${neuron?.index}`}
        target="_blank"
        rel="noreferrer"
      >
        <div className="flex w-full flex-row items-center justify-start px-2">
          {neuron?.modelId.toUpperCase()} · {neuron?.layer.toUpperCase()} · {neuron?.index.toUpperCase()}
          <ArrowUpRight className="ml-1 h-2.5 w-2.5" />
        </div>
      </a>
      <div className="mb-1.5 flex w-full flex-row items-start justify-between gap-x-2 px-2">
        {!listItem.isEditing ? (
          <div className="flex w-full flex-row justify-between gap-x-1.5 border-b border-slate-200 pt-1">
            {!hideDescription && (
              <div className="flex h-10 max-h-10 w-full flex-row items-start justify-between overflow-y-auto px-1 text-left font-sans text-[11.5px] font-medium leading-snug text-slate-700">
                {listItem.description ? (
                  <div className="flex flex-1 flex-row items-center justify-between gap-x-1.5">
                    {listItem.description}
                    <Popover.Root>
                      <Popover.Trigger asChild>
                        <button
                          type="button"
                          className="flex flex-row items-center justify-center text-center text-[10px] font-medium uppercase text-slate-500"
                        >
                          <CircleAlert className="h-3.5 w-3.5" />
                        </button>
                      </Popover.Trigger>
                      <Popover.Portal>
                        <Popover.Content
                          className="z-50 w-[360px] rounded-lg border bg-slate-100 p-3 px-4 shadow-lg"
                          sideOffset={-5}
                        >
                          <div className="flex flex-col gap-2">
                            <div className="flex flex-col text-xs font-medium text-slate-500">
                              This explanation was provided by the list author.
                            </div>
                            <div className="flex flex-row gap-x-2">
                              <Popover.Close
                                className="flex-1 rounded-lg bg-slate-300 py-2 text-center text-xs font-bold text-slate-600 outline-none hover:bg-slate-200 hover:text-slate-700"
                                aria-label="Close"
                              >
                                Close
                              </Popover.Close>
                            </div>
                          </div>
                        </Popover.Content>
                      </Popover.Portal>
                    </Popover.Root>
                  </div>
                ) : neuron && neuron.explanations && neuron.explanations.length > 0 ? (
                  neuron.explanations[0].description
                ) : (
                  ''
                )}
              </div>
            )}

            {/* if it's list mode and we're the owner, OR if it's umap mode */}
            {((listToUse && listToUse?.userId === session.data?.user.id) || umap) && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  setSelectedFeatures((previousMap) => {
                    const newMap = new Map(previousMap);
                    newMap.set(feature, {
                      isEditing: true,
                      description: listItem.description,
                      neuron: listItem.neuron,
                    });
                    return newMap;
                  });
                }}
                className="flex h-5 w-5 flex-row items-center justify-center gap-x-2 rounded-full bg-slate-200 p-1 text-xs font-medium text-slate-600 hover:bg-slate-400"
              >
                <Pencil className="h-3 w-3" />
              </button>
            )}

            {/* if it's list mode and we're the owner, OR if it's umap mode */}
            {((listToUse && listToUse?.userId === session.data?.user.id) || umap) && (
              <button
                type="button"
                onClick={async () => {
                  if (window.confirm('Remove feature from this list?')) {
                    if (listToUse) {
                      await fetch(`/api/list/remove`, {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                          listId: listToUse.id,
                          modelId: neuron?.modelId,
                          layer: neuron?.layer,
                          index: neuron?.index,
                        }),
                      });
                    }
                    setSelectedFeatures((previousMap) => {
                      const newMap = new Map(previousMap);
                      deleteFeatureFromUmapMap(feature, newMap);
                      return newMap;
                    });
                    showToastMessage('Removed feature.');
                  }
                }}
                className="flex h-5 w-5 flex-row items-center justify-center gap-x-2 rounded-full bg-slate-200 p-1 text-xs font-medium text-red-600 hover:bg-red-300"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        ) : (
          <Formik
            initialValues={{
              description: listItem.description,
            }}
            onSubmit={async (values) => {
              if (listToUse) {
                await fetch(`/api/list/edit-feature`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    listId: listToUse.id,
                    modelId: neuron?.modelId,
                    layer: neuron?.layer,
                    index: neuron?.index,
                    description: values.description,
                  }),
                });
              }
              setSelectedFeatures((previousMap) => {
                const newMap = new Map(previousMap);
                newMap.set(feature, {
                  isEditing: false,
                  description: values.description,
                  neuron: listItem.neuron,
                });
                return newMap;
              });
              showToastMessage('Saved description.');
            }}
          >
            {({ submitForm, setFieldValue, isSubmitting }) => (
              <Form
                onClick={(e) => {
                  e.stopPropagation();
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    submitForm();
                  }
                }}
                className="flex flex-1 flex-col gap-y-1.5"
              >
                <Field
                  type="text"
                  name="description"
                  disabled={isSubmitting}
                  required
                  placeholder="Override this explanation"
                  className="mt-0 flex-1 rounded border border-slate-200 px-2 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-xs"
                />

                <div className="flex flex-row gap-x-1">
                  <button
                    type="button"
                    disabled={isSubmitting}
                    onClick={async (e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setFieldValue('description', listItem.description);
                      setSelectedFeatures((previousMap) => {
                        const newMap = new Map(previousMap);
                        newMap.set(feature, {
                          isEditing: false,
                          description: listItem.description,
                          neuron: listItem.neuron,
                        });
                        return newMap;
                      });
                    }}
                    className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-slate-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                  >
                    Cancel
                  </button>
                  <button
                    disabled={isSubmitting}
                    type="submit"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      submitForm();
                    }}
                    className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-sky-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                  >
                    Save
                  </button>
                </div>
              </Form>
            )}
          </Formik>
        )}
      </div>
      {/* <div
        className={`mb-0 flex w-full flex-col items-center px-3 pb-0 text-left ${"justify-center gap-x-1"} text-right font-mono text-[10px] font-bold leading-[1.25] text-sky-700 sm:text-[11px]`}
      >
        <div className="flex h-10 max-h-10 w-full items-center overflow-y-scroll border-b border-slate-200 pb-2 text-left align-middle font-sans text-[11.5px] font-medium text-slate-700">
          {listItem.description
            ? listItem.description
            : neuron && neuron.explanations && neuron.explanations.length > 0
            ? neuron.explanations[0].description
            : ""}
        </div>
      </div> */}

      <div className="flex w-full flex-col px-3 py-1">
        {neuron?.pos_str && neuron?.pos_str.length > 0 && (
          <div className="pointer-events-none mb-0 mt-1.5 w-full pt-1 sm:flex">
            <FeatureStats currentNeuron={neuron as NeuronWithPartialRelations} vertical smallText />
          </div>
        )}
        <div
          className={`mb-1 mt-3 ${
            neuron?.pos_str && neuron?.pos_str.length > 0 ? 'border-t' : ''
          } border-slate-200 pt-1`}
        >
          {neuron?.activations?.slice(0, NUM_ACTIVATIONS_TO_SHOW).map((activation) => (
            <div key={activation.id} className="my-2 flex justify-start text-left">
              <ActivationItem
                activation={activation}
                tokensToDisplayAroundMaxActToken={8}
                enableExpanding={false}
                overrideLeading="leading-none"
                overrideTextSize="text-[11px]"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
