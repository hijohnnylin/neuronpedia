'use client';

// TODO: Clean up this file. it's got too much going on and has shared code with UMAP

import ActivationItem from '@/components/activation-item';
import PanelLoader from '@/components/panel-loader';
import { deleteFeatureFromUmapMap, UmapListItem, useUmapContext } from '@/components/provider/umap-provider';
import UmapSelectedItem from '@/components/umap/umap-selected-item';
import { env } from '@/lib/env';
import { useScreenSize } from '@/lib/hooks/use-screen-size';
import { MAX_LIST_FEATURES_FOR_TEST_TEXT, MAX_LIST_TEST_TEXT_LENGTH_CHARS } from '@/lib/utils/list';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import copy from 'copy-to-clipboard';
import { Field, Form, Formik } from 'formik';
import { Copy, LayoutDashboard, List as ListIcon, Pencil, WrapText, X } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next-nprogress-bar';
import { useSearchParams } from 'next/navigation';
import { ListWithPartialRelations, NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useMemo, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import { FixedSizeList } from 'react-window';

export default function List({ listId }: { listId: string }) {
  const screenSize = useScreenSize();
  const session = useSession();
  const router = useRouter();
  const [listView, setListView] = useState(false);
  const [showEdit, setShowEdit] = useState(false);
  const [showTestTextEdit, setShowTestTextEdit] = useState(false);
  const [list, setList] = useState<ListWithPartialRelations>();
  const [selectedFeaturesReversed, setSelectedFeaturesReversed] = useState<[NeuronIdentifier, UmapListItem][]>([]);
  const { selectedFeatures, setSelectedFeatures } = useUmapContext();
  const searchParams = useSearchParams();
  const isEmbed = searchParams.get('embed') === 'true';
  const fixedSizeListRef = useRef<FixedSizeList>(null);
  const [isSavingTestText, setIsSavingTestText] = useState(false);
  const [hoveredFeature, setHoveredFeature] = useState<NeuronIdentifier | null>(null);

  async function reloadList() {
    const response = await fetch(`/api/list/get`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        listId,
      }),
    });
    // make source selected indexes
    const respJson = (await response.json()) as ListWithPartialRelations;
    const newMap = new Map<NeuronIdentifier, UmapListItem>();
    respJson.neurons?.forEach((n) => {
      newMap.set(new NeuronIdentifier(n.neuron?.modelId, n.neuron?.layer, n.neuron?.index), {
        isEditing: false,
        description: n.description || '',
        neuron: n.neuron as NeuronWithPartialRelations,
      });
    });
    setSelectedFeatures(newMap);
    setSelectedFeaturesReversed(Array.from(newMap).toReversed());
    setList(respJson);
  }

  useEffect(() => {
    reloadList();
  }, []);

  useEffect(() => {
    if (showEdit) {
      setShowTestTextEdit(false);
    }
  }, [showEdit]);

  useEffect(() => {
    if (showTestTextEdit) {
      setShowEdit(false);
    }
  }, [showTestTextEdit]);

  useEffect(() => {
    setSelectedFeaturesReversed(Array.from(selectedFeatures).toReversed());
  }, [selectedFeatures]);

  function findSelectedFeatureDescription(feature: NeuronIdentifier) {
    // eslint-disable-next-line no-restricted-syntax
    for (const [key, value] of selectedFeatures) {
      if (key.equals(feature)) {
        if (value.description) {
          return value.description;
        }
        if (value.neuron?.explanations && value.neuron?.explanations.length > 0) {
          return value.neuron?.explanations[0].description;
        }
      }
    }
    return undefined;
  }

  function scrollToFeature(feature: NeuronIdentifier) {
    if (fixedSizeListRef.current) {
      fixedSizeListRef.current.scrollToItem(
        selectedFeaturesReversed.findIndex(([f]) => f.equals(feature)),
        'start',
      );
    }
  }

  const [wrap, setWrap] = useState(false);

  const detailView = useMemo(() => {
    // eslint-disable-next-line
    function Column({ data, index, style }: { data: [NeuronIdentifier, UmapListItem][]; index: number; style: any }) {
      return (
        <div
          onMouseEnter={() => {
            setHoveredFeature(data[index][0]);
          }}
          onMouseLeave={() => {
            setHoveredFeature(null);
          }}
        >
          <UmapSelectedItem
            feature={data[index][0]}
            listItem={data[index][1]}
            style={style}
            key={
              (data[index][1].neuron?.modelId || '') +
              (data[index][1].neuron?.layer || '') +
              (data[index][1].neuron?.index || '')
            }
            listToUse={list}
            isHovered={hoveredFeature ? hoveredFeature.equals(data[index][0]) : false}
          />
        </div>
      );
    }
    return (
      <div className="flex w-full max-w-screen-xl flex-row gap-x-2 overflow-x-scroll">
        {/* @ts-ignore */}
        <FixedSizeList
          height={400}
          width={Math.min(screenSize.width - 30, 1280)}
          itemData={selectedFeaturesReversed}
          itemSize={310}
          ref={fixedSizeListRef}
          layout="horizontal"
          itemCount={selectedFeaturesReversed.length}
          className={selectedFeaturesReversed.length > 4 ? 'forceShowScrollBarHorizontal' : ''}
        >
          {Column}
        </FixedSizeList>
      </div>
    );
  }, [screenSize.width, selectedFeaturesReversed, hoveredFeature]);

  return list ? (
    <div
      className={`flex w-full flex-col items-center justify-center gap-x-1 gap-y-2 px-3 xl:gap-x-2 ${
        isEmbed ? 'mt-2' : 'mt-5'
      }`}
    >
      <div className="mb-1 flex w-full max-w-screen-xl flex-row items-center justify-start">
        {list.userId === session.data?.user?.id && showEdit ? (
          <Formik
            initialValues={{
              name: list.name,
              description: list.description,
              defaultTestText: list.defaultTestText,
            }}
            onSubmit={async (values) => {
              setIsSavingTestText(true);
              if (values.name.trim().length === 0) {
                alert('List name cannot be blank.');
                setIsSavingTestText(false);
                return;
              }
              if (values.defaultTestText) {
                if (values.defaultTestText.length > MAX_LIST_TEST_TEXT_LENGTH_CHARS) {
                  alert(`Default test text is too long. Max is ${MAX_LIST_TEST_TEXT_LENGTH_CHARS} characters`);
                  setIsSavingTestText(false);
                  return;
                }
                if (list.neurons && list.neurons.length > MAX_LIST_FEATURES_FOR_TEST_TEXT) {
                  alert(`List has too many neurons. Max is ${MAX_LIST_FEATURES_FOR_TEST_TEXT}`);
                  setIsSavingTestText(false);
                  return;
                }
                if (values.defaultTestText.trim().length === 0) {
                  // eslint-disable-next-line no-param-reassign
                  values.defaultTestText = null;
                }
              }
              await fetch(`/api/list/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  listId: list.id,
                  name: values.name,
                  description: values.description,
                  defaultTestText: values.defaultTestText,
                }),
              });
              list.name = values.name;
              list.description = values.description;
              list.defaultTestText = values.defaultTestText;
              reloadList();
              setIsSavingTestText(false);
              setShowEdit(false);
            }}
          >
            {({ submitForm, values, setFieldValue }) => (
              <Form className="mb-3 flex flex-1 flex-col gap-y-1 rounded border border-slate-200 bg-slate-100 px-3 py-2">
                <div className="w-full text-center text-base font-medium text-slate-700">Edit List Metadata</div>
                <label htmlFor="name" className="flex flex-col gap-y-1 text-xs font-medium text-slate-700">
                  List Name
                  <Field
                    type="text"
                    name="name"
                    disabled={isSavingTestText}
                    required
                    placeholder="New List Name"
                    className="mb-1 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                  />
                </label>

                <label htmlFor="defaultTestText" className="flex flex-col gap-y-1 text-xs font-medium text-slate-700">
                  Text to Test List Features (Optional)
                  <ReactTextareaAutosize
                    name="defaultTestText"
                    value={values.defaultTestText || ''}
                    minRows={3}
                    onChange={(e) => {
                      setFieldValue('defaultTestText', e.target.value);
                    }}
                    required
                    disabled={isSavingTestText}
                    placeholder="Optional text to test features with."
                    className="mt-0 flex-1 resize-none rounded border border-slate-200 px-3 py-2 text-left font-mono text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200"
                  />
                </label>

                <div className="mx-auto mt-1 flex min-w-[280px] max-w-screen-md flex-row gap-x-1">
                  <button
                    type="button"
                    disabled={isSavingTestText}
                    onClick={(e) => {
                      e.preventDefault();
                      setFieldValue('name', list.name);
                      setFieldValue('description', list.description);
                      setFieldValue('defaultTestText', list.defaultTestText);
                      setShowEdit(false);
                    }}
                    className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-slate-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    disabled={isSavingTestText}
                    onClick={async (e) => {
                      e.preventDefault();
                      if (window.confirm('Are you sure you want to delete this list? You cannot undo this action.')) {
                        await fetch(`/api/list/delete`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({
                            listId: list.id,
                          }),
                        });
                        router.push(`/user/${session.data?.user?.name}/lists`);
                      }
                    }}
                    className="mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-red-500 py-2 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                  >
                    Delete List
                  </button>

                  <button
                    type="submit"
                    disabled={isSavingTestText}
                    onClick={(e) => {
                      e.preventDefault();
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
        ) : (
          <div className="flex flex-1 flex-row items-center justify-between gap-x-2">
            <div className="flex flex-row gap-x-2">
              {list.userId === session.data?.user?.id && (
                <button
                  type="button"
                  onClick={() => {
                    setShowEdit(true);
                  }}
                  className="flex h-7 w-7 cursor-pointer flex-row items-center justify-center gap-x-2 rounded-full bg-slate-200 p-1 text-xs font-medium text-slate-600 hover:bg-slate-400"
                >
                  <Pencil className="h-4 w-4" />
                </button>
              )}
              <div className="flex flex-col text-left">
                <div className="flex-1 text-sm font-medium text-slate-700 sm:text-base">
                  {list.name} <span className="text-slate-400">({list.neurons?.length})</span>
                </div>
              </div>
            </div>
            {list.defaultTestText && (
              <button
                type="button"
                onClick={() => {
                  setWrap(!wrap);
                }}
                className="mr-1 flex items-center justify-center"
              >
                <WrapText
                  className={`h-8 w-8 rounded-md p-1.5 ${
                    wrap
                      ? 'bg-slate-600 text-white hover:bg-slate-400'
                      : 'bg-slate-200 text-slate-600 hover:bg-slate-400'
                  }`}
                />
              </button>
            )}
          </div>
        )}
      </div>

      {list.defaultTestText && (
        <div
          className={`mb-2 flex w-full max-w-screen-xl flex-col items-start gap-x-2 gap-y-1.5 rounded pt-1 ${
            wrap ? 'pb-1' : 'forceShowScrollBarHorizontal overflow-x-scroll whitespace-nowrap pb-2'
          }`}
        >
          {list.activations?.toReversed().map((listOnActivation) => {
            const { activation } = listOnActivation;
            // get the overall max value for all list.activations
            const overallMaxValue = Math.max(...(list.activations?.map((la) => la.activation?.maxValue || 0) || []));
            if (!activation) {
              return null;
            }
            return (
              <div
                key={activation?.id}
                className={`flex flex-row items-center ${wrap ? '' : 'whitespace-nowrap'}`}
                onMouseEnter={() => {
                  scrollToFeature(new NeuronIdentifier(activation?.modelId, activation?.layer, activation?.index));
                  setHoveredFeature(new NeuronIdentifier(activation?.modelId, activation?.layer, activation?.index));
                }}
              >
                <div className="sticky left-0 flex w-[290px] max-w-[290px] flex-row items-center gap-x-1">
                  <a
                    className={`mt-0 flex w-[290px] max-w-[290px] flex-col items-center gap-x-1 rounded bg-slate-200 px-2 py-[6px] text-center font-mono text-[9px] font-medium leading-snug text-slate-700 hover:bg-sky-200 hover:text-sky-700 sm:py-1 sm:text-[10px] ${
                      hoveredFeature &&
                      hoveredFeature.equals(
                        new NeuronIdentifier(activation?.modelId, activation?.layer, activation?.index),
                      )
                        ? 'border border-sky-600 bg-sky-200 text-sky-700'
                        : 'border border-slate-200 hover:border-sky-600'
                    }`}
                    href={`/${activation?.modelId}/${activation?.layer}/${activation?.index}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <div className="mb-[1px] flex w-full flex-row items-center justify-start whitespace-normal text-left font-sans">
                      {findSelectedFeatureDescription(
                        new NeuronIdentifier(activation?.modelId, activation?.layer, activation?.index),
                      )}
                    </div>
                    <div className="flex w-full flex-row items-center justify-end whitespace-nowrap text-right text-[7px] text-slate-500">
                      {activation?.modelId?.toUpperCase()} · {activation?.layer?.toUpperCase()} ·{' '}
                      {activation?.index?.toUpperCase()}
                    </div>
                  </a>
                </div>
                <div className="ml-3">
                  <ActivationItem
                    activation={activation}
                    overallMaxActivationValueInList={overallMaxValue}
                    overrideTextSize="text-[11px]"
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
      <div className="flex w-full max-w-screen-xl flex-col gap-x-3 sm:flex-row">
        <div className="flex flex-col items-start justify-start gap-y-1 pb-2 text-slate-600">
          {listView ? (
            <div className="table w-full table-auto">
              <thead>
                <tr className="text-left text-[12px] font-normal text-slate-400">
                  <th className="pb-2 pr-2 font-normal">Description</th>
                  <th className="px-2 pb-2 font-normal">Feature</th>
                  <th className="px-2 pb-2 font-normal">Explanation</th>
                  <th className="px-2 pb-2 font-normal">Activation Density</th>
                </tr>
              </thead>
              {Array.from(selectedFeatures)
                .toReversed()
                .map(([feature, listItem]) => {
                  const { neuron } = listItem;
                  return (
                    <tr key={feature.modelId + feature.layer + feature.index} className="rounded text-left text-sm">
                      <td className="my-auto w-32 align-middle sm:w-48 lg:w-64">
                        {!listItem.isEditing ? (
                          <div className="flex w-full flex-row justify-between gap-x-2">
                            <div className="flex flex-1 flex-row items-center justify-start gap-x-2">
                              {list.userId === session.data?.user?.id && (
                                <>
                                  <button
                                    type="button"
                                    onClick={async () => {
                                      if (window.confirm('Remove feature from this list?')) {
                                        if (list) {
                                          await fetch(`/api/list/remove`, {
                                            method: 'POST',
                                            headers: {
                                              'Content-Type': 'application/json',
                                            },
                                            body: JSON.stringify({
                                              listId: list.id,
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
                                      }
                                    }}
                                    className="flex h-5 w-5 flex-row items-center justify-center gap-x-2 rounded-full bg-slate-200 p-1 text-xs font-medium text-red-600 hover:bg-red-300"
                                  >
                                    <X className="h-3 w-3" />
                                  </button>
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
                                    className="mr-1 flex h-5 w-5 flex-row items-center justify-center gap-x-2 rounded-full bg-slate-200 p-1 text-xs font-medium text-slate-600 hover:bg-slate-400"
                                  >
                                    <Pencil className="h-3 w-3" />
                                  </button>
                                </>
                              )}
                              {listItem.description ? (
                                <div className="flex-1 pr-2 text-left text-xs font-medium leading-snug text-slate-700 sm:text-[13px]">
                                  {listItem.description}
                                </div>
                              ) : (
                                <div className="pr-2 text-xs text-slate-400">No description provided.</div>
                              )}
                            </div>
                          </div>
                        ) : (
                          <Formik
                            initialValues={{
                              description: listItem.description,
                            }}
                            onSubmit={async (values) => {
                              if (list) {
                                await fetch(`/api/list/edit-feature`, {
                                  method: 'POST',
                                  headers: {
                                    'Content-Type': 'application/json',
                                  },
                                  body: JSON.stringify({
                                    listId: list.id,
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
                                  placeholder="Describe the relevance/context"
                                  className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
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
                      </td>
                      <td
                        onClick={(e) => {
                          e.stopPropagation();
                          window.open(`/${neuron?.modelId}/${neuron?.layer}/${neuron?.index}`, '_blank', 'noreferrer');
                        }}
                        className="cursor-pointer whitespace-pre px-1 py-2.5 align-middle font-mono text-[10px] font-medium uppercase text-sky-800 hover:text-sky-700 sm:px-2 sm:text-xs"
                      >
                        <span className="hidden sm:inline-block">{neuron?.modelId}@</span>
                        {neuron?.layer}:{neuron?.index}
                      </td>
                      <td className="px-1 py-2.5 text-left align-middle text-[11px] font-medium leading-tight text-slate-600 sm:px-2 sm:text-xs">
                        {neuron?.explanations && neuron?.explanations.length > 0 && neuron?.explanations[0].description}
                      </td>
                      <td className="px-1 py-2.5 text-left align-middle text-[11px] font-medium leading-tight text-slate-600 sm:px-2 sm:text-xs">
                        {((neuron?.frac_nonzero ? neuron?.frac_nonzero || 0 : 0) * 100).toFixed(3)}%
                      </td>
                    </tr>
                  );
                })}
            </div>
          ) : (
            detailView
          )}
        </div>
      </div>
      <div className="mb-3 flex w-full max-w-screen-xl items-center justify-between gap-x-1.5">
        <ToggleGroup.Root
          className="inline-flex h-7 overflow-hidden rounded border-slate-300 bg-slate-200 px-0 py-0 sm:rounded"
          type="single"
          defaultValue={listView ? 'listView' : 'detailView'}
          value={listView ? 'listView' : 'detailView'}
          onValueChange={(value) => {
            setListView(value === 'listView');
          }}
        >
          <ToggleGroup.Item
            key="listView"
            className="flex flex-auto flex-row items-center gap-x-1.5 px-3 py-0.5 text-[10px] font-medium text-slate-500 transition-all hover:bg-sky-200 data-[state=on]:bg-sky-300 data-[state=on]:text-slate-700 sm:px-4 sm:text-[11px]"
            value="listView"
            aria-label="listView"
          >
            <ListIcon className="hidden h-4 w-4 sm:flex" />
            List View
          </ToggleGroup.Item>
          <ToggleGroup.Item
            key="detailView"
            className="flex flex-auto flex-row items-center gap-x-1.5 px-3 py-0.5 text-[10px] font-medium text-slate-500 transition-all hover:bg-sky-200 data-[state=on]:bg-sky-300 data-[state=on]:text-slate-700 sm:px-4 sm:text-[11px]"
            value="detailView"
            aria-label="detailView"
          >
            <LayoutDashboard className="hidden h-4 w-4 sm:flex" /> Detail View
          </ToggleGroup.Item>
        </ToggleGroup.Root>
        {!isEmbed && (
          <div className="flex flex-row items-center justify-center gap-x-1.5">
            <div className="flex flex-row items-center justify-center gap-x-1.5">
              <button
                type="button"
                className="flex h-7 flex-row items-center justify-center gap-x-1.5 rounded bg-slate-200 px-2 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-300 active:bg-sky-300 sm:text-[11px]"
                onClick={(e) => {
                  e.preventDefault();
                  copy(
                    `<iframe src="${env.NEXT_PUBLIC_URL}/list/${list.id}?embed=true" title="Neuronpedia" style="height: 400px; width: 540px;"></iframe>`,
                  );
                  alert('Copied iFrame embed code to clipboard.');
                }}
              >
                <Copy className="h-3.5 w-3.5" /> iFrame Embed
              </button>
            </div>
            <div className="flex flex-row items-center justify-center gap-x-1.5">
              <button
                type="button"
                className="flex h-7 flex-row items-center justify-center gap-x-1.5 rounded bg-slate-200 px-2 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-300 active:bg-sky-300 sm:text-[11px]"
                onClick={(e) => {
                  e.preventDefault();
                  copy(`${env.NEXT_PUBLIC_URL}/list/${list.id}?embed=true`);
                  alert('Copied embed link to clipboard.');
                }}
              >
                <Copy className="h-3.5 w-3.5" /> Link Embed
              </button>
            </div>
          </div>
        )}
      </div>
      {/* <div
        className={`mx-auto flex w-full max-w-screen-sm flex-col items-start justify-center gap-x-3 pb-10 sm:flex-row`}
      >
        <div className="flex flex-1 flex-col overflow-hidden rounded border border-slate-200 bg-white sm:basis-1/3">
          {list.comments?.length === 0 ? (
            <div className="py-5 text-center text-sm font-bold text-slate-300">
              No comments. Add one below.
            </div>
          ) : (
            <div className="flex min-h-[100px] flex-col gap-y-1 py-2 pl-3 pr-2 text-xs text-slate-600">
              {list.comments?.map((comment) => {
                return (
                  <div key={comment.id} className="flex w-full flex-col pb-0.5">
                    <div className="font-bold">{comment.user?.name}</div>
                    <div className="flex w-full flex-1 flex-row pl-1">
                      <div className="flex-1">{comment.text}</div>
                      {comment.userId === session.data?.user.id && (
                        <button
                          onClick={async () => {
                            if (
                              confirm(
                                "Are you sure you want to delete this comment?",
                              )
                            ) {
                              await fetch(`/api/list/comment/delete`, {
                                method: "POST",
                                headers: {
                                  "Content-Type": "application/json",
                                },
                                body: JSON.stringify({
                                  commentId: comment.id,
                                }),
                              });
                              reloadList();
                            }
                          }}
                          className=""
                        >
                          <X className="h-3 w-3 text-red-700" />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <Formik
            initialValues={{ listId: list.id, text: "" }}
            onSubmit={async (values, { resetForm }) => {
              if (!session.data?.user) {
                setSignInModalOpen(true);
                return;
              }
              if (values.text.trim().length < 2) {
                alert("Comment too short.");
                return;
              }
              await fetch(`/api/list/comment/new`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  listId: list.id,
                  text: values.text,
                }),
              });
              resetForm();
              reloadList();
            }}
          >
            {({ submitForm, values, setFieldValue }) => (
              <Form
                className={`flex w-full flex-row items-center justify-center`}
              >
                <Field
                  type="text"
                  name="text"
                  required
                  placeholder={`Comment Text`}
                  className={`mt-0 flex-1 border-0 bg-slate-100 px-3 py-3 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]`}
                />
                <button
                  type="submit"
                  onClick={(e) => {
                    e.preventDefault();
                    submitForm();
                  }}
                  className={`mb-0 mt-0 flex w-16 items-center justify-center overflow-hidden bg-slate-500 py-3 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400`}
                >
                  Add
                </button>
              </Form>
            )}
          </Formik>
        </div>
      </div> */}
    </div>
  ) : (
    <div>
      <PanelLoader showBackground={false} />
    </div>
  );
}
