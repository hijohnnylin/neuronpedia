'use client';

// internal tool for testing to see if dashboards were generated correctly
import cuid from 'cuid';
import { ActivationPartial, NeuronPartialWithRelations, NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useState } from 'react';
import FeatureDashboard from '../[modelId]/[layer]/[index]/feature-dashboard';
import { MANUAL_DASHBOARD_EXAMPLE_DATA } from './example-data';

type InternalFeatureActivation = {
  tokens: string[];
  values: number[];

  dfa_values: number[] | undefined;
  dfa_maxValue: number | undefined;
  dfa_targetIndex: number | undefined;

  bin_min: number | undefined;
  bin_max: number | undefined;
  bin_contains: number | undefined;
};

type InternalFeature = {
  model_id: string | undefined;
  layer: number | undefined;
  sae_set: string | undefined;
  feature_index: number;
  activations: InternalFeatureActivation[];
  neg_str: string[];
  neg_values: number[];
  pos_str: string[];
  pos_values: number[];
  frac_nonzero: number;
  freq_hist_data_bar_heights: number[];
  freq_hist_data_bar_values: number[];

  logits_hist_data_bar_heights: number[];
  logits_hist_data_bar_values: number[];

  decoder_weights_dist: number[] | undefined;

  n_prompts_total: number;
  n_tokens_in_prompt: number;
  dataset: string;

  neuron_alignment_indices: number[];
  neuron_alignment_values: number[];
  neuron_alignment_l1: number[];
  correlated_neurons_indices: number[];
  correlated_neurons_l1: number[];
  correlated_neurons_pearson: number[];
  correlated_features_indices: number[] | undefined;
  correlated_features_l1: number[] | undefined;
  correlated_features_pearson: number[] | undefined;
};

export default function ManualDashboard() {
  const [neuron, setNeuron] = useState<NeuronWithPartialRelations | undefined>(undefined);
  const [featureJson, setFeatureJson] = useState('');

  const defaultLayer = '1-sae_set';
  const defaultModel = 'model';
  const defaultSourceSetName = 'sae_set';

  function setValue(value: string) {
    if (value === '') {
      setFeatureJson('');
      setNeuron(undefined);
      return;
    }
    try {
      let feature: InternalFeature;
      const parsed = JSON.parse(value);
      if (!parsed) {
        throw new Error('Invalid JSON parse');
      }

      // eslint-disable-next-line prefer-const
      feature = parsed as InternalFeature;
      const index = feature.feature_index.toString();
      let maxActApprox = 0;
      feature.activations.forEach((a: InternalFeatureActivation) => {
        const max = Math.max(...a.values);
        if (max > maxActApprox) {
          maxActApprox = max;
        }
      });
      const feat: NeuronPartialWithRelations = {
        modelId: feature.model_id || defaultModel,
        layer: `${feature.layer || defaultLayer}-${feature.sae_set || defaultSourceSetName}`,
        index,
        creatorId: 'user',
        createdAt: new Date(),

        sourceSetName: feature.sae_set || defaultSourceSetName,
        maxActApprox,

        neg_str: feature.neg_str.map((s: string) => s.replaceAll('\u0000', 'NULL_CHAR')),
        neg_values: feature.neg_values,
        pos_str: feature.pos_str.map((s: string) => s.replaceAll('\u0000', 'NULL_CHAR')),
        pos_values: feature.pos_values,
        frac_nonzero: feature.frac_nonzero,
        freq_hist_data_bar_heights: feature.freq_hist_data_bar_heights,
        freq_hist_data_bar_values: feature.freq_hist_data_bar_values,
        logits_hist_data_bar_heights: feature.logits_hist_data_bar_heights,
        logits_hist_data_bar_values: feature.logits_hist_data_bar_values,

        decoder_weights_dist: feature.decoder_weights_dist || [],

        neuron_alignment_indices: feature.neuron_alignment_indices,
        neuron_alignment_values: feature.neuron_alignment_values,
        neuron_alignment_l1: feature.neuron_alignment_l1,
        correlated_neurons_indices: feature.correlated_neurons_indices,
        correlated_neurons_pearson: feature.correlated_neurons_pearson,
        correlated_neurons_l1: feature.correlated_neurons_l1,
        correlated_features_indices: feature.correlated_features_indices || [],
        correlated_features_pearson: feature.correlated_features_pearson || [],
        correlated_features_l1: feature.correlated_features_l1 || [],

        activations: [] as ActivationPartial[],
      };

      const activationsToAdd = [];

      for (const activation of feature.activations) {
        activationsToAdd.push({
          id: cuid(),
          tokens: activation.tokens,
          index,
          layer: `${feature.layer || defaultLayer}-${feature.sae_set || defaultSourceSetName}`,
          modelId: feature.model_id || defaultModel,
          creatorId: 'actuserid',
          createdAt: undefined,
          values: activation.values,
          dataIndex: null,
          dataSource: null,
          maxValue: Math.max(...activation.values),
          maxValueTokenIndex: activation.values.indexOf(Math.max(...activation.values)),
          minValue: Math.min(...activation.values),
          binMin: activation.bin_min,
          binMax: activation.bin_max,
          binContains: activation.bin_contains,
          dfaValues: activation.dfa_values,
          dfaTargetIndex: activation.dfa_targetIndex,
          dfaMaxValue: activation.dfa_maxValue,
        });
      }
      feat.activations = activationsToAdd;

      setNeuron(feat as NeuronWithPartialRelations);
      setFeatureJson(value);
    } catch (e) {
      alert(`Invalid JSON: ${e}`);
    }
  }

  return (
    <div className="flex w-full flex-col">
      <div className="flex w-full flex-row gap-x-2">
        <textarea
          className="h-20 w-full rounded border border-gray-300 p-2 text-xs"
          placeholder="Paste feature JSON here (this is one of the elements in a batch-x.json under 'features'). NOT the whole thing - it will likely be too large to paste."
          value={featureJson}
          onChange={(e) => {
            setFeatureJson(e.target.value);
          }}
        />
        <button
          type="button"
          className="rounded bg-slate-200 px-3 py-5 text-sm text-slate-600 disabled:opacity-50"
          disabled={neuron !== undefined || featureJson === ''}
          onClick={() => {
            setValue(featureJson);
          }}
        >
          Render
        </button>
        <button
          type="button"
          disabled={neuron !== undefined}
          className="rounded bg-slate-200 px-3 py-5 text-sm text-slate-600 disabled:opacity-50"
          onClick={() => {
            setValue(MANUAL_DASHBOARD_EXAMPLE_DATA);
          }}
        >
          Example
        </button>
        <button
          type="button"
          disabled={neuron === undefined}
          className="rounded bg-slate-200 px-3 py-5 text-sm text-slate-600 disabled:opacity-50"
          onClick={() => {
            setValue('');
          }}
        >
          Reset
        </button>
      </div>
      {neuron && <FeatureDashboard initialNeuron={neuron} />}
    </div>
  );
}
