// Server reads of neuron dashboards stored as files (SourceSet.externalDashboardsUrl).
// Routes that read Activation rows or write rows that reference a Neuron use these helpers.
// Such neurons have no Activation rows, and no Neuron row until something is written for them.

import { notFound } from '@/lib/api-error';
import { prisma } from '@/lib/db';
import { fetchNeuronDashboard, NeuronDashboard, NeuronDashboardError } from '@/lib/utils/neuron-dashboards';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { ActivationPartialWithRelations } from '@/prisma/generated/zod';
import { Activation, Neuron } from '@prisma/client';

type FeatureIds = { modelId: string; layer: string; index: string };

// The dashboard from files, or null when the feature's source set keeps its dashboards in the database.
export async function getExternalNeuronDashboard(ids: FeatureIds): Promise<NeuronDashboard | null> {
  const sourceSet = await prisma.sourceSet.findUnique({
    where: { modelId_name: { modelId: ids.modelId, name: getSourceSetNameFromSource(ids.layer) } },
    select: { externalDashboardsUrl: true },
  });
  if (!sourceSet?.externalDashboardsUrl) {
    return null;
  }
  try {
    return await fetchNeuronDashboard(sourceSet.externalDashboardsUrl, ids);
  } catch (error) {
    if (error instanceof NeuronDashboardError) {
      throw notFound(`No dashboard for ${ids.modelId}/${ids.layer}/${ids.index}.`);
    }
    throw error;
  }
}

function toActivationRow(activation: ActivationPartialWithRelations, ids: FeatureIds): Activation {
  return {
    id: activation.id || '',
    tokens: activation.tokens || [],
    dataIndex: activation.dataIndex ?? null,
    index: ids.index,
    layer: ids.layer,
    modelId: ids.modelId,
    dataSource: activation.dataSource ?? null,
    maxValue: activation.maxValue ?? 0,
    maxValueTokenIndex: activation.maxValueTokenIndex ?? 0,
    minValue: activation.minValue ?? 0,
    values: activation.values || [],
    dfaValues: [],
    dfaTargetIndex: null,
    dfaMaxValue: null,
    zQIndices: [],
    zKIndices: [],
    zValues: [],
    creatorId: '',
    createdAt: new Date(),
    lossValues: [],
    logitContributions: null,
    binMin: activation.binMin ?? null,
    binMax: activation.binMax ?? null,
    binContains: activation.binContains ?? null,
    qualifyingTokenIndex: null,
  };
}

// Share of the neuron's largest |activation| below which a whole example counts as non-activating.
export const NEAR_ZERO_FRACTION = 0.01;

// Scorers take rows with maxValue 0 as non-activating examples. Dashboard files have none, but their
// bands nearest zero do. Rows whose every value is below NEAR_ZERO_FRACTION of the largest |act| get
// maxValue 0 and zero values. Use this only for scoring: it changes the stored values.
export function markNearZeroAsNonActivating(activations: Activation[], dashboard: NeuronDashboard): Activation[] {
  const { max, min } = dashboard.record.stats;
  const limit = NEAR_ZERO_FRACTION * Math.max(Math.abs(max), Math.abs(min));
  return activations.map((a) =>
    a.values.length > 0 && a.values.every((v) => Math.abs(v) < limit)
      ? { ...a, maxValue: 0, minValue: 0, values: a.values.map(() => 0) }
      : a,
  );
}

// A feature's activations, from its dashboard files when it has them, otherwise from the database.
export async function getActivationsForFeature(
  ids: FeatureIds,
): Promise<{ activations: Activation[]; dashboard: NeuronDashboard | null }> {
  const dashboard = await getExternalNeuronDashboard(ids);
  if (dashboard) {
    return { activations: dashboard.activations.map((a) => toActivationRow(a, ids)), dashboard };
  }
  const activations = await prisma.activation.findMany({
    where: { neuron: { modelId: ids.modelId, layer: ids.layer, index: ids.index } },
  });
  return { activations, dashboard: null };
}

// Rows that reference a feature (explanations, for example) need its Neuron row. Creates a bare one
// if it is missing and returns it with the dashboard's stats and logits on top.
export async function ensureNeuronRowForDashboard(ids: FeatureIds, dashboard: NeuronDashboard): Promise<Neuron> {
  const key = { modelId: ids.modelId, layer: ids.layer, index: ids.index };
  const row = await prisma.neuron.upsert({
    where: { modelId_layer_index: key },
    create: {
      ...key,
      sourceSetName: getSourceSetNameFromSource(ids.layer),
      maxActApprox: dashboard.neuron.maxActApprox,
    },
    update: {},
  });
  return { ...row, ...dashboard.neuron, maxActApprox: dashboard.neuron.maxActApprox ?? row.maxActApprox };
}

// Call before writing a row that references a feature (comment, bookmark, list entry, explanation).
// When its source set keeps dashboards in files and it has no Neuron row, creates one. Reading the
// file first also checks that the feature exists.
export async function ensureNeuronRowForFeature(ids: FeatureIds): Promise<void> {
  const key = { modelId: ids.modelId, layer: ids.layer, index: ids.index };
  const existing = await prisma.neuron.findUnique({ where: { modelId_layer_index: key }, select: { index: true } });
  if (existing) {
    return;
  }
  const dashboard = await getExternalNeuronDashboard(ids);
  if (dashboard) {
    await ensureNeuronRowForDashboard(ids, dashboard);
  }
}
