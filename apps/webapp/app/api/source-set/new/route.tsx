import { getModelById } from '@/lib/db/model';
import { createSourceSetWithSources, getSourceSet } from '@/lib/db/source';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { Visibility } from '@prisma/client';
import { NextResponse } from 'next/server';

import * as yup from 'yup';

const NewSourceSetRequestSchema = yup.object({
  name: yup
    .string()
    .required()
    .matches(
      /^[a-z][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$/,
      'Name must contain only lowercase letters, digits, and hyphens, start with a letter or digit, not end with a hyphen, and not contain double hyphens',
    )
    .test('not-neurons', 'Name cannot be "neurons"', (value) => value !== 'neurons')
    .test('no-double-hyphens', 'Name cannot contain double hyphens', (value) => !value || !value.includes('--')),
  modelId: yup.string().required(),
  description: yup.string().optional().nullable().max(128, 'Description must be less than 128 characters'),
  url: yup.string().optional().nullable().url('Must be a valid URL'),
});

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const requestJson = await request.json();

  let validatedRequest;
  try {
    validatedRequest = await NewSourceSetRequestSchema.validate(requestJson);
  } catch (error) {
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: error.message, path: error.path }, { status: 400 });
    }
    throw error;
  }

  // check the model and ensure user has access
  const model = await getModelById(validatedRequest.modelId);
  if (!model) {
    return NextResponse.json({ error: 'Model not found' }, { status: 400 });
  }

  // check if source set already exists
  const existingSourceSet = await getSourceSet(validatedRequest.modelId, validatedRequest.name);
  if (existingSourceSet) {
    return NextResponse.json({ error: 'Source set name already exists for this model.' }, { status: 400 });
  }

  const sourceSetData = {
    name: validatedRequest.name,
    modelId: validatedRequest.modelId,
    description: validatedRequest.description ?? validatedRequest.name,
    urls: [validatedRequest.url ?? ''],
    visibility: Visibility.UNLISTED,
    type: 'Custom',
    creatorName: request.user.name,
    creatorId: request.user.id,
    hasDashboards: true,
    allowInferenceSearch: false,
    releaseName: null,
    defaultOfModelId: null,
    defaultShowBreaks: true,
    showDfa: false,
    showCorrelated: false,
    showHeadAttribution: false,
    showUmap: false,
    creatorEmail: null,
    defaultRange: 1,
    createdAt: new Date(),
  };

  const set = await createSourceSetWithSources(sourceSetData, model.layers, request.user);

  // get the sourceset with the sources
  const sourcesetWithSources = await getSourceSet(set.modelId, set.name);

  return NextResponse.json(sourcesetWithSources);
});
