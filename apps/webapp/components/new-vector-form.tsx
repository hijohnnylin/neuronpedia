import { VECTOR_VALID_HOOK_TYPE_TO_DETAILS } from '@/lib/utils/neuron-vector';
import { STEER_STRENGTH_MAX, STEER_STRENGTH_MIN } from '@/lib/utils/steer';
import { Model } from '@prisma/client';
import * as Select from '@radix-ui/react-select';
import { Field, Form, Formik } from 'formik';
import { ChevronDown } from 'lucide-react';
import { useEffect, useState } from 'react';
import ModelSelector from './feature-selector/model-selector';
import { useGlobalContext } from './provider/global-provider';
import { Button } from './shadcn/button';

export default function NewVectorForm({ defaultModel, callback }: { defaultModel?: Model; callback: () => void }) {
  const [model, setModel] = useState<Model | undefined>(defaultModel);
  const { globalModels } = useGlobalContext();
  const { getFirstInferenceEnabledModel } = useGlobalContext();

  useEffect(() => {
    const modelId = model?.id || getFirstInferenceEnabledModel();
    if (!model && modelId) {
      setModel(globalModels[modelId]);
    }
  }, []);

  return (
    <Formik
      initialValues={{
        modelId: '',
        layerNumber: '',
        hookType: Object.keys(VECTOR_VALID_HOOK_TYPE_TO_DETAILS)[0],
        vector: '',
        vectorLabel: '',
        vectorDefaultSteerStrength: '20',
      }}
      validate={(values) => {
        const errors: { [key: string]: string } = {};

        // Layer number validation
        if (!values.layerNumber) {
          errors.layerNumber = 'Required';
        } else {
          const num = parseInt(values.layerNumber, 10);
          if (Number.isNaN(num) || !Number.isInteger(num) || num < 0) {
            errors.layerNumber = 'Must be a positive integer';
          } else if (model && num >= (model.layers || 0)) {
            errors.layerNumber = `${model.id} only has ${model.layers} layers`;
          }
        }

        // Vector validation
        if (!values.vector || values.vector.length === 0) {
          errors.vector = 'Required';
        } else {
          try {
            const vectorArray = JSON.parse(values.vector as string);
            if (!Array.isArray(vectorArray)) {
              errors.vector = 'Must be a JSON array';
            } else if (!vectorArray.every((n) => typeof n === 'number')) {
              errors.vector = 'All elements must be numbers';
            } else if (model && vectorArray.length !== model.dimension) {
              errors.vector = `Vector must have ${model.dimension} elements`;
            }
          } catch (e) {
            errors.vector = 'Invalid JSON number array.';
          }
        }

        // Vector label validation
        if (!values.vectorLabel || values.vectorLabel.trim() === '') {
          errors.vectorLabel = 'Required';
        }

        if (!values.vectorDefaultSteerStrength) {
          errors.vectorDefaultSteerStrength = 'Required';
        }
        if (Number.isNaN(parseInt(values.vectorDefaultSteerStrength, 10))) {
          errors.vectorDefaultSteerStrength = 'Must be a number';
        }
        if (
          parseInt(values.vectorDefaultSteerStrength, 10) < STEER_STRENGTH_MIN ||
          parseInt(values.vectorDefaultSteerStrength, 10) > STEER_STRENGTH_MAX
        ) {
          errors.vectorDefaultSteerStrength = `Must be between ${STEER_STRENGTH_MIN} and ${STEER_STRENGTH_MAX}`;
        }
        return errors;
      }}
      onSubmit={async (values, { setSubmitting }) => {
        try {
          values.modelId = model?.id || '';
          const vectorArray = JSON.parse(values.vector as string);

          const response = await fetch('/api/vector/new', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              ...values,
              layerNumber: parseInt(values.layerNumber, 10),
              vector: vectorArray,
            }),
          });

          if (!response.ok) {
            throw new Error('Failed to create vector');
          }

          await response.json();
          callback();
        } catch (error) {
          console.log(JSON.stringify(error));
          alert('Error creating vector');
        }
        setSubmitting(false);
      }}
    >
      {({ isSubmitting, errors, touched }) => (
        <Form className="flex flex-col gap-y-2.5">
          <div className="mt-1 flex flex-col">
            <label htmlFor="vectorLabel" className="text-[10px] font-medium uppercase text-slate-400">
              Vector Label (Keep It Short)
              <Field
                id="vectorLabel"
                name="vectorLabel"
                placeholder="the color red"
                className={`mt-1 h-[32px] w-full rounded-md border ${
                  errors.vectorLabel && touched.vectorLabel ? 'border-red-500' : 'border-slate-300'
                } px-2 py-1 text-xs placeholder-slate-300`}
              />
            </label>
            {errors.vectorLabel && touched.vectorLabel && (
              <div className="mt-1 text-[10px] text-red-500">{errors.vectorLabel}</div>
            )}
          </div>
          <div className="flex w-full flex-row gap-x-2">
            <div className="flex flex-col">
              <label htmlFor="modelId" className="flex flex-col text-[10px] font-medium uppercase text-slate-400">
                Model
                {defaultModel ? (
                  <Field
                    id="modelId"
                    name="modelId"
                    readOnly
                    value={model?.id}
                    className="mt-1 h-[40px] w-28 max-w-28 select-none rounded-md border border-slate-300 px-2 text-center text-[11px] text-slate-600"
                  />
                ) : (
                  <div className="mt-1 h-[40px] w-36 min-w-36 select-none text-center">
                    <ModelSelector
                      modelId={model?.id || getFirstInferenceEnabledModel() || ''}
                      modelIdChangedCallback={(newModelId) => {
                        setModel(globalModels[newModelId]);
                      }}
                    />
                  </div>
                )}
              </label>
            </div>
            <div className="flex w-14 max-w-14 flex-col">
              <label htmlFor="layerNumber" className="text-[10px] font-medium uppercase text-slate-400">
                Layer
                <Field
                  id="layerNumber"
                  name="layerNumber"
                  type="number"
                  placeholder="1"
                  className={`mt-1 h-[40px] w-full rounded-md border ${
                    errors.layerNumber && touched.layerNumber ? 'border-red-500' : 'border-slate-300'
                  } px-2 text-center text-sm text-slate-600 placeholder-slate-300`}
                />
              </label>
              {errors.layerNumber && touched.layerNumber && (
                <div className="mt-1 text-[10px] text-red-500">{errors.layerNumber}</div>
              )}
            </div>

            <div className="flex flex-1 flex-col items-start justify-start">
              <label htmlFor="hookType" className="text-[10px] font-medium uppercase text-slate-400">
                Hook Type
                <Select.Root name="hookType" defaultValue={Object.keys(VECTOR_VALID_HOOK_TYPE_TO_DETAILS)[0]}>
                  <Select.Trigger className="mt-1 flex h-[40px] w-full flex-row items-center justify-between rounded-md border border-slate-300 px-3 text-[11.5px] font-medium text-slate-600 focus:outline-none focus:ring-0">
                    <Select.Value placeholder="Select a Hook" />
                    <ChevronDown className="h-3 w-3" />
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content>
                      <Select.Viewport className="rounded-md border border-slate-200 bg-white p-2 shadow">
                        {Object.entries(VECTOR_VALID_HOOK_TYPE_TO_DETAILS).map(([hookType, { displayName }]) => (
                          <Select.Item
                            key={hookType}
                            value={hookType}
                            className="cursor-pointer rounded px-2 py-1 text-[12px] font-medium text-slate-600 hover:bg-slate-100"
                          >
                            <Select.ItemText>{displayName}</Select.ItemText>
                          </Select.Item>
                        ))}
                      </Select.Viewport>
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              </label>
            </div>
          </div>

          <div className="flex w-full flex-row gap-x-2">
            <div className="flex flex-1 flex-col">
              <label htmlFor="vector" className="text-[10px] font-medium uppercase text-slate-400">
                Paste Vector (JSON Array Format)
                <Field
                  as="textarea"
                  name="vector"
                  className={`mt-1 h-[40px] w-full rounded-md border ${
                    errors.vector && touched.vector ? 'border-red-500' : 'border-slate-300'
                  } resize-none px-2 py-2 text-[11px] leading-tight text-slate-600 placeholder-slate-300`}
                  placeholder="[ 0.0065, 0.0184, 0.0206, -0.0247, ... ]"
                />
              </label>
              {errors.vector && touched.vector && (
                <div className="mt-0.5 text-[10px] text-red-500">{errors.vector}</div>
              )}
            </div>
            <div className="flex w-24 max-w-24 flex-col">
              <label htmlFor="vectorDefaultSteerStrength" className="text-[10px] font-medium uppercase text-slate-400">
                Steer Strength
                <Field
                  id="vectorDefaultSteerStrength"
                  name="vectorDefaultSteerStrength"
                  type="number"
                  placeholder="10"
                  className={`mt-1 h-[40px] w-full rounded-md border ${
                    errors.vectorDefaultSteerStrength && touched.vectorDefaultSteerStrength
                      ? 'border-red-500'
                      : 'border-slate-300'
                  } px-2 text-center text-sm text-slate-600 placeholder-slate-300`}
                />
              </label>
              {errors.vectorDefaultSteerStrength && touched.vectorDefaultSteerStrength && (
                <div className="mt-1 text-[10px] text-red-500">{errors.vectorDefaultSteerStrength}</div>
              )}
            </div>
          </div>

          <Button type="submit" disabled={isSubmitting} className="mt-1 text-xs">
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </Button>
        </Form>
      )}
    </Formik>
  );
}
