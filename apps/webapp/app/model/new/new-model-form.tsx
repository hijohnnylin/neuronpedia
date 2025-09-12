'use client';

// TODO: not actually used (and api prevents public use)

import { env } from '@/lib/env';
import { ModelPartialSchema } from 'prisma/generated/zod';
import { useState } from 'react';

export default function NewModelForm() {
  const [submitting, setSubmitting] = useState(false);
  const [modelId, setModelId] = useState('');
  const [modelDisplayName, setModelDisplayName] = useState('');

  const handleSubmit = async (event: React.SyntheticEvent<HTMLFormElement>) => {
    setSubmitting(true);
    event.preventDefault();
    const form = event.currentTarget;
    const formElements = form.elements as typeof form.elements & {
      id: { value: string };
      displayName: { value: string };
      owner: { value: string };
      layers: { value: number };
    };

    const regexp = /^[a-zA-Z0-9-_]+$/; // alphanumeric, dash, underscore
    if (formElements.id.value.search(regexp) === -1) {
      alert('Model ID must only contain numbers, letters, and underscores.');
      setSubmitting(false);
      return;
    }
    if (modelId.trim().length === 0 || modelDisplayName.trim().length === 0) {
      alert('Model ID and Display Name are required.');
      setSubmitting(false);
      return;
    }

    const modelData = {
      id: formElements.id.value,
      displayName: modelDisplayName,
      owner: formElements.owner.value,
      layers: Number(formElements.layers.value),
    };
    const body = ModelPartialSchema.parse(modelData);

    const response = await fetch(`/api/local/new-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const model = await response.json();

    window.location.href = `/${model.id}`;
  };

  return (
    <div className="w-full max-w-xl px-5 xl:px-0">
      <h1 className="mt-5 text-center text-xl font-medium text-slate-800">Add Model</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="id" className="mt-5 block">
          <div className="text-slate-700">Model ID (Lowercased Letters, Numbers, Hyphens, and Underscores)</div>
          <div className="font-mono text-xs font-bold text-slate-400">
            {env.NEXT_PUBLIC_URL}/{modelId || '[Model-ID]'}
          </div>
          <input
            type="text"
            id="id"
            value={modelId}
            onChange={(e) => {
              setModelId(e.target.value);
            }}
            name="id"
            required
            className="form-input mt-1 block w-full"
            placeholder="myneatmodel-4"
          />
        </label>
        <label htmlFor="displayName" className="mt-5 block">
          <div className="text-slate-700">Model Display Name (All Casing Allowed)</div>
          <div className="font-mono text-xs font-bold text-slate-400">
            For display on the user interfaces, with casing.
          </div>
          <input
            type="text"
            id="displayName"
            value={modelDisplayName}
            onChange={(e) => {
              setModelDisplayName(e.target.value);
            }}
            name="displayName"
            required
            className="form-input mt-1 block w-full"
            placeholder="MyNeatModel-4"
          />
        </label>
        <label htmlFor="owner" className="mt-5 block">
          <div className="text-slate-700">Owner or Creator</div>
          <div className="font-mono text-xs font-bold text-slate-400">
            For display on the user interfaces, with casing.
          </div>
          <input
            type="text"
            id="owner"
            name="owner"
            required
            className="form-input mt-1 block w-full"
            placeholder="Jane Appleseed"
          />
        </label>
        <label htmlFor="layers" className="mt-4 block">
          <span className="text-slate-700">Layers</span>
          <input
            type="number"
            id="layers"
            name="layers"
            required
            className="form-input mt-1 block w-full"
            placeholder="48"
          />
        </label>
        {modelId.length > 0 && (
          <div className="pt-5">
            Your model will be created at:
            <br /> {`${env.NEXT_PUBLIC_URL}/${modelId}`}
          </div>
        )}
        <div className="pt-5 text-center">
          <button
            disabled={submitting}
            type="submit"
            className="h-full rounded bg-sky-200 px-4 py-3 text-sm font-semibold uppercase text-sky-700 hover:bg-sky-300 focus:ring-sky-600 disabled:bg-slate-300 disabled:text-slate-400 disabled:opacity-50"
          >
            {submitting ? 'Submitting...' : 'Submit'}
          </button>
        </div>
      </form>
    </div>
  );
}
