'use client';

// this isn't officially supported yet

import ModelSelector from '@/components/feature-selector/model-selector';
import { env } from '@/lib/env';
import { SourceSet } from '@prisma/client';
import { useState } from 'react';

export default function NewSourceSetForm() {
  const [submitting, setSubmitting] = useState(false);
  const [modelId, setModelId] = useState('');
  const [saeId, setSaeId] = useState('');

  const handleSubmit = async (event: React.SyntheticEvent<HTMLFormElement>) => {
    setSubmitting(true);
    event.preventDefault();
    const form = event.currentTarget;
    const formElements = form.elements as typeof form.elements & {
      description: { value: string };
      saeId: { value: string };
      type: { value: string };
      owner: { value: string };
    };

    if (formElements.saeId.value.split('-').length < 2) {
      alert('SAE ID must have at least 1 dash. E.g, res-jb');
      setSubmitting(false);
      return;
    }
    const saeData = {
      modelId,
      description: formElements.description.value,
      saeId,
      type: formElements.type.value,
      owner: formElements.owner.value,
    };

    const response = await fetch(`/api/local/new-sae-set`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(saeData),
    });
    const sae = (await response.json()) as SourceSet;

    window.location.href = `/${sae.modelId}/${sae.name}`;
  };

  return env.IS_LOCALHOST ? (
    <div className="w-full max-w-xl px-5 xl:px-0">
      <h1 className="mt-0 text-center text-xl font-medium text-slate-800">Add SAE Set</h1>
      <div className="mt-2 text-sm">
        An SAE Set is one or more SAEs. Usually, this is to group together SAEs with the same hook point, but different
        layers.
        <br />
        For example, the{' '}
        <a href="https://neuronpedia.org/gpt2-small/res-jb" target="_blank" rel="noreferrer">
          <code className="font-bold text-sky-700 underline">RES-JB</code>
        </a>{' '}
        (link to production Neuronpedia) SAE Set has 12 SAEs for 12 layers of GPT2-Small, all for the residual stream.
      </div>
      <div className="text-sm">
        This form creates the SAEs entries in the database, so that you can upload features to it. It does not upload
        weights or run the model.
      </div>
      <form onSubmit={handleSubmit}>
        {/* eslint-disable-next-line */}
        <label htmlFor="id" className="mt-5 block">
          <ModelSelector
            modelId={modelId}
            modelIdChangedCallback={(newModelId) => {
              setModelId(newModelId);
            }}
          />
        </label>
        <label htmlFor="saeId" className="mt-5 block">
          <span className="text-slate-700">
            SAE Set ID (letters/numbers/underscores with one dash. no spaces.)
            <br />
            <span className="text-xs">[abbreviated hook name and identifier]-[abbreviated author/org]</span>
            <br />
            <span className="text-xs">Example: res-ja would mean residuals (res) created by Jane Appleseed (ja)</span>
          </span>
          <input
            type="text"
            id="saeId"
            name="saeId"
            value={saeId}
            onChange={(e) => {
              setSaeId(e.target.value);
            }}
            required
            className="form-input mt-1 block w-full"
            placeholder="res-ja"
          />
        </label>
        <label htmlFor="description" className="mt-5 block">
          <span className="text-slate-700">
            Display Name
            <br />
            <span className="text-xs">
              Displayed on the SAE Set page. Example: SAEs for all Residual Stream Layers of GPT2-Small
            </span>
          </span>
          <input
            type="text"
            id="description"
            name="description"
            required
            className="form-input mt-1 block w-full"
            placeholder="Residuals for MyCoolModel-5"
          />
        </label>

        <label htmlFor="type" className="mt-5 block">
          <span className="text-slate-700">
            Display Name for Hook / Grouping
            <br />
            <span className="text-xs">
              What is the hook or grouping that all these SAEs have in common? Keep this short.
              <br />
              Example: Attention Out
            </span>
          </span>
          <input
            type="text"
            id="type"
            name="type"
            required
            className="form-input mt-1 block w-full"
            placeholder="Attention Out"
          />
        </label>

        <label htmlFor="owner" className="mt-5 block">
          <span className="text-slate-700">
            Display Name for Creator / Owner
            <br />
            <span className="text-xs">Keep it short.</span>
          </span>
          <input
            type="text"
            id="owner"
            name="owner"
            required
            className="form-input mt-1 block w-full"
            placeholder="Jane Appleseed"
          />
        </label>

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
  ) : (
    <>This does not work outside of localhost.</>
  );
}
