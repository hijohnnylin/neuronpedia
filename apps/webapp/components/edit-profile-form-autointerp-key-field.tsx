import { UserSecret, UserSecretType } from '@prisma/client';
import { useState } from 'react';

const NO_API_KEY_SET_VALUE = 'NO_API_KEY_SET';

export function EditProfileFormAutointerpKeyField({
  keyLabel,
  keyType,
  keyMessage,
}: {
  keyLabel: string;
  keyType: UserSecretType;
  keyMessage: string;
}) {
  const [showKey, setShowKey] = useState(false);
  const [key, setKey] = useState('');

  return (
    <label htmlFor="key" className="mb-2.5 block text-sm leading-none">
      <span className="block text-xs leading-none text-slate-600">{keyLabel}</span>
      <div className="mt-1.5 flex flex-row items-center justify-center gap-x-1.5">
        <input
          type="text"
          id="key"
          name="key"
          disabled
          className="disabled form-input block w-full rounded-md border-2 border-slate-300 bg-slate-200 text-xs text-slate-500 focus:outline-none focus:ring-0"
          value={
            showKey
              ? key === ''
                ? 'Loading...'
                : key === NO_API_KEY_SET_VALUE
                  ? "No API key set. Click 'Edit' to set one."
                  : key
              : "Click 'Show' to Reveal Key"
          }
        />
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            setShowKey(!showKey);
            fetch(`/api/user/api-key`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                type: keyType,
              }),
            })
              .then((response) => {
                if (response.status === 404) {
                  return null;
                }
                return response.json();
              })
              .then((result: UserSecret | null) => {
                if (result) {
                  setKey(result.value);
                } else {
                  setKey("No API key set. Click 'Edit' to set one.");
                }
              });
          }}
          className="h-9 min-w-12 rounded-md bg-slate-200 px-2 py-2 text-xs font-medium text-slate-500 hover:bg-slate-300 active:bg-sky-300"
        >
          {showKey ? 'Hide' : 'Show'}
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            alert(keyMessage);
            const newKey = prompt(
              "Your key will only be used for your requests and not for any other user's requests.\n\nIt is your responsibility to monitor your key's usage and to disable it when necessary.\n\nIf you agree with these terms, you can save your OpenRouter API key to Neuronpedia by pasting it below.\n\nIf you wish to remove your key, set it to blank.",
            );
            if (newKey || newKey === '') {
              fetch(`/api/user/api-key-update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  type: keyType,
                  value: newKey,
                }),
              })
                .then((response) => {
                  if (response.status !== 200) {
                    alert("Error setting API key. Please check that it's valid.");
                  } else {
                    if (newKey === '') {
                      alert('API key removed successfully.');
                    } else {
                      alert('API key set successfully. You may now generate autointerp and scoring autointerp.');
                    }
                    setKey(newKey);
                  }
                })
                .catch(() => {
                  alert("Error setting API key. Please check that it's valid and that it has access to models.");
                });
            }
          }}
          className="h-9 min-w-12 rounded-md bg-slate-200 px-1.5 py-2 text-xs font-medium text-slate-500 hover:bg-slate-300 active:bg-sky-300"
        >
          Edit
        </button>
      </div>
    </label>
  );
}
