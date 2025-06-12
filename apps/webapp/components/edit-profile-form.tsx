'use client';

import { UserSecret, UserSecretType } from '@prisma/client';
import * as Switch from '@radix-ui/react-switch';
import copy from 'copy-to-clipboard';
import { CopyIcon } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';
import { User } from 'prisma/generated/zod';
import { useEffect, useState } from 'react';
import { EditProfileFormAutointerpKeyField } from './edit-profile-form-autointerp-key-field';
import { useGlobalContext } from './provider/global-provider';
import { Button } from './shadcn/button';
// eslint-ignore-next-line
export const dynamic = 'force-dynamic';

export default function EditProfileForm({
  defaultUsername,
  defaultNewsletterNotifyEmail,
  defaultUnsubscribeAllEmail,
}: {
  defaultUsername?: string | null;
  defaultNewsletterNotifyEmail?: boolean;
  defaultUnsubscribeAllEmail?: boolean;
}) {
  const session = useSession();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [name, setName] = useState(defaultUsername || session.data?.user.name);
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [newsletterNotifyEmail, setNewsletterNotifyEmail] = useState<boolean>(
    defaultNewsletterNotifyEmail !== undefined ? defaultNewsletterNotifyEmail : true,
  );
  const [unsubscribeAllEmail, setUnsubscribeAllEmail] = useState<boolean>(
    defaultUnsubscribeAllEmail !== undefined ? defaultUnsubscribeAllEmail : false,
  );
  const { showToastMessage, user, setUser } = useGlobalContext();

  useEffect(() => {
    if (user) {
      setName(user.name);
      setNewsletterNotifyEmail(user.emailNewsletterNotification);
      setUnsubscribeAllEmail(user.emailUnsubscribeAll);
    }
  }, [user]);

  if (!session) {
    return redirect('/');
  }
  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitting(true);
    fetch(`/api/user/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        newsletterNotifyEmail,
        unsubscribeAllEmail,
      }),
    })
      .then((response) => response.json())
      .then((result: User) => {
        if (result) {
          setUser(result);
        }
        showToastMessage(<span className="font-semibold text-slate-500">Profile Successfully Updated</span>);
        session.update({ name });
        setIsSubmitting(false);
        window.location.href = '/';
      })
      .catch(() => {
        showToastMessage(
          <span className="font-semibold text-slate-500">
            Error Updating Profile. Check that the username is unique.
          </span>,
        );
        setIsSubmitting(false);
      });
  };
  return (
    <form onSubmit={submit}>
      <label htmlFor="name" className="block text-base leading-none">
        <span className="block text-sm leading-none text-slate-600">Username</span>
        <div className="pt-2 text-[11px] font-medium text-slate-400">Public. Must be letters, numbers, and dashes.</div>
        <input
          type="text"
          id="name"
          name="name"
          min="1"
          max="39"
          required
          className="form-input mt-1 block w-full rounded-md border-2 border-slate-300 text-sm focus:border-sky-700 focus:outline-none focus:ring-0"
          value={name}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
            }
          }}
          onChange={(e) => {
            e.preventDefault();
            setName(e.target.value);
          }}
        />
      </label>

      <div className="mt-2 flex flex-col">
        <label htmlFor="apiKey" className="mt-3 block text-base leading-none">
          <span className="block text-sm leading-none text-slate-600">Neuronpedia API Key</span>
          <div className="pt-2 text-[11px] font-medium text-slate-400">Private. For accessing Neuronpedia by API.</div>
          <div className="mt-1 flex flex-row items-center justify-center gap-x-1.5">
            <input
              type="text"
              id="apiKey"
              name="apiKey"
              disabled
              className="disabled form-input block w-full rounded-md border-2 border-slate-300 bg-slate-200 text-xs text-slate-500 focus:outline-none focus:ring-0"
              value={showApiKey ? (apiKey === '' ? 'Loading...' : apiKey) : "Click 'Show' to Reveal Key"}
            />
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                setShowApiKey(!showApiKey);
                fetch(`/api/user/api-key`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    type: UserSecretType.NEURONPEDIA,
                  }),
                })
                  .then((response) => response.json())
                  .then((result: UserSecret) => {
                    setApiKey(result.value);
                  });
              }}
              className="h-9 min-w-12 rounded-md bg-slate-200 px-1.5 py-2 text-xs font-medium text-slate-500 hover:bg-slate-300 active:bg-sky-300"
            >
              {showApiKey ? 'Hide' : 'Show'}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                if (apiKey === '') {
                  fetch(`/api/user/api-key`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      type: UserSecretType.NEURONPEDIA,
                    }),
                  })
                    .then((response) => response.json())
                    .then((result: UserSecret) => {
                      setApiKey(result.value);
                      copy(result.value);
                      alert('Copied to clipboard.');
                    });
                }
                if (apiKey !== '') {
                  copy(apiKey);
                  alert('Copied to clipboard.');
                }
              }}
            >
              <CopyIcon className="h-9 min-w-12 rounded-md bg-slate-200 px-2.5 py-2.5 text-slate-500 hover:bg-slate-300 active:bg-sky-300" />
            </button>
          </div>
        </label>
      </div>

      <div className="mb-2 mt-6 flex flex-col text-left font-semibold text-slate-600">Email Settings</div>
      <label htmlFor="newsletterNotifyEmail" className="mt-0 flex flex-row items-center justify-between pr-2 text-sm">
        <span className="block leading-none text-slate-600">Quarterly Newsletter</span>
        <Switch.Root
          className="relative h-[25px] w-[42px] cursor-default rounded-full bg-slate-200 outline-none focus:shadow-black data-[state=checked]:bg-emerald-600"
          id="newsletterNotifyEmail"
          checked={newsletterNotifyEmail}
          onCheckedChange={(val) => {
            setNewsletterNotifyEmail(val);
          }}
        >
          <Switch.Thumb className="block h-[21px] w-[21px] translate-x-0.5 rounded-full bg-white shadow-sm transition-transform duration-100 will-change-transform data-[state=checked]:translate-x-[19px]" />
        </Switch.Root>
      </label>

      <label htmlFor="unsubscribeAll" className="mt-1 flex flex-row items-center justify-between pr-2 text-sm">
        <span className="block leading-none text-slate-600">Unsubscribe All</span>
        <Switch.Root
          className="relative h-[25px] w-[42px] cursor-default rounded-full bg-slate-200 outline-none focus:shadow-black data-[state=checked]:bg-emerald-600"
          id="unsubscribeAll"
          checked={unsubscribeAllEmail}
          onCheckedChange={(val) => {
            setUnsubscribeAllEmail(val);
          }}
        >
          <Switch.Thumb className="block h-[21px] w-[21px] translate-x-0.5 rounded-full bg-white shadow-sm transition-transform duration-100 will-change-transform data-[state=checked]:translate-x-[19px]" />
        </Switch.Root>
      </label>

      <div className="flex flex-col">
        <div className="mb-3 mt-6 flex flex-col text-left font-semibold text-slate-600">
          Auto-Interp Keys
          <div className="text-[11px] font-normal text-slate-400">
            Used to run your auto-interp explanation and scoring requests.
          </div>
        </div>
        {user?.canTriggerExplanations && (
          <div className="mb-5 text-xs font-semibold text-sky-600">
            Your account has been granted permission to do auto-interp explanations and scoring for free. You do not
            need to be set the following API keys.
          </div>
        )}

        <EditProfileFormAutointerpKeyField
          keyLabel="OpenRouter Key (Required for Some Auto-Interp Types)"
          keyType={UserSecretType.OPENROUTER}
          keyMessage={`By adding your OpenRouter API key, you agree that Neuronpedia will store it and use it when you perform OpenRouter calls for auto-interp or other OpenRouter-necessary features.\n\nYou agree that requests to OpenRouter on your behalf will incur charges to your OpenRouter account. A test, one-token 'hi' call will be made with your key to ensure it's valid, which will incur charges of a fraction of a cent.`}
        />

        <EditProfileFormAutointerpKeyField
          keyLabel="OpenAI Key"
          keyType={UserSecretType.OPENAI}
          keyMessage={`By adding your OpenAI API key, you agree that Neuronpedia will store it and use it when you perform OpenAI calls for auto-interp or other OpenAI-necessary features.\n\nYou agree that requests to OpenAI on your behalf will incur charges to your OpenAI account. A test, one-token 'hi' call will be made with your key to ensure it's valid, which will incur charges of a fraction of a cent.\n\nEnsure that your key has access to the 'gpt-4o-mini' model.`}
        />

        <EditProfileFormAutointerpKeyField
          keyLabel="Anthropic Key"
          keyType={UserSecretType.ANTHROPIC}
          keyMessage={`By adding your Anthropic API key, you agree that Neuronpedia will store it and use it when you perform Anthropic calls for auto-interp or other Anthropic-necessary features.\n\nYou agree that requests to Anthropic on your behalf will incur charges to your Anthropic account. A test, one-token 'hi' call will be made with your key to ensure it's valid, which will incur charges of a fraction of a cent.`}
        />

        <EditProfileFormAutointerpKeyField
          keyLabel="Google Gemini Key"
          keyType={UserSecretType.GOOGLE}
          keyMessage={`By adding your Google Gemini API key, you agree that Neuronpedia will store it and use it when you perform Google Gemini calls for auto-interp or other Google Gemini-necessary features.\n\nYou agree that requests to Google Gemini on your behalf will incur charges to your Google Gemini account. A test, one-token 'hi' call will be made with your key to ensure it's valid, which will incur charges of a fraction of a cent.`}
        />
      </div>

      <div className="mb-0 pt-6 text-center">
        <Button type="submit" className="w-full px-5" disabled={isSubmitting}>
          Save
        </Button>
      </div>
    </form>
  );
}
