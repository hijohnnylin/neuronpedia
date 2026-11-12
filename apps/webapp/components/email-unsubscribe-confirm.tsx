'use client';

import { Button } from '@/components/shadcn/button';
import type { EmailUnsubscribeList } from '@/lib/db/user';
import { useState } from 'react';

const LIST_DESCRIPTIONS: Record<EmailUnsubscribeList, string> = {
  newsletter: 'the Neuronsletter',
  all: 'all emails',
};

function InvalidLink({ contactEmail }: { contactEmail: string }) {
  return (
    <>
      <h2 className="mb-3 font-bold">Invalid Unsubscribe Link</h2>
      <p>
        Sorry, that unsubscribe link seems to be invalid. Please either email us at{' '}
        <a href={`mailto:${contactEmail}`} className="text-sky-600">
          {contactEmail}
        </a>
        , or log in and unsubscribe via Settings.
      </p>
    </>
  );
}

// The user must click to unsubscribe. Link scanners open every link in an email, so an
// unsubscribe on page load removes people who did not ask for it.
export default function EmailUnsubscribeConfirm({
  code,
  list,
  isValidCode,
  contactEmail,
}: {
  code: string;
  list: EmailUnsubscribeList;
  isValidCode: boolean;
  contactEmail: string;
}) {
  const [state, setState] = useState<'idle' | 'loading' | 'done' | 'error'>(isValidCode ? 'idle' : 'error');

  const unsubscribe = async () => {
    setState('loading');
    const params = new URLSearchParams({ code, list });
    const response = await fetch(`/api/email/unsubscribe?${params.toString()}`, { method: 'POST' }).catch(() => null);
    setState(response?.ok ? 'done' : 'error');
  };

  return (
    <div className="mt-16 w-full max-w-screen-md text-center text-slate-600">
      {state === 'error' && <InvalidLink contactEmail={contactEmail} />}
      {state === 'done' && (
        <>
          <h2 className="mb-3 font-bold">Unsubscribed Successfully</h2>
          <p>
            You have been unsubscribed from {LIST_DESCRIPTIONS[list]}.
            <br />
            To subscribe again, sign in and go to Settings.
          </p>
        </>
      )}
      {(state === 'idle' || state === 'loading') && (
        <>
          <h2 className="mb-3 font-bold">Unsubscribe</h2>
          <p className="mb-5">Click the button below to unsubscribe from {LIST_DESCRIPTIONS[list]}.</p>
          <Button onClick={unsubscribe} disabled={state === 'loading'}>
            {state === 'loading' ? 'Unsubscribing...' : 'Unsubscribe'}
          </Button>
        </>
      )}
    </div>
  );
}
