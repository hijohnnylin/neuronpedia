'use client';

import { Button } from '@/components/shadcn/button';
import { useTurnstile } from '@/components/turnstile-widget';
import { signInErrorMessage } from '@/lib/utils/sign-in-email';
import emailSpellChecker from '@zootools/email-spell-checker';
import { MailIcon } from 'lucide-react';
import { signIn } from 'next-auth/react';
import Link from 'next/link';
import { useRef, useState } from 'react';
import { generateFromEmail } from 'unique-username-generator';
import isEmail from 'validator/lib/isEmail';

type LatestPost = {
  title: string;
  description: string;
  date: string;
  slug: string;
  dateString: string;
};

export default function HomeNewsletterSignup({ latestPost }: { latestPost?: LatestPost }) {
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState('');
  const [turnstileShown, setTurnstileShown] = useState(false);
  const emailInputRef = useRef<HTMLInputElement>(null);
  const turnstile = useTurnstile(emailInputRef);
  const submitDisabled = submitting || !turnstile.ready;

  async function handleSubmit() {
    if (submitDisabled) {
      return;
    }
    setSubmitting(true);
    setError('');
    if (!isEmail(email)) {
      setError('Invalid email. Please try again.');
      setSubmitting(false);
      return;
    }
    let finalEmail = email;
    const suggestedEmail = emailSpellChecker.run({ email });
    if (
      suggestedEmail &&
      window.confirm(
        `You typed "${email}", which seems like a typo.\nDid you mean "${suggestedEmail.full}"?\nClick OK to use the corrected email.`,
      )
    ) {
      finalEmail = suggestedEmail.full;
    }
    const result = await signIn(
      'email',
      {
        email: finalEmail,
        name: generateFromEmail(finalEmail),
        redirect: false,
      },
      turnstile.signInParams,
    );
    turnstile.reset();
    if (!result || result.error) {
      setError(signInErrorMessage(result?.error));
      setSubmitting(false);
    } else {
      setSubmitted(true);
    }
  }

  return (
    <div className="flex w-full flex-1 flex-row items-stretch gap-x-0 overflow-hidden px-0 py-0">
      {/* put back after new blog post */}
      {latestPost && (
        <Link
          href={`/blog/${latestPost.slug}`}
          className="group flex flex-1 flex-col items-center justify-center bg-sky-100 px-3 py-2 transition-all hover:border-sky-300 hover:bg-sky-200 sm:py-3"
        >
          <div className="mb-1 flex flex-row items-center justify-start gap-x-2 sm:mb-1">
            {/* <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Latest Post</div> */}
            <div className="text-[8px] font-bold uppercase tracking-wide text-sky-700/70 group-hover:text-slate-500">
              Latest Post -{' '}
              {(() => {
                // Remove ordinal suffixes ("st", "nd", "rd", "th") from the date string for reliable parsing
                const cleanDateStr = latestPost.dateString.replace(/(\d{1,2})(st|nd|rd|th)/g, '$1');
                const d = new Date(cleanDateStr);
                if (isNaN(d.getTime())) return '';
                return d.toLocaleString('default', { month: 'long', year: 'numeric' });
              })()}{' '}
            </div>
          </div>
          <div className="text-center text-[12px] font-semibold leading-snug text-sky-700 group-hover:text-sky-800">
            {latestPost.title}
          </div>
          <div className="mt-[3px] hidden text-center text-[10.5px] font-medium leading-snug text-slate-500 group-hover:text-sky-600 sm:block">
            {latestPost.description}
          </div>
        </Link>
      )}
      <div className="flex w-full max-w-[220px] flex-1 flex-col items-center gap-y-2 px-0 sm:flex-row sm:items-stretch sm:gap-x-4 sm:gap-y-0">
        {submitted ? (
          <div className="flex flex-1 items-center justify-center text-sm font-medium text-emerald-600">
            Check your email for a confirmation link.
          </div>
        ) : (
          <div className="flex h-full w-full flex-1 flex-col gap-y-1">
            <div className="relative flex h-full flex-1 flex-col items-center justify-center gap-y-0">
              <div className="absolute left-0 top-1.5 w-full">
                <div className="flex h-full flex-1 flex-col items-center justify-center gap-y-0">
                  <div className="text-[8px] font-bold uppercase leading-none text-slate-300">Newsletter</div>
                </div>
              </div>
              <input
                ref={emailInputRef}
                type="email"
                placeholder="name@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onFocus={() => setTurnstileShown(true)}
                onKeyUp={(e) => {
                  if (e.key === 'Enter') handleSubmit();
                }}
                className="mx-0 w-full flex-1 rounded-none rounded-b-none border border-b-0 border-none border-slate-200 bg-slate-50 pt-5 text-center text-[12.5px] leading-none outline-none ring-0 placeholder:text-slate-400 focus:border-sky-600 focus:bg-sky-50 focus:outline-none focus:ring-0 focus:ring-sky-600"
              />
              <Button
                disabled={submitDisabled}
                onClick={() => handleSubmit()}
                className="mx-0 h-8 min-h-8 w-full shrink-0 gap-x-1.5 rounded-none border-none border-slate-300 bg-slate-200 px-4 text-[10px] font-semibold uppercase text-slate-600 shadow-none hover:border-sky-600 hover:bg-sky-300 hover:text-sky-800"
                size="sm"
              >
                <MailIcon className="h-3.5 w-3.5 text-slate-600" />
                <span>Get Updates</span>
              </Button>
            </div>
            {(turnstileShown || email !== '') && turnstile.widget}
            {error && <div className="text-xs text-red-500">{error}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
