'use client';

import { Button } from '@/components/shadcn/button';
import emailSpellChecker from '@zootools/email-spell-checker';
import { MailIcon } from 'lucide-react';
import { signIn } from 'next-auth/react';
import Link from 'next/link';
import { useState } from 'react';
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

  async function handleSubmit() {
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
    const result = await signIn('email', {
      email: finalEmail,
      name: generateFromEmail(finalEmail),
      redirect: false,
    });
    if (result?.error) {
      setError('Something went wrong. Please try again.');
      setSubmitting(false);
    } else {
      setSubmitted(true);
    }
  }

  return (
    <div className="flex w-full flex-1 flex-col items-stretch gap-y-2.5 rounded-xl bg-white px-5 py-3.5 shadow">
      {latestPost && (
        <Link
          href={`/blog/${latestPost.slug}`}
          className="group flex flex-col rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 transition-all hover:border-sky-300 hover:bg-sky-50"
        >
          <div className="mb-0.5 flex flex-row items-center justify-start gap-x-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-sky-700">Latest Update</div>
            <div className="text-[10px] font-medium text-slate-400">
              {(() => {
                // Remove ordinal suffixes ("st", "nd", "rd", "th") from the date string for reliable parsing
                const cleanDateStr = latestPost.dateString.replace(/(\d{1,2})(st|nd|rd|th)/g, '$1');
                const d = new Date(cleanDateStr);
                if (isNaN(d.getTime())) return '';
                return d.toLocaleString('default', { month: 'long', year: 'numeric' });
              })()}
            </div>
          </div>
          <div className="text-[12.5px] font-semibold leading-snug text-slate-800 group-hover:text-sky-700">
            {latestPost.title}
          </div>
          <div className="mt-[3px] text-[11px] font-medium leading-snug text-slate-400">{latestPost.description}</div>
        </Link>
      )}
      <div className="flex w-full flex-col items-center gap-y-2 px-1 sm:flex-row sm:gap-x-4 sm:gap-y-0">
        <div className="flex flex-col items-center text-center sm:items-start sm:gap-y-0 sm:text-left">
          <div className="mb-0 flex flex-row items-center justify-center text-[13px] font-semibold text-slate-700">
            <MailIcon className="mr-1.5 h-4 w-4" /> Get Updates
          </div>
          <div className="text-xs text-slate-500">
            <a
              href="/privacy"
              target="_blank"
              rel="noreferrer noopener"
              className="text-[11px] text-sky-700 hover:underline"
            >
              No spam
            </a>
            , unsubscribe anytime.
          </div>
        </div>
        {submitted ? (
          <div className="flex flex-1 items-center justify-center text-sm font-medium text-emerald-600">
            Check your email for a confirmation link.
          </div>
        ) : (
          <div className="flex w-full flex-1 flex-col gap-y-1">
            <div className="flex flex-row items-center gap-x-2">
              <input
                type="email"
                placeholder="your-email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onKeyUp={(e) => {
                  if (e.key === 'Enter') handleSubmit();
                }}
                className="h-8.5 flex-1 rounded-md border border-slate-300 px-3 text-xs focus:border-sky-600 focus:outline-none focus:ring-1 focus:ring-sky-600 sm:text-[13px]"
              />
              <Button
                disabled={submitting}
                onClick={() => handleSubmit()}
                className="gap-x-1.5 bg-sky-600 text-white hover:bg-sky-700"
                size="sm"
              >
                <span>Submit</span>
              </Button>
            </div>
            {error && <div className="text-xs text-red-500">{error}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
