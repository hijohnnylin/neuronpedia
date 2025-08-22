import { setUserEmailUnsubscribeAll } from '@/lib/db/user';
import { env } from '@/lib/env';

export default async function Page({ searchParams }: { searchParams?: { [key: string]: string | undefined } }) {
  const unsubscribeCode = searchParams?.code;
  const unsubscribed = await setUserEmailUnsubscribeAll(unsubscribeCode || '');

  return unsubscribed ? (
    <div className="mt-16 w-full max-w-screen-md text-center text-slate-600">
      <h2 className="mb-3 font-bold">Unsubscribed Successfully</h2>
      <p>
        You have been unsubscribed from all emails.
        <br />
        To subscribe again, sign in and go to Settings.
      </p>
    </div>
  ) : (
    <div className="mt-16 w-full max-w-screen-md text-center text-slate-600">
      <h2 className="mb-3 font-bold">Invalid Unsubscribe Link</h2>
      <p>
        Sorry, that unsubscribe link seems to be invalid. Please either email us at{' '}
        <a href={`mailto:${env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS}`} className="text-sky-600">
          {env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS}
        </a>
        , or log in and unsubscribe via Settings.
      </p>
    </div>
  );
}
