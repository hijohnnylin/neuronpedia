import { env } from '@/lib/env';

export default function NotFound() {
  return (
    <div className="text-center">
      <h2 className="mb-4 mt-20 flex max-w-screen-xl flex-col items-center justify-center px-5 text-lg font-light sm:text-lg">
        <div>
          Couldn&lsquo;t find that page.
          <div className="mb-2 mt-2 text-sm font-medium text-slate-600">
            If you are trying to access a specific model, source/SAE, or feature, check available public models/sources
            (including which ones have inference enabled) at{' '}
            <a
              href={`https://${env.NEXT_PUBLIC_URL}/available-resources`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-700 hover:underline"
            >
              {env.NEXT_PUBLIC_URL}/available-resources
            </a>
          </div>
        </div>
        <div className="flex flex-row gap-x-1">
          Please{' '}
          <a
            href="/contact"
            className="flex cursor-pointer items-center whitespace-nowrap px-0 py-0.5 text-sky-700 transition-all hover:underline sm:px-0 sm:py-0"
          >
            report this
          </a>{' '}
          if you think it&#39;s a bug.
        </div>
      </h2>
      <div>
        <a
          href="/"
          className="mx-5 inline-flex items-center justify-center rounded-full border border-transparent bg-sky-700 px-8 py-3 text-base font-medium leading-6 text-white shadow-md hover:bg-sky-600 focus:outline-none"
        >
          Return Home
        </a>
      </div>
    </div>
  );
}
