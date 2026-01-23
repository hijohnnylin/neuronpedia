'use client';

import Link from 'next/link';

export default function NotFoundFeature() {
  const pathSegments = window.location.pathname.split('/').filter(Boolean);
  const modelId = pathSegments[0] || '';
  const layer = pathSegments[1] || '';
  const index = pathSegments[2] || '';
  return (
    <div className="text-center">
      <h2 className="mb-4 mt-20 flex max-w-screen-xl flex-col items-center justify-center px-5 text-lg font-light sm:text-lg">
        <div className="font-medium">
          Latent {index} Not Found
          <div className="mb-2 mt-4 text-sm font-medium text-slate-600">
            Model: {modelId}
            <br />
            Source: {layer}
            <br />
            <br />
            Double check that the width is large enough to contain this latent.
          </div>
        </div>
        <div className="mt-4 flex flex-row gap-x-1 text-base">
          Please{' '}
          <Link
            href="/contact"
            className="flex cursor-pointer items-center whitespace-nowrap px-0 py-0.5 text-base text-sky-700 transition-all hover:underline sm:px-0 sm:py-0"
          >
            report this
          </Link>{' '}
          if you think it&#39;s a bug.
        </div>
      </h2>
      <div>
        <a
          href={`/${modelId}`}
          className="mx-5 inline-flex items-center justify-center rounded-full border border-transparent bg-sky-700 px-8 py-3 text-base font-medium leading-6 text-white shadow-md hover:bg-sky-600 focus:outline-none"
        >
          Return to {modelId}
        </a>
      </div>
    </div>
  );
}
