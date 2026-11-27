'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { DEMO_MODE, IS_LOCALHOST, NEXT_PUBLIC_ENABLE_SIGNIN, SITE_NAME_VERCEL_DEPLOY } from '@/lib/env';
import { Session } from 'next-auth';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import GithubCorner from 'react-github-corner';
import CustomTooltip from '../custom-tooltip';
import NavBarButtons from './navbar-buttons';
import { useSignInModal } from './sign-in-modal';
import UserDropdown from './user-dropdown';

export default function NavBar({ session }: { session: Session | null }) {
  const { SignInModal } = useSignInModal();
  const { setSignInModalOpen, refreshUser } = useGlobalContext();

  const pathname = usePathname();
  const isNeuralTransparency = pathname.toLowerCase().startsWith('/drift-calibrator');

  /** Which tool of the site this page is, named under the wordmark, or nothing on the site at large. */
  const sectionLabel = pathname.endsWith('/graph') ? 'Circuit Tracer' : null;

  useEffect(() => {
    if (session) {
      refreshUser();
    }
  }, [session]);

  if (pathname.toLowerCase().includes('sae-bench-anonymized')) {
    return (
      <div className="fixed top-0 z-20 flex h-12 w-full select-none items-center justify-center bg-slate-100 px-5 text-[15px] text-slate-600 transition-all">
        <div className="mt-2 font-medium">SAEBench Result Explorer for Anonymous Review</div>
      </div>
    );
  }

  return (
    <>
      <SignInModal />
      <nav
        aria-label="Main navigation"
        className="fixed top-0 z-20 w-full select-none border-b border-slate-200 bg-white pl-[max(0.75rem,env(safe-area-inset-left))] pr-[max(0.75rem,env(safe-area-inset-right))] transition-all sm:pl-[max(1.25rem,env(safe-area-inset-left))] sm:pr-[max(1.25rem,env(safe-area-inset-right))] 2xl:pr-[max(2rem,env(safe-area-inset-right))]"
      >
        <GithubCorner
          href="https://github.com/hijohnnylin/neuronpedia#readme"
          size={36}
          className="hidden 2xl:inline"
          target="_blank"
          rel="noreferrer"
        />
        <div className="relative flex h-12 items-center justify-between gap-x-2">
          <div className="flex min-w-0 items-center">
            <Link href="/#" className="flex min-w-0 items-center text-base">
              {!SITE_NAME_VERCEL_DEPLOY && (
                <div className="mr-1.5 h-6 w-6 shrink-0 sm:h-5 sm:w-5">
                  <Image
                    src="/logo.png"
                    alt="Neuronpedia logo - a computer chip with a rounded viewfinder border around it"
                    width="28"
                    height="28"
                    className=""
                  />
                </div>
              )}
              {SITE_NAME_VERCEL_DEPLOY ? (
                <p className="ml-1.5 truncate font-medium text-[#7B3F00]">{SITE_NAME_VERCEL_DEPLOY}</p>
              ) : (
                // The wordmark and, under it, which tool of the site this is. A column rather than
                // the label living inside the wordmark, because the mark itself is a row here: on
                // this one page it is Neuronpedia and MIT together, and the label belongs under
                // both of them rather than under the first half.
                <div className="flex flex-col items-start justify-start gap-y-[1px] pl-0.5">
                  <span className="flex flex-row items-center font-normal text-sky-800">
                    Neuronpedia
                    {isNeuralTransparency && (
                      <>
                        <span className="ml-1 mr-1 text-[13px] font-bold text-slate-400/70">×</span>
                        <Image
                          src="/usedby/mit-media-lab.png"
                          alt="MIT Media Lab logo"
                          width="597"
                          height="335"
                          className="-my-1 h-[28px] w-auto"
                        />
                      </>
                    )}
                  </span>
                  {sectionLabel && (
                    <span className="hidden whitespace-nowrap text-[9px] font-semibold uppercase leading-none tracking-wide text-slate-500 sm:inline">
                      {sectionLabel}
                    </span>
                  )}
                </div>
              )}
              {DEMO_MODE ? (
                <CustomTooltip
                  trigger={
                    <div className="ml-1.5 hidden h-5 shrink-0 items-center justify-center rounded bg-emerald-600 px-2 text-[9px] font-bold leading-none text-white md:flex">
                      DEMO MODE
                    </div>
                  }
                >
                  <div>This is a mode that demonstrates hosting Neuronpedia publicly. It has read-only access.</div>
                </CustomTooltip>
              ) : IS_LOCALHOST ? (
                <CustomTooltip
                  trigger={
                    <div className="ml-1.5 hidden h-5 shrink-0 items-center justify-center rounded bg-emerald-600 px-2 text-[9px] font-bold leading-none text-white md:flex">
                      LOCALHOST
                    </div>
                  }
                >
                  <div>Running Locally</div>
                </CustomTooltip>
              ) : (
                ''
              )}
            </Link>
          </div>

          <div className="flex shrink-0 items-center gap-x-2">
            <div className="hidden items-center justify-end gap-x-0.5 font-light text-slate-500 md:flex">
              <NavBarButtons placement="bar" />
            </div>

            {!session && NEXT_PUBLIC_ENABLE_SIGNIN && (
              <button
                type="button"
                className="flex h-11 shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-slate-200 bg-white px-3 text-xs text-slate-600 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 2xl:h-8"
                onClick={() => setSignInModalOpen(true)}
              >
                Sign In
              </button>
            )}
            <UserDropdown session={session} />
          </div>
        </div>
      </nav>
    </>
  );
}
