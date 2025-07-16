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
      <div className="fixed top-0 z-20 w-full select-none border-b border-slate-200 bg-white px-5 transition-all sm:pr-8">
        <GithubCorner
          href="https://github.com/hijohnnylin/neuronpedia#readme"
          size={36}
          className="hidden sm:inline"
          target="_blank"
          rel="noreferrer"
        />
        <div className="relative flex h-12 items-center justify-between sm:mx-auto sm:h-12">
          <div className="flex flex-row gap-8 sm:mt-0">
            <Link href="/#" className="flex items-center justify-center text-base sm:text-[16px]">
              {!SITE_NAME_VERCEL_DEPLOY && (
                <div className="mr-1.5 h-6 w-6 sm:h-5 sm:w-5">
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
                <p className="ml-1.5 font-medium text-[#7B3F00] sm:mt-0">{SITE_NAME_VERCEL_DEPLOY}</p>
              ) : (
                <p className="font-normal text-sky-800 sm:mt-0">
                  Neuronpedia
                  {pathname.endsWith('/graph') ? <span className="hidden sm:inline"> - Circuit Tracer</span> : ''}
                </p>
              )}
              {DEMO_MODE ? (
                <CustomTooltip
                  trigger={
                    <div className="ml-1.5 flex h-5 items-center justify-center rounded bg-emerald-600 px-2 text-[9px] font-bold leading-none text-white">
                      DEMO MODE
                    </div>
                  }
                >
                  <div>This is a mode that demonstrates hosting Neuronpedia publicly. It has read-only access.</div>
                </CustomTooltip>
              ) : IS_LOCALHOST ? (
                <CustomTooltip
                  trigger={
                    <div className="ml-1.5 flex h-5 items-center justify-center rounded bg-emerald-600 px-2 text-[9px] font-bold leading-none text-white">
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

          <div className="flex-inline flex items-center">
            <div className="hidden select-none items-center justify-end gap-0 font-light text-slate-500 sm:flex">
              <NavBarButtons session={session} />
            </div>

            {session ? (
              <UserDropdown session={session} />
            ) : NEXT_PUBLIC_ENABLE_SIGNIN ? (
              <>
                <button
                  type="button"
                  className="flex h-9 w-full items-center justify-center rounded-md border border-slate-200 bg-white px-3 text-sm transition-all duration-75 hover:bg-slate-50 focus:outline-none sm:ml-2.5 sm:h-8"
                  onClick={() => {
                    setSignInModalOpen(true);
                  }}
                >
                  <p className="w-full whitespace-nowrap text-center text-xs font-normal text-slate-600">Sign In</p>
                </button>
                <UserDropdown session={session} />
              </>
            ) : (
              ''
            )}
          </div>
        </div>
      </div>
    </>
  );
}
