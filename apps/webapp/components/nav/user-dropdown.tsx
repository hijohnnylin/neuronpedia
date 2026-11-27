'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import Popover from '@/components/sign-in-modal/popover';
import { useRouter } from '@bprogress/next';
import { Bookmark, ExternalLink, LayoutList, LogOut, Menu, MoveUpLeft, Settings, User } from 'lucide-react';
import { Session } from 'next-auth';
import { signOut } from 'next-auth/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import NavBarButtons from './navbar-buttons';

export default function UserDropdown({ session }: { session: Session | null }) {
  const router = useRouter();
  const pathname = usePathname();
  const { name } = session?.user || {};
  const { user, showUserPopover, setShowUserPopover } = useGlobalContext();

  useEffect(() => {
    setShowUserPopover(false);
  }, [pathname, setShowUserPopover]);

  return (
    <div className={`relative z-50 items-center text-left ${session ? 'flex' : 'flex 2xl:hidden'}`}>
      <Popover
        content={
          <div className="w-full rounded-md bg-white p-2 text-slate-500 md:max-h-[var(--radix-popover-content-available-height)] md:w-[min(20rem,calc(100vw-1rem))] md:overflow-y-auto md:overscroll-contain 2xl:w-44">
            {user && (
              <>
                <Link
                  className="relative z-50 mb-1 flex w-full select-none flex-col items-center justify-center space-x-2 rounded-xl border-0 border-slate-200 bg-slate-100 p-2 py-3 text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-200 sm:mb-0.5"
                  href={`/user/${user?.name || name}`}
                  onClick={() => {
                    setShowUserPopover(false);
                  }}
                >
                  <div className="py-1.5 text-base leading-none text-slate-700 sm:py-0 sm:text-[13px]">
                    {user?.name || name}
                  </div>
                  <div className="mt-1.5 flex flex-row items-center gap-x-1 text-xs font-bold uppercase leading-none text-sky-700 sm:text-[11px]">
                    <div className="-mb-[2px] leading-none">My Profile</div>
                    <ExternalLink className="h-3.5 w-3.5" />
                  </div>
                </Link>
              </>
            )}
            <div
              className={`mx-auto flex w-full flex-col items-center justify-center gap-3 border-slate-200 pb-4 ${
                user && 'mb-4 mt-4 border-b border-t pt-4'
              } 2xl:hidden`}
              onClick={(event) => {
                if (event.target instanceof Element && event.target.closest('a[href]')) {
                  setShowUserPopover(false);
                }
              }}
            >
              <NavBarButtons placement="menu" />
            </div>
            {user && (
              <>
                <button
                  type="button"
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                  onClick={(e) => {
                    e.preventDefault();
                    setShowUserPopover(false);
                    router.push('/account');
                  }}
                >
                  <Settings className="h-4 w-4" />
                  <p className="text-xs sm:text-sm">Settings + API</p>
                </button>

                <Link
                  prefetch={false}
                  href={`/user/${user?.name || name}/lists`}
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                >
                  <LayoutList className="h-4 w-4" />
                  <p className="text-xs sm:text-sm">My Lists</p>
                </Link>
                <Link
                  prefetch={false}
                  href={`/user/${user?.name || name}/vectors`}
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                >
                  <MoveUpLeft className="h-4 w-4" />
                  <p className="text-xs sm:text-sm">My Vectors</p>
                </Link>
                <button
                  type="button"
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                  onClick={(e) => {
                    e.preventDefault();
                    setShowUserPopover(false);
                    router.push(`/user/${user?.name || name || ''}`);
                  }}
                >
                  <Bookmark className="h-4 w-4" />
                  <p className="text-xs sm:text-sm">Bookmarks</p>
                </button>

                <button
                  type="button"
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                  onClick={() => {
                    setShowUserPopover(false);
                    signOut();
                  }}
                >
                  <LogOut className="h-4 w-4" />
                  <p className="text-xs sm:text-sm">Logout</p>
                </button>
              </>
            )}
          </div>
        }
        align="end"
        openPopover={showUserPopover}
        setOpenPopover={setShowUserPopover}
      >
        <button
          type="button"
          aria-expanded={showUserPopover}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-600 transition-colors hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 2xl:h-8 2xl:w-8 2xl:rounded-full"
          onClick={() => setShowUserPopover(!showUserPopover)}
        >
          <Menu aria-hidden className="h-5 w-5 2xl:hidden" />
          <span className="sr-only 2xl:hidden">{session ? 'Navigation and account menu' : 'Navigation menu'}</span>
          {session && (
            <>
              <User aria-hidden className="hidden h-4 w-4 2xl:block" />
              <span className="hidden 2xl:sr-only 2xl:block">Account menu</span>
            </>
          )}
        </button>
      </Popover>
    </div>
  );
}
