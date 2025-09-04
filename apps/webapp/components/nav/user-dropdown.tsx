'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import Popover from '@/components/sign-in-modal/popover';
import { Bookmark, ExternalLink, LayoutList, LogOut, Menu, MoveUpLeft, Settings, User } from 'lucide-react';
import { Session } from 'next-auth';
import { signOut } from 'next-auth/react';
import { useRouter } from 'next-nprogress-bar';
import Link from 'next/link';
import NavBarButtons from './navbar-buttons';

export default function UserDropdown({ session }: { session: Session | null }) {
  const router = useRouter();
  const { name } = session?.user || {};
  const { user, showUserPopover, setShowUserPopover } = useGlobalContext();

  return (
    <div className="relative z-50 flex items-center text-left">
      <Popover
        content={
          <div className="z-50 w-full rounded-md bg-white p-2 text-slate-500 sm:w-44">
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
                  <p className="mt-1.5 flex flex-row items-center gap-x-1 text-xs font-bold uppercase leading-none text-sky-700 sm:text-[11px]">
                    <div className="-mb-[2px] leading-none">My Profile</div>
                    <ExternalLink className="h-3.5 w-3.5" />
                  </p>
                </Link>

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
                  <p className="text-base sm:text-sm">Settings + API</p>
                </button>

                <Link
                  prefetch={false}
                  href={`/user/${user?.name || name}/lists`}
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                >
                  <LayoutList className="h-4 w-4" />
                  <p className="text-base sm:text-sm">My Lists</p>
                </Link>
                <Link
                  prefetch={false}
                  href={`/user/${user?.name || name}/vectors`}
                  className="relative flex w-full items-center justify-center space-x-2 rounded-md p-2 text-left text-sm outline-none ring-0 transition-all duration-75 hover:bg-slate-100 sm:justify-start"
                >
                  <MoveUpLeft className="h-4 w-4" />
                  <p className="text-base sm:text-sm">My Vectors</p>
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
                  <p className="text-base sm:text-sm">Bookmarks</p>
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
                  <p className="text-base sm:text-sm">Logout</p>
                </button>
              </>
            )}
            {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions */}
            <div
              className={`mx-auto flex w-full flex-col items-center justify-center gap-3 border-slate-200 pb-4 ${
                user && 'mt-4 border-t pt-4'
              } sm:hidden`}
              onClick={() => {
                setShowUserPopover(false);
              }}
            >
              <NavBarButtons />
            </div>
          </div>
        }
        align="end"
        openPopover={showUserPopover}
        setOpenPopover={setShowUserPopover}
      >
        {session ? (
          <button
            type="button"
            className={`flex cursor-pointer select-none flex-row items-center justify-center gap-x-1 rounded-md border border-slate-200 px-2 py-2 text-slate-500 transition-all focus:outline-none sm:ml-2.5 sm:rounded-full sm:px-2 sm:py-2 ${
              showUserPopover ? 'shadow-0 bg-white hover:bg-white' : 'bg-white hover:bg-slate-50 hover:shadow'
            }`}
            onClick={() => setShowUserPopover(!showUserPopover)}
          >
            <div className="flex flex-col items-center justify-center gap-y-0.5 text-xs font-medium leading-none text-slate-500">
              <User className="h-4 w-4" />
            </div>
          </button>
        ) : (
          <div className="ml-2 block sm:hidden">
            <button
              type="button"
              className={`flex h-10 cursor-pointer select-none items-center justify-center rounded-md border border-slate-200 px-2.5 py-1 text-slate-500 transition-all focus:outline-none sm:rounded-lg sm:px-4 sm:py-2 ${
                showUserPopover
                  ? 'shadow-0 bg-white hover:bg-white'
                  : 'bg-white shadow hover:bg-slate-50 hover:shadow sm:shadow-md'
              }`}
              onClick={() => setShowUserPopover(!showUserPopover)}
            >
              <Menu />
            </button>
          </div>
        )}
      </Popover>
    </div>
  );
}
