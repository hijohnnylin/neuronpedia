import Link from 'next/link';
import { ReactNode } from 'react';

// An icon link with a small label below it, the same height and style as the Prev/Next buttons.
export default function SelectorLinkButton({
  href,
  title,
  label,
  icon,
  className = '',
}: {
  href: string;
  title: string;
  label: string;
  icon: ReactNode;
  className?: string;
}) {
  return (
    <Link
      href={href}
      title={title}
      aria-label={title}
      className={`group flex h-10 min-h-[40px] select-none flex-col items-center justify-center rounded bg-slate-200 px-2 text-slate-500 hover:bg-sky-700 hover:text-white ${className}`}
    >
      {icon}
      <div className="mt-0.5 text-center text-[8px] font-medium uppercase leading-none text-slate-400 group-hover:text-white">
        {label}
      </div>
    </Link>
  );
}
