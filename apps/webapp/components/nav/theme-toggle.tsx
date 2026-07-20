'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

export default function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <button
      type="button"
      aria-label="Toggle dark mode"
      onClick={() => {
        setTheme(resolvedTheme === 'dark' ? 'light' : 'dark');
      }}
      className="flex h-6 w-6 cursor-pointer items-center justify-center rounded-full text-slate-500 transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-sky-400"
    >
      {mounted && resolvedTheme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
    </button>
  );
}
