'use client';

import { ThemeProvider as NextThemesProvider } from 'next-themes';

export const ThemeProvider: React.FC<React.ComponentProps<typeof NextThemesProvider>> = (props) => (
  <NextThemesProvider {...props} />
);
