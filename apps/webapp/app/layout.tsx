import Navbar from '@/components/nav/navbar';
import AuthProvider from '@/components/provider/auth-provider';
import { Providers } from '@/components/provider/providers';
import Toast from '@/components/toast';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/auth/utils';
import {
  getExplanationModels,
  getExplanationScoreModelTypes,
  getExplanationScoreTypes,
  getExplanationTypes,
} from '@/lib/db/explanation-type';
import { getGlobalSourceReleases } from '@/lib/db/source';
import { env } from '@/lib/env';
import { formatToGlobalModels } from '@/lib/utils/general';
import { Analytics } from '@vercel/analytics/react';
import { Metadata } from 'next';
import { getServerSession } from 'next-auth/next';
import { headers } from 'next/headers';
import { Suspense } from 'react';
import FeatureModal from '../components/feature-modal';
import { getGlobalModels } from '../lib/db/model';
import { authOptions } from './api/auth/[...nextauth]/authOptions';
import { inter } from './fonts';
import Footer from './footer';
import './globals.css';

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export async function generateMetadata(): Promise<Metadata> {
  const description = 'Open Source Interpretability Platform';
  return {
    title: {
      template: '%s ｜ Neuronpedia',
      default: 'Neuronpedia',
    },
    metadataBase: new URL(env.NEXT_PUBLIC_URL),
    description,
    openGraph: {
      title: {
        template: '%s',
        default: 'Neuronpedia',
      },
      url: env.NEXT_PUBLIC_URL,
      siteName: 'Neuronpedia',
      locale: 'en_US',
      type: 'website',
    },
    manifest: '/site.webmanifest',
    robots: {
      index: true,
      follow: true,
      nocache: false,
      googleBot: {
        index: true,
        follow: true,
      },
    },
    icons: {
      icon: [{ url: '/favicon-32x32.png' }, new URL('/favicon-32x32.png', 'https://neuronpedia.org')],
      apple: [{ url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' }],
    },
  };
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await getServerSession(authOptions);
  const user = await makeAuthedUserFromSessionOrReturnNull(session);
  const [models, explanationTypes, explanationModels, releases, explanationScoreTypes, explanationScoreModelTypes] =
    await Promise.all([
      getGlobalModels(user || undefined),
      getExplanationTypes(),
      getExplanationModels(),
      getGlobalSourceReleases(user || undefined),
      getExplanationScoreTypes(),
      getExplanationScoreModelTypes(),
    ]);

  const initialModels = formatToGlobalModels(models);
  const headersList = headers();
  const isEmbed = headersList.get('x-is-embed') === 'true';

  return (
    <html lang="en">
      <body
        className={`flex h-screen flex-col bg-slate-50 bg-fixed ${inter.className} ${
          isEmbed ? 'overscroll-auto' : 'overscroll-none'
        }`}
      >
        <AuthProvider session={session}>
          <Providers
            initialModels={initialModels}
            initialExplanationTypes={explanationTypes}
            initialExplanationModels={explanationModels}
            initialReleases={releases}
            initialExplanationScoreTypes={explanationScoreTypes}
            initialExplanationScoreModelTypes={explanationScoreModelTypes}
          >
            <Toast />
            {!isEmbed && (
              <Suspense fallback="">
                <Navbar session={session} />
              </Suspense>
            )}
            <FeatureModal />
            <main className={`flex w-full flex-1 flex-col items-center gap-0 ${isEmbed ? 'pt-0' : 'pt-12 sm:pt-12'}`}>
              {children}
            </main>
            {!isEmbed && <Footer />}
            {env.ENABLE_VERCEL_ANALYTICS && <Analytics />}
          </Providers>
        </AuthProvider>
      </body>
    </html>
  );
}
