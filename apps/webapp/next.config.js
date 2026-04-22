/** @type {import('next').NextConfig} */

const withMDX = require('@next/mdx')();

const nextConfig = {
  outputFileTracingRoot: __dirname,
  outputFileTracingIncludes: {
    '/api-doc': ['./app/api/**/*.{ts,tsx}'],
  },
  pageExtensions: ['js', 'jsx', 'mdx', 'ts', 'tsx'],
  output: 'standalone', // for docker
  reactStrictMode: false,
  compress: true,
  compiler: {
    // Remove console logs only in production
    removeConsole: process.env.NODE_ENV === 'production',
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'avatars.githubusercontent.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'www.gravatar.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'akka5peax3ifzm8o.public.blob.vercel-storage.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'neuronpedia.s3.us-east-1.amazonaws.com',
        port: '',
        pathname: '/**',
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/assistant-axis',
        destination: '/llama3.3-70b-it/assistant-axis',
      },
    ];
  },
  async redirects() {
    return [
      {
        source: '/sae-evals',
        destination: '/sae-bench',
        permanent: true,
      },
      {
        source: '/saebench',
        destination: '/sae-bench',
        permanent: true,
      },
      {
        source: '/temporal-saes',
        destination: '/temporal-feature-analysis',
        permanent: true,
      },
      {
        source: '/circuit-tracer',
        destination: '/graph',
        permanent: true,
      },
      {
        source: '/problems',
        destination: '/explorer',
        permanent: true,
      },
      {
        source: '/problems/:id',
        destination: '/explorer/:id',
        permanent: true,
      },
    ];
  },
  // https://github.com/mjmlio/mjml/issues/2562
  // https://github.com/vercel/next.js/issues/50042
  // runpod-sdk needs to be external because it requires its own package.json at runtime
  serverExternalPackages: ['mjml', 'runpod-sdk'],
  // https://github.com/aws-amplify/amplify-js/issues/11030
  webpack: (config, { webpack, isServer, nextRuntime }) => {
    // https://github.com/handlebars-lang/handlebars.js/issues/953
    config.resolve.alias.handlebars = 'handlebars/dist/handlebars.js';
    // https://github.com/i18next/next-i18next/issues/1545
    config.module = {
      ...config.module,
      exprContextCritical: false,
    };
    // Avoid AWS SDK Node.js require issue
    if (isServer && nextRuntime === 'nodejs')
      config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /^aws-crt$/ }));

    // // https://github.com/APIDevTools/json-schema-ref-parser?tab=readme-ov-file#browser-support
    // config.resolve.fallback = {
    //   path: require.resolve('path-browserify'),
    //   fs: require.resolve('browserify-fs'),
    // };
    // config.plugins.push(
    //   new webpack.ProvidePlugin({
    //     Buffer: ['buffer', 'Buffer'],
    //   }),
    // );

    return config;
  },
};

module.exports = withMDX(nextConfig);

// const withBundleAnalyzer = require("@next/bundle-analyzer")({
//   enabled: process.env.ANALYZE === "true",
// });

// module.exports = withBundleAnalyzer(nextConfig);

// Injected content via Sentry wizard below

const { withSentryConfig } = require('@sentry/nextjs');

module.exports = withSentryConfig(module.exports, {
  // For all available options, see:
  // https://github.com/getsentry/sentry-webpack-plugin#options

  org: 'johnny-66',
  project: 'neuronpedia',

  // Only print logs for uploading source maps in CI
  silent: !process.env.CI,

  // For all available options, see:
  // https://docs.sentry.io/platforms/javascript/guides/nextjs/manual-setup/

  // Upload a larger set of source maps for prettier stack traces (increases build time)
  widenClientFileUpload: true,

  // Hides source maps from generated client bundles
  hideSourceMaps: true,
  telemetry: false,

  sourcemaps: {
    deleteSourcemapsAfterUpload: true,
  },

  webpack: {
    // Automatically annotate React components to show their full name in breadcrumbs and session replay
    reactComponentAnnotation: {
      enabled: true,
    },

    // Automatically tree-shake Sentry logger statements to reduce bundle size
    treeshake: {
      removeDebugLogging: true,
    },

    // Enables automatic instrumentation of Vercel Cron Monitors.
    automaticVercelMonitors: true,
  },
});
