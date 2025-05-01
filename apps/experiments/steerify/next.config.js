/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://www.neuronpedia.org/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;