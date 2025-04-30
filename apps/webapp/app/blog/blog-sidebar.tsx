import Image from 'next/image';

export default function BlogSidebar() {
  return (
    <div className="flex w-full flex-col items-center justify-center gap-x-5 sm:h-full sm:w-auto sm:flex-col sm:items-center sm:justify-start sm:gap-y-5 sm:pt-6">
      <a href="/blog" className="mb-4 h-32 w-32 sm:mb-0 sm:h-52 sm:min-h-52 sm:w-52 sm:min-w-52">
        <Image
          src="/images/blog/the-residual-stream-logo.jpg"
          alt="The Residual Stream"
          className="h-32 w-32 rounded-full shadow transition-all hover:scale-105 sm:h-52 sm:min-h-52 sm:w-52 sm:min-w-52"
          width={420}
          height={420}
        />
      </a>
      <div className="flex w-full flex-col">
        <div className="mb-3 flex flex-col items-center justify-center border-b pb-3 sm:mb-5 sm:pb-5">
          <a href="/blog">
            <h1 className="whitespace-pre text-sm font-bold hover:underline sm:pb-1.5 sm:text-[18px]">
              The Residual Stream
            </h1>
          </a>
          <h2 className="mt-0 text-xs font-medium text-slate-600 sm:text-[13px]">Neuronpedia&apos;s Blog</h2>{' '}
          <a href="/feed.xml">
            <img
              src="https://img.shields.io/badge/rss-F88900?style=for-the-badge&logo=rss&logoColor=white"
              alt="RSS"
              className="mt-2 rounded"
              width={60}
              height={30}
            />
          </a>
        </div>
        <div className="flex flex-col items-center justify-center pb-3">
          <h1 className="whitespace-pre text-sm font-bold sm:pb-1.5 sm:text-[18px]">The Babble</h1>
          <h2 className="mt-0 text-center text-xs font-medium text-slate-600 sm:text-[13px]">Podcast by NotebookLM</h2>
          <div className="flex flex-row items-center justify-center gap-x-2">
            <a
              href="https://podcasts.apple.com/us/podcast/the-babble-a-residual-stream-podcast/id1809283401"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src="/images/blog/apple-podcasts.svg"
                alt="Apple Podcasts"
                className="mt-2 rounded"
                width={30}
                height={30}
              />
            </a>
            <a href="https://open.spotify.com/show/5n0RlvZTMmW5hDNf0OYYdd" target="_blank" rel="noopener noreferrer">
              <img
                src="/images/blog/spotify-podcasts.png"
                alt="Spotify Podcasts"
                className="mt-2 rounded"
                width={30}
                height={30}
              />
            </a>
            <a href="https://anchor.fm/s/103d41c38/podcast/rss" target="_blank" rel="noopener noreferrer">
              <img
                src="/images/blog/rss-podcasts.png"
                alt="Podcast RSS"
                className="mt-2 rounded"
                width={30}
                height={30}
              />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
