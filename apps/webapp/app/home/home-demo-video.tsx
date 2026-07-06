'use client';

import { Pause, Play } from 'lucide-react';
import { useRef, useState } from 'react';

export default function HomeDemoVideo({ src }: { src: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(true);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
    } else {
      video.pause();
    }
  };

  return (
    <div className="group relative h-full w-full cursor-pointer" onClick={togglePlay}>
      <video
        ref={videoRef}
        src={src}
        autoPlay
        loop
        muted
        playsInline
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        className="h-full w-full overflow-hidden rounded-3xl bg-slate-50 p-1"
      />
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-0 transition-opacity duration-200 group-hover:opacity-100">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-black/45 text-white shadow-md backdrop-blur-sm">
          {isPlaying ? <Pause className="h-6 w-6 fill-current" /> : <Play className="ml-0.5 h-6 w-6 fill-current" />}
        </div>
      </div>
    </div>
  );
}
