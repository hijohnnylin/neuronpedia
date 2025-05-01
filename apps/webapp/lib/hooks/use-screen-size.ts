// TODO: this might be the same as use-window-size

import { useEffect, useState } from 'react';

export const useScreenSize = () => {
  const windowIsUndefined = typeof window === 'undefined';

  const [screenSize, setScreenSize] = useState({
    width: windowIsUndefined ? 0 : window.innerWidth,
    height: windowIsUndefined ? 0 : window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setScreenSize({
        width: windowIsUndefined ? 0 : window.innerWidth,
        height: windowIsUndefined ? 0 : window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);

    // Clean up the event listener when the component unmounts
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return screenSize;
};
