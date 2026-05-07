import { X } from 'lucide-react';
import { Dispatch, ReactNode, SetStateAction, useEffect, useState } from 'react';

export default function Leaflet({
  setShow,
  children,
}: {
  setShow: Dispatch<SetStateAction<boolean>>;
  children: ReactNode;
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
  }, []);

  function handleClose() {
    setVisible(false);
    setTimeout(() => setShow(false), 200);
  }

  return (
    <>
      <div
        className="group fixed inset-x-0 top-0 z-40 flex max-h-[calc(100vh-3rem)] w-screen flex-col overflow-y-auto bg-white transition-transform duration-200 ease-out sm:hidden"
        style={{ transform: visible ? 'translateY(0)' : 'translateY(-100%)' }}
      >
        <div className="sticky top-0 z-50 flex justify-end bg-white px-2 pt-2">
          <button
            type="button"
            aria-label="Close menu"
            onClick={handleClose}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-white text-slate-500 transition-all hover:bg-slate-100 hover:text-slate-700 focus:outline-none"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        {children}
        <div className="rounded-b-4xl -mt-1 flex h-7 w-full shrink-0 items-center justify-center border-b border-slate-200">
          <div className="-mr-1 h-1 w-6 rounded-full bg-slate-300" />
          <div className="h-1 w-6 rounded-full bg-slate-300" />
        </div>
      </div>
      <div
        className="fixed inset-0 z-30 bg-slate-100 bg-opacity-10 backdrop-blur transition-opacity duration-200"
        style={{ opacity: visible ? 1 : 0 }}
        onClick={handleClose}
      />
    </>
  );
}
