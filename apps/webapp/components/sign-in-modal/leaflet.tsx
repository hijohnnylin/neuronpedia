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
        className="group fixed inset-x-0 bottom-0 z-40 w-screen bg-white pb-5 transition-transform duration-200 ease-out sm:hidden"
        style={{ transform: visible ? 'translateY(0)' : 'translateY(100%)' }}
      >
        <div className="rounded-t-4xl -mb-1 flex h-7 w-full items-center justify-center border-t border-slate-200">
          <div className="-mr-1 h-1 w-6 rounded-full bg-slate-300" />
          <div className="h-1 w-6 rounded-full bg-slate-300" />
        </div>
        {children}
      </div>
      <div
        className="fixed inset-0 z-30 bg-slate-100 bg-opacity-10 backdrop-blur transition-opacity duration-200"
        style={{ opacity: visible ? 1 : 0 }}
        onClick={handleClose}
      />
    </>
  );
}
