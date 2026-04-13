export default async function Page(props: { searchParams?: Promise<{ [key: string]: string | undefined }> }) {
  const searchParams = await props.searchParams;
  const query = searchParams;

  return (
    <div className="flex w-full max-w-md flex-col justify-center px-1 pt-20">
      <form action="/api/auth/callback/email" method="get" className="flex flex-col text-center">
        <label
          htmlFor="layer"
          className="mb-3 flex flex-col items-center justify-center gap-x-2 text-sm font-medium text-slate-500"
        >
          <input
            type="hidden"
            name="token"
            value={query?.token}
            required
            className="form-input mt-1 block w-24 rounded-md border-slate-200 text-center font-mono text-base text-slate-700 focus:border-slate-200"
          />
          Click the button below to complete signing in.
        </label>
        <input type="hidden" name="callbackUrl" value={query?.callbackUrl} />
        <input type="hidden" name="email" value={query?.email} />
        <button
          type="submit"
          className="mx-auto mt-2.5 rounded-lg bg-sky-600 px-6 py-3 text-lg text-white shadow-lg transition-all hover:scale-105 hover:bg-sky-500"
        >
          Continue
        </button>
      </form>
      <p className="mt-2 flex flex-row items-center justify-center gap-1 text-center text-sm text-slate-600" />
    </div>
  );
}
