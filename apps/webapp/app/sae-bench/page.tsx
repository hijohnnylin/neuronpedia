import { prisma } from '@/lib/db';
import { EvalTypeWithPartialRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import EvalsTable from './evals-table';

export async function generateMetadata(): Promise<Metadata> {
  const title = `SAE Bench - Evals`;
  return {
    title,
    description: '',
    openGraph: {
      title,
      description: '',
      url: `/sae-bench`,
    },
  };
}

export default async function Page(props: { searchParams: Promise<{ embed: string }> }) {
  const searchParams = await props.searchParams;
  // only get the fields that we need - this can get huge
  let evalTypes: {
    id: number;
    name: string;
    featured: boolean;
    description: string | null;
    evals: {
      id: number;
      typeName: string;
      modelId: string;
      sourceId: string;
      output: Record<string, any>;
    }[];
  }[] = await prisma.$queryRaw`
    SELECT 
      et.*,
      json_agg(
        json_build_object(
          'typeName', e."typeName",
          'modelId', e."modelId",
          'sourceId', e."sourceId",
          'output', (
            e."output" - 'eval_config' - 'sae_bench_commit_hash' - 'eval_result_unstructured' - 'eval_id' - 'datetime_epoch_millis' - 'sae_lens_version' 
            || jsonb_build_object(
              'sae_cfg_dict', 
              jsonb_build_object(
                'training_tokens', (e."output"->'sae_cfg_dict'->>'training_tokens')::integer,
                'architecture', (e."output"->'sae_cfg_dict'->>'architecture'),
                'activation_fn_str', (e."output"->'sae_cfg_dict'->>'activation_fn_str'),
                'd_sae', (e."output"->'sae_cfg_dict'->>'d_sae')::integer
              )
            )
          )
        )
      ) as evals
    FROM "EvalType" et
    LEFT JOIN "Eval" e ON et."name" = e."typeName"
    WHERE et.featured = true
    GROUP BY et.name
  `;

  //  reorder evalTypes so that type="core" is first
  evalTypes = evalTypes.sort((a, b) => {
    if (a.name === 'core') {
      return -1;
    }
    if (b.name === 'core') {
      return 1;
    }
    if (a.name === 'scr' || a.name === 'tpp') {
      return 1;
    }
    if (b.name === 'scr' || b.name === 'tpp') {
      return -1;
    }
    return a.name.localeCompare(b.name);
  });

  const isEmbed = searchParams.embed === 'true';

  return (
    <div className="flex h-full w-full flex-col items-center">
      {!isEmbed && (
        <div className="sticky top-12 z-20 flex w-full flex-row items-center justify-between border-b border-slate-200 bg-white px-4 pb-2 sm:px-6 sm:pb-3 sm:pt-3">
          <div className="flex w-full flex-col gap-x-3 gap-y-1 leading-none sm:flex-row sm:items-center sm:justify-between">
            <div className="mt-0 flex-col gap-y-1 text-center font-sans text-[12px] font-bold leading-snug text-slate-700 sm:flex sm:text-left sm:text-[18px] sm:leading-none">
              SAEBench: A Comprehensive Benchmark for Sparse Autoencoders
              <div className="mt-0.5 text-xs font-medium text-slate-600 sm:text-sm">
                Adam Karvonen · Can Rager · March 2025
              </div>
            </div>
            <div className="flex flex-row items-center justify-center gap-x-2 text-center font-sans text-[10px] font-medium leading-none text-slate-500 sm:justify-end sm:text-left sm:text-[14px]">
              <a
                href="/sae-bench/info"
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-row items-center whitespace-pre rounded-full bg-emerald-600 px-3 py-2 text-[10px] font-bold text-white hover:bg-emerald-700 sm:ml-1 sm:flex sm:px-4 sm:py-2.5 sm:text-xs"
              >
                📜 Read Blog Post
              </a>
              <a
                href="https://github.com/adamkarvonen/SAEBench"
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-row items-center whitespace-pre rounded-full bg-black px-3 py-2 text-[10px] font-bold text-white hover:bg-slate-800 sm:ml-1 sm:flex sm:px-4 sm:py-2.5 sm:text-xs"
              >
                <svg className="mr-1.5 hidden h-4 w-4 sm:block" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
                GitHub
              </a>
              <a
                href="mailto:adam.karvonen@gmail.com,canrager@gmail.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-row items-center whitespace-pre rounded-full bg-slate-600 px-3 py-2 text-[10px] font-bold text-white hover:bg-slate-700 sm:ml-1 sm:flex sm:px-4 sm:py-2.5 sm:text-xs"
              >
                ✉️ Contact
              </a>
            </div>
          </div>
        </div>
      )}

      <div className="h-full w-full flex-row items-center justify-center gap-x-3 px-0 pt-0 text-slate-600">
        <EvalsTable evalTypes={evalTypes as unknown as EvalTypeWithPartialRelations[]} anonymized={false} />
      </div>
    </div>
  );
}
