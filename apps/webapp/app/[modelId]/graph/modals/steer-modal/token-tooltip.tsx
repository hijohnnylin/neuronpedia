import CustomTooltip from '@/components/custom-tooltip';
import { SteerResponseLogitsByToken } from '@/lib/utils/graph';

export default function TokenTooltip({ logitsByToken }: { logitsByToken: SteerResponseLogitsByToken }) {
  return (
    <div className="flex flex-wrap items-center gap-x-0 gap-y-[0px]">
      {logitsByToken.map((token, index) => {
        if (token.top_logits.length === 0) {
          return (
            <span
              key={`${token.token}-${index}`}
              className="h-[29px] max-h-[29px] min-h-[29px] cursor-default font-mono text-[12px] leading-[29px] text-slate-800"
            >
              {token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
            </span>
          );
        }
        return (
          <CustomTooltip
            key={`${token.token}-${index}`}
            trigger={
              <span
                className={`ml-[3px] h-[29px] max-h-[29px] min-h-[29px] cursor-pointer rounded px-[3px] py-[3px] font-mono text-[12px] leading-[29px] text-slate-800 transition-all ${
                  token.top_logits.length === 0 ? 'bg-slate-100' : 'bg-sky-100 hover:bg-sky-200'
                }`}
              >
                {token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
              </span>
            }
          >
            {token.top_logits.length === 0 ? (
              <div>{token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}</div>
            ) : (
              <div className="flex w-full min-w-[160px] flex-col gap-y-0.5">
                <div className="mb-2.5 self-center rounded bg-sky-200 px-[3px] py-[3px] font-mono text-slate-700">
                  {token.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                </div>
                <div className="mb-2 flex flex-row justify-between gap-x-3 border-b border-slate-300 pb-1">
                  <span className="text-xs text-slate-500">Next Token</span>
                  <span className="text-xs text-slate-500">Probability</span>
                </div>
                {token.top_logits.map((logit) => (
                  <div key={logit.token} className="flex flex-row items-center justify-between gap-x-1 font-mono">
                    <span className="rounded bg-slate-200 px-[3px] py-[2px] text-[11px] text-slate-700">
                      {logit.token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                    </span>
                    <span className="text-[11px] text-slate-600">{logit.prob.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </CustomTooltip>
        );
      })}
    </div>
  );
}
