import CustomTooltip from '@/components/custom-tooltip';
import { Info } from 'lucide-react';
import { useEffect, useState } from 'react';
import ActivationItem from './activation-item';
import { useExplanationScoreDetailContext } from './provider/explanation-score-detail-provider';
import { useGlobalContext } from './provider/global-provider';
import { Dialog, DialogContent, DialogTitle } from './shadcn/dialog';
import { LoadingSquare } from './svg/loading-square';

type RecallAltJsonDetails = {
  top: {
    id: string;
    tokens: string[];
    values: number[];
    match: boolean;
    reason: string;
  }[];
  zero: {
    id: string;
    tokens: string[];
    values: number[];
    match: boolean;
    reason: string;
  }[];
  decoy: {
    id: number;
    tokens: string[];
    values: number[];
    match: boolean;
    reason: string;
  }[];
};

type EleutherFuzzOrRecallJsonDetails = {
  text: string;
  str_tokens: string[];
  activations: number[];
  distance: number;
  ground_truth: boolean;
  prediction: number;
  highlighted: boolean;
  correct: boolean;
  probability: number;
}[];

type EleutherEmbeddingJsonDetails = {
  text: string;
  distance: number;
  similarity: number;
  activations: number[];
  str_tokens: string[];
}[];

type NlaVerbalizerJsonDetails = {
  nla_explanation: string;
  nla_text_used: string;
  original_explanation: string;
  embedding_model: string;
  embedding_dimensions: number;
  cosine_similarity: number;
  hfRepoId: string;
  hfFolderId: string;
  index: number;
};

export default function ExplanationScoreDetailDialog() {
  const {
    explanationScoreDetailLoading,
    explanationScoreDetailOpen,
    setExplanationScoreDetailOpen,
    explanationScoreDetailExplanation,
    explanationScoreDetailScore,
  } = useExplanationScoreDetailContext();

  const [recallParsedJson, setRecallParsedJson] = useState<RecallAltJsonDetails>();
  const [eleutherFuzzOrRecallJson, setEleutherFuzzOrRecallJson] = useState<EleutherFuzzOrRecallJsonDetails>();
  const [eleutherEmbeddingJson, setEleutherEmbeddingJson] = useState<EleutherEmbeddingJsonDetails>();
  const [nlaVerbalizerJson, setNlaVerbalizerJson] = useState<NlaVerbalizerJsonDetails>();

  const { explanationScoreTypes } = useGlobalContext();

  useEffect(() => {
    if (explanationScoreDetailScore) {
      const parsedJson = JSON.parse(explanationScoreDetailScore.jsonDetails);
      if (explanationScoreDetailScore.explanationScoreTypeName === 'recall_alt') {
        setRecallParsedJson(parsedJson);
      } else if (
        explanationScoreDetailScore.explanationScoreTypeName === 'eleuther_fuzz' ||
        explanationScoreDetailScore.explanationScoreTypeName === 'eleuther_recall'
      ) {
        let result = parsedJson as EleutherFuzzOrRecallJsonDetails;
        // eleuther's quantiles are flipped, so we need to change 10 -> 1, 9 -> 2, 1 -> 10, etc
        result = result.map((res) => {
          if (res.distance === -1) {
            return res;
          }
          res.distance = 11 - res.distance;
          return res;
        });
        // sort by result["distance"]
        result.sort((a, b) => b.distance - a.distance);
        result = result.filter((item) => item.prediction !== -1);
        setEleutherFuzzOrRecallJson(result);
      } else if (explanationScoreDetailScore.explanationScoreTypeName === 'eleuther_embedding') {
        let result = parsedJson as EleutherEmbeddingJsonDetails;
        // eleuther's quantiles are flipped, so we need to change 10 -> 1, 9 -> 2, 1 -> 10, etc
        result = result.map((res) => {
          if (res.distance === -1) {
            return res;
          }
          res.distance = 11 - res.distance;
          return res;
        });
        // sort by result["distance"]
        result.sort((a, b) => b.distance - a.distance);
        setEleutherEmbeddingJson(result);
      } else if (
        explanationScoreDetailScore.explanationScoreTypeName === 'nla_verbalizer' ||
        explanationScoreDetailScore.explanationScoreTypeName === 'nla_verbalizer_last'
      ) {
        setNlaVerbalizerJson(parsedJson as NlaVerbalizerJsonDetails);
      }
    }
  }, [explanationScoreDetailScore]);

  return (
    <Dialog open={explanationScoreDetailOpen} onOpenChange={setExplanationScoreDetailOpen}>
      <DialogContent className="fixed left-[50%] top-[50%] z-50 flex max-h-[92vh] w-[95vw] max-w-[95vw] translate-x-[-50%] translate-y-[-50%] flex-col items-center justify-center rounded-md bg-slate-50 shadow-xl transition-all focus:outline-none sm:top-[50%] sm:w-[90vw] md:rounded-md md:border md:border-slate-200">
        {explanationScoreDetailLoading ? (
          <div className="flex flex-1 flex-col items-center justify-center py-5">
            <DialogTitle className="text-center">
              <div className="text-xs font-bold uppercase text-slate-400">Loading Score Details</div>
            </DialogTitle>
            <LoadingSquare size={32} className="text-sky-700" />
          </div>
        ) : (
          <div className="z-50 flex w-full flex-col gap-y-0 overflow-scroll px-0 text-xs">
            <div className="top-0 flex w-full flex-col items-start justify-center border-b bg-slate-50 pt-4 sm:sticky">
              <div className="relative mb-4 flex w-full flex-row gap-x-5 px-2 text-left text-sm text-slate-600 sm:px-6 sm:text-base">
                <div className="flex basis-2/6 flex-col items-start">
                  <div className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                    Score
                    <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                      The score of the explanation according to the scoring method. See details for the score below.
                    </CustomTooltip>
                  </div>
                  <div className="rounded-xl bg-emerald-500 p-3 text-lg font-bold text-white">
                    {explanationScoreDetailScore && Math.round(explanationScoreDetailScore.value * 100)}
                  </div>
                  <div className="mt-2 text-xs text-slate-600">
                    {
                      explanationScoreTypes.find(
                        (s) => s.name === explanationScoreDetailScore?.explanationScoreTypeName,
                      )?.scoreDescription
                    }
                  </div>
                </div>
                <div className="flex basis-1/3 flex-col">
                  <div className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                    Explanation
                    <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                      The explanation that is being scored.
                    </CustomTooltip>
                  </div>
                  <div className="text-[13px] font-medium leading-normal">
                    {explanationScoreDetailExplanation?.description}
                  </div>

                  <div className="mt-1 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                    Auto-Interp Method and Model
                    <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                      The method and model used to generate the explanation.
                    </CustomTooltip>
                  </div>
                  <a
                    href={`/score-type/${explanationScoreDetailExplanation?.typeName}`}
                    className="mt-0 whitespace-pre text-[11px] leading-none text-sky-700 hover:underline"
                    target="_blank"
                    rel="noreferrer"
                  >
                    {explanationScoreDetailExplanation?.typeName} ·{' '}
                    {explanationScoreDetailExplanation?.explanationModelName}
                  </a>
                </div>
                <div className="flex basis-2/6 flex-col">
                  <div className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                    Scorer Type + Model
                    <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                      The scoring type and model used to score the explanation.
                    </CustomTooltip>
                  </div>
                  <a
                    href={`/score-type/${explanationScoreDetailScore?.explanationScoreTypeName}`}
                    className="whitespace-pre text-sm text-sky-700 hover:underline"
                    target="_blank"
                    rel="noreferrer"
                  >
                    {explanationScoreDetailScore?.explanationScoreTypeName} ·{' '}
                    {explanationScoreDetailScore?.explanationScoreModelName}
                  </a>
                  <div className="mt-2 text-xs text-slate-600">
                    {
                      explanationScoreTypes.find(
                        (s) => s.name === explanationScoreDetailScore?.explanationScoreTypeName,
                      )?.description
                    }
                  </div>
                </div>
              </div>
            </div>
            {explanationScoreDetailScore?.explanationScoreTypeName === 'recall_alt' && (
              <div className="flex w-full flex-col gap-y-2 px-4 pb-10 pt-5">
                <div className="flex flex-row items-start justify-center gap-x-3">
                  <div className="mb-1 mt-1 w-36 min-w-36 max-w-36 text-left text-sm font-bold text-slate-600">
                    <span className="whitespace-pre">Top Activations</span>
                    <div className="mt-3 text-xs font-medium text-slate-600">
                      Should Match
                      <div className="mt-2 leading-snug text-slate-500">
                        Test ten texts known to be top activations for the feature.
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-1 flex-col items-start justify-start gap-y-2">
                    {recallParsedJson?.top.map((activation, i) => (
                      <div key={i} className="flex w-full flex-col">
                        <div className="flex w-full flex-row items-center justify-between gap-x-3">
                          <div className="w-[50%] min-w-[50%]">
                            <ActivationItem
                              tokensToDisplayAroundMaxActToken={32}
                              activation={{
                                tokens: activation.tokens,
                                values: activation.values,
                                maxValueTokenIndex: activation.values.indexOf(Math.max(...activation.values)),
                              }}
                              overallMaxActivationValueInList={activation.values.indexOf(
                                Math.max(...activation.values),
                              )}
                              overrideTextSize="text-[10.5px]"
                            />
                          </div>
                          <div className="w-32 min-w-32 max-w-32 text-center text-[13px] font-bold text-slate-600">
                            {activation.match ? 'Matched' : 'Not Matched'}
                          </div>
                          <div className="flex-1 text-xs text-slate-600">{activation.reason}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                {recallParsedJson?.zero && recallParsedJson.zero.length > 0 && (
                  <div className="mt-3 flex flex-row items-start justify-center gap-x-3">
                    <div className="mb-1 mt-1 w-36 min-w-36 max-w-36 text-left text-sm font-bold text-slate-600">
                      <span className="whitespace-pre">Zero Activations</span>
                      <div className="mt-3 text-xs font-medium text-slate-600">
                        Should Not Match
                        <div className="mt-2 leading-snug text-slate-500">
                          Test five texts known to not activate the feature.
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-col items-center justify-center gap-y-2">
                      {recallParsedJson?.zero.map((activation, i) => (
                        <div key={i} className="flex w-full flex-col">
                          <div className="flex w-full flex-row items-center justify-between gap-x-3">
                            <div className="w-[50%] min-w-[50%]">
                              <ActivationItem
                                tokensToDisplayAroundMaxActToken={32}
                                activation={{
                                  tokens: activation.tokens,
                                  values: activation.values,
                                  maxValueTokenIndex: activation.values.indexOf(Math.max(...activation.values)),
                                }}
                                overallMaxActivationValueInList={activation.values.indexOf(
                                  Math.max(...activation.values),
                                )}
                                overrideTextSize="text-[10.5px]"
                              />
                            </div>
                            <div className="w-32 min-w-32 max-w-32 text-center text-sm font-bold text-slate-600">
                              {activation.match ? 'Matched' : 'Not Matched'}
                            </div>
                            <div className="flex-1 text-xs text-slate-600">{activation.reason}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="mt-3 flex flex-row items-start justify-center gap-x-3">
                  <div className="mb-1 mt-1 w-36 min-w-36 max-w-36 text-left text-sm font-bold text-slate-600">
                    <span className="whitespace-pre">Decoy Activations</span>
                    <div className="mt-3 text-xs font-medium text-slate-600">
                      Should Not Match
                      <div className="mt-2 leading-snug text-slate-500">
                        Test five pre-determined activation texts with made-up activation values. This checks that the
                        explanation {`isn't just matching random texts.`}
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-center justify-center gap-y-2">
                    {recallParsedJson?.decoy.map((activation, i) => (
                      <div key={i} className="flex w-full flex-col">
                        <div className="flex w-full flex-row items-center gap-x-3">
                          <div className="w-[50%] min-w-[50%] max-w-[50%]">
                            <ActivationItem
                              tokensToDisplayAroundMaxActToken={32}
                              activation={{
                                tokens: activation.tokens,
                                values: activation.values,
                                maxValueTokenIndex: activation.values.indexOf(Math.max(...activation.values)),
                              }}
                              overallMaxActivationValueInList={activation.values.indexOf(
                                Math.max(...activation.values),
                              )}
                              overrideTextSize="text-[10.5px]"
                            />
                          </div>
                          <div className="w-32 min-w-32 max-w-32 text-center text-sm font-bold text-slate-600">
                            {activation.match ? 'Matched' : 'Not Matched'}
                          </div>
                          <div className="flex-1 text-xs text-slate-600">{activation.reason}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {(explanationScoreDetailScore?.explanationScoreTypeName === 'eleuther_fuzz' ||
              explanationScoreDetailScore?.explanationScoreTypeName === 'eleuther_recall') && (
              <div className="flex w-full flex-col gap-y-2 px-5 pb-10 pt-2 text-slate-600">
                <table>
                  <tr className="">
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Text{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>The text that was scored.</CustomTooltip>
                    </th>
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Quantile{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        Quantile of the text - higher is more activating.
                      </CustomTooltip>
                    </th>
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Probability{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        The probability that the model thinks that the context (or token, in the case of fuzzing)
                        activates the feature.
                        <br />
                        <br />
                        Specifically, the prompt asks the model to output 0 if it thinks that the feature is not active,
                        or 1 if it thinks the features is active. We take the logprob of 0 and 1, we exponentiate it and
                        then do prob(1)/(prob(1)+prob(0)).
                      </CustomTooltip>
                    </th>
                    {/* <th className="px-2 py-1.5">Ground Truth</th> */}
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Prediction{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        The {`model's`} prediction for this text.
                      </CustomTooltip>
                    </th>
                    {/* <th className="px-2 py-1.5">Highlighted</th> */}
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Correct{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        Whether the {`model's`} prediction is correct.
                      </CustomTooltip>
                    </th>
                  </tr>
                  {eleutherFuzzOrRecallJson?.map((item, i) => (
                    <tr key={i} className="">
                      <td className="px-2 py-1.5 font-mono text-[10px]">
                        <ActivationItem
                          tokensToDisplayAroundMaxActToken={32}
                          activation={{
                            tokens: item.str_tokens,
                            values: item.activations,
                          }}
                          overallMaxActivationValueInList={item.activations.indexOf(Math.max(...item.activations))}
                          overrideTextSize="text-[10.5px]"
                        />
                      </td>
                      <td className="whitespace-pre px-2 text-center">
                        {item.distance === -1 ? 'Non-Activating' : item.distance}
                      </td>
                      <td className="whitespace-pre px-2 text-center">{(item.probability * 100).toFixed(2)}%</td>
                      <td className="whitespace-pre px-2 text-center">
                        {item.prediction ? 'Activating' : 'Non-Activating'}
                      </td>
                      <td className="whitespace-pre px-2 text-center">{item.correct ? 'Correct' : 'Incorrect'}</td>
                    </tr>
                  ))}
                </table>
              </div>
            )}

            {(explanationScoreDetailScore?.explanationScoreTypeName === 'nla_verbalizer' ||
              explanationScoreDetailScore?.explanationScoreTypeName === 'nla_verbalizer_last') &&
              nlaVerbalizerJson && (
                <div className="flex w-full flex-col gap-y-4 px-5 pb-10 pt-4 text-slate-600">
                  <div className="flex flex-col gap-y-1">
                    <div className="text-[10px] font-medium uppercase text-slate-400">NLA Explanation (Generated)</div>
                    <div className="whitespace-pre-wrap rounded-md border border-slate-200 bg-white px-3 py-2 text-[13px] leading-relaxed text-slate-700">
                      {nlaVerbalizerJson.nla_explanation}
                    </div>
                  </div>
                  {nlaVerbalizerJson.nla_text_used !== nlaVerbalizerJson.nla_explanation && (
                    <div className="flex flex-col gap-y-1">
                      <div className="text-[10px] font-medium uppercase text-slate-400">
                        NLA Text Used for Comparison (Last Paragraph)
                      </div>
                      <div className="whitespace-pre-wrap rounded-md border border-slate-200 bg-white px-3 py-2 text-[13px] leading-relaxed text-slate-700">
                        {nlaVerbalizerJson.nla_text_used}
                      </div>
                    </div>
                  )}
                  <div className="flex flex-col gap-y-1">
                    <div className="text-[10px] font-medium uppercase text-slate-400">
                      Original Explanation (Being Scored)
                    </div>
                    <div className="whitespace-pre-wrap rounded-md border border-slate-200 bg-white px-3 py-2 text-[13px] leading-relaxed text-slate-700">
                      {nlaVerbalizerJson.original_explanation}
                    </div>
                  </div>
                  <div className="flex flex-row gap-x-6">
                    <div className="flex flex-col gap-y-1">
                      <div className="text-[10px] font-medium uppercase text-slate-400">Embedding Model</div>
                      <div className="font-mono text-xs text-slate-600">{nlaVerbalizerJson.embedding_model}</div>
                    </div>
                    <div className="flex flex-col gap-y-1">
                      <div className="text-[10px] font-medium uppercase text-slate-400">Dimensions</div>
                      <div className="font-mono text-xs text-slate-600">{nlaVerbalizerJson.embedding_dimensions}</div>
                    </div>
                    <div className="flex flex-col gap-y-1">
                      <div className="text-[10px] font-medium uppercase text-slate-400">Cosine Similarity</div>
                      <div className="font-mono text-xs text-slate-600">
                        {nlaVerbalizerJson.cosine_similarity.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

            {explanationScoreDetailScore?.explanationScoreTypeName === 'eleuther_embedding' && (
              <div className="flex w-full flex-col gap-y-2 px-5 pb-10 pt-2 text-slate-600">
                <table>
                  <tr className="">
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Text{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>The text that was scored.</CustomTooltip>
                    </th>
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Quantile{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        Quantile of the text - higher is more activating.
                      </CustomTooltip>
                    </th>
                    <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                      Similarity{' '}
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        The cosine similarity between the {`text's`} embedding and the {`explanation's`} embedding.
                      </CustomTooltip>
                    </th>
                  </tr>
                  {eleutherEmbeddingJson?.map((item, i) => (
                    <tr key={i} className="">
                      <td className="px-2 py-1.5 font-mono text-[10px]">
                        <ActivationItem
                          tokensToDisplayAroundMaxActToken={32}
                          activation={{
                            tokens: item.str_tokens,
                            values: item.activations,
                          }}
                          overallMaxActivationValueInList={item.activations.indexOf(Math.max(...item.activations))}
                          overrideTextSize="text-[10.5px]"
                        />
                      </td>
                      <td className="whitespace-pre px-2 text-center">
                        {item.distance === -1 ? 'Non-Activating' : item.distance}
                      </td>
                      <td className="whitespace-pre px-2 text-center">{item.similarity.toFixed(2)}</td>
                    </tr>
                  ))}
                </table>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
