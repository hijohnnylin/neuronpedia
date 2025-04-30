/* eslint no-nested-ternary: 0 */

import { useUmapContext, ZOOM_PADDING_X, ZOOM_PADDING_Y } from '@/components/provider/umap-provider';
import { getExplanationNeuronIdentifier, NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import * as Checkbox from '@radix-ui/react-checkbox';
import { CheckIcon } from 'lucide-react';
import { ExplanationWithPartialRelations } from 'prisma/generated/zod';

export default function UmapListRow({ modelId, exp }: { modelId: string; exp: ExplanationWithPartialRelations }) {
  const {
    searchText,
    loadingFeature,
    selectedFeatures,
    setSelectedFeatures,
    setGraphRanges,
    // , addAnnotationForExp
  } = useUmapContext();

  function findSelectedFeature(feature: NeuronIdentifier) {
    const selectedFeature = [...selectedFeatures.keys()].find((feat) => feat.equals(feature));
    if (selectedFeature) {
      return selectedFeature;
    }
    return undefined;
  }

  function makeHighlightedSearchText(text: string) {
    // highlight searchText in text
    const startIndex = text.toLowerCase().indexOf(searchText.toLowerCase());
    if (startIndex === -1) {
      return text;
    }
    return (
      <div>
        {text.substring(0, startIndex)}
        <span className="bg-amber-200">{text.substring(startIndex, startIndex + searchText.length)}</span>
        {text.substring(startIndex + searchText.length, text.length)}
      </div>
    );
  }

  function setZoomFromCenter(x: number, y: number) {
    setGraphRanges({
      minX: x * (1 - ZOOM_PADDING_X),
      maxX: x * (1 + ZOOM_PADDING_X),
      minY: y * (1 - ZOOM_PADDING_Y),
      maxY: y * (1 + ZOOM_PADDING_Y),
    });
  }

  return (
    <button
      type="button"
      className={`${
        loadingFeature && loadingFeature.equals(getExplanationNeuronIdentifier(exp))
          ? 'bg-slate-300'
          : findSelectedFeature(getExplanationNeuronIdentifier(exp))
            ? 'bg-emerald-200'
            : 'hover:bg-emerald-50'
      } group mr-0.5 flex flex-1 flex-row rounded border border-transparent px-1 py-[3px] text-left hover:border-emerald-600`}
      onClick={() => {
        setSelectedFeatures(() => {
          const newMap = new Map(selectedFeatures);
          const expFeature = getExplanationNeuronIdentifier(exp);
          const selectedFeature = findSelectedFeature(expFeature);
          if (selectedFeature) {
            newMap.delete(selectedFeature);
          } else {
            newMap.set(expFeature, {
              isEditing: false,
              description: '',
              neuron: null,
            });
          }
          return newMap;
        });
      }}
      onMouseEnter={() => {
        // addAnnotationForExp(exp as ExplanationWithPartialRelations);
      }}
    >
      <div className="flex w-20 flex-row items-start gap-x-1.5">
        <Checkbox.Root
          checked={findSelectedFeature(getExplanationNeuronIdentifier(exp)) !== undefined}
          className="mt-[0.5px] h-[14px] w-[14px] appearance-none items-center justify-center rounded-[3px] border border-slate-200 bg-white shadow outline-none hover:bg-slate-100"
        >
          <Checkbox.Indicator>
            <CheckIcon className="h-3 w-3 text-emerald-600" />
          </Checkbox.Indicator>
        </Checkbox.Root>
        <div className="text-xs uppercase">{exp.index}</div>
      </div>
      <div className="relative flex-1 text-xs">
        {searchText.length > 0 ? makeHighlightedSearchText(exp.description) : exp.description}
        <div className="absolute right-1 top-0 hidden flex-row gap-x-2.5 rounded bg-white px-2 text-emerald-600 group-hover:flex">
          <button
            type="button"
            onClick={async (e) => {
              e.stopPropagation();
              setZoomFromCenter(exp.umap_x || 0, exp.umap_y || 0);
            }}
            className="underline"
          >
            Zoom to
          </button>
          <button
            type="button"
            className="underline"
            onClick={(e) => {
              e.stopPropagation();
              window.open(`/${modelId}/${exp.layer}/${exp.index}`, '_blank', 'noreferrer');
            }}
          >
            ↗ Popup
          </button>
        </div>
      </div>
    </button>
  );
}
