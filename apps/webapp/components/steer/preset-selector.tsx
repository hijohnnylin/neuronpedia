import { Button } from '@/components/shadcn/button';
import { IS_ACTUALLY_NEURONPEDIA_ORG } from '@/lib/env';
import { FeaturePreset, SteerFeature } from '@/lib/utils/steer';
import { X } from 'lucide-react';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import { LoadingSquare } from '../svg/loading-square';

export default function SteerPresetSelector({
  loadingPresets,
  featurePresets,
  presetIsSelected,
  selectedFeatures,
  setSelectedFeatures,
  setSteerMethod,
  setShowSettingsOnMobile,
  deleteUserVector,
  loadSavedSteerOutput,
}: {
  loadingPresets: boolean;
  featurePresets: FeaturePreset[];
  presetIsSelected: (preset: FeaturePreset) => boolean;
  selectedFeatures: SteerFeature[];
  setSelectedFeatures: (features: SteerFeature[]) => void;
  setSteerMethod: (method: NPSteerMethod) => void;
  setShowSettingsOnMobile: (show: boolean) => void;
  deleteUserVector: (preset: FeaturePreset) => void;
  loadSavedSteerOutput: (id: string) => void;
}) {
  return (
    <div className="forceShowScrollBar flex max-h-[185px] flex-col gap-y-1 overflow-y-scroll rounded-lg bg-slate-50 py-2 pl-2 pr-2 sm:ml-2">
      {loadingPresets ? (
        <div className="flex h-[165px] min-h-[165px] flex-row items-center justify-center text-sky-500">
          <LoadingSquare className="text-sky-600" />
        </div>
      ) : (
        featurePresets.map((preset, pi) => (
          <div key={pi} className="flex flex-row items-center gap-x-2">
            <Button
              onClick={() => {
                if (presetIsSelected(preset)) {
                  setSelectedFeatures([]);
                } else {
                  setSelectedFeatures(preset.features);
                  if (preset.steerMethod) {
                    setSteerMethod(preset.steerMethod);
                  }
                  setShowSettingsOnMobile(false);
                }
              }}
              variant={presetIsSelected(preset) ? 'default' : 'outline'}
              className="relative flex flex-1 flex-row items-center justify-center border-slate-300 text-right text-xs font-normal"
            >
              {!preset.features[0].hasVector ? (
                <div className="absolute left-1 top-1 rounded bg-slate-100 px-1.5 py-1 text-[7px] leading-none text-slate-600">
                  Feature
                </div>
              ) : (
                <div className="absolute left-1 top-1 rounded bg-slate-100 px-1.5 py-1 text-[7px] leading-none text-slate-600">
                  Vector
                </div>
              )}
              {preset.isUserVector && (
                <div className="absolute right-1 top-1 rounded bg-amber-100 px-1.5 py-1 text-[7px] leading-none text-amber-700">
                  Created By You
                </div>
              )}
              <div>
                {preset.name}
                {preset.isUserVector ? '' : ' Mode'}
              </div>
            </Button>
            {preset.isUserVector && (
              <Button
                variant="outline"
                className="px-3 py-0 text-[10px] leading-tight text-slate-500"
                onClick={() => {
                  deleteUserVector(preset);
                }}
              >
                <X className="h-4 w-4 text-red-700" />
              </Button>
            )}
            {preset.exampleSteerOutputId && IS_ACTUALLY_NEURONPEDIA_ORG && (
              <Button
                variant="outline"
                className={`px-3 py-0 text-[10px] leading-tight ${
                  selectedFeatures.length === 0 ? 'border-sky-300 bg-sky-100 text-sky-600 shadow' : 'text-slate-500'
                }`}
                onClick={() => {
                  loadSavedSteerOutput(preset.exampleSteerOutputId || '');
                  setShowSettingsOnMobile(false);
                }}
              >
                Demo
              </Button>
            )}
          </div>
        ))
      )}
    </div>
  );
}
