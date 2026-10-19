'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { saeHasConfig } from '@/lib/utils/source';
import { SourceWithPartialRelations } from '@/prisma/generated/zod';
import { Prisma } from '@prisma/client';
import copy from 'copy-to-clipboard';
import { ArrowUpRight, HelpCircle } from 'lucide-react';
import { useState } from 'react';

const FEATURED_METADATA_KEYS = [
  'd_sae',
  'architecture',
  'activation_fn_str',
  'context_size',
  'hook_name',
  'hook_layer',
  'hook_point',
  'hook_point_layer',
  'dataset_path',
  'dtype',
];

const KEY_TO_LABEL: Record<string, { label: string; description: string }> = {
  d_sae: {
    label: 'Features',
    description:
      "[d_sae] Sets the dimension of the SAE's hidden layer. This represents the number of possible feature activations.",
  },
  architecture: {
    label: 'Architecture',
    description: '[architecture] Specifies the type of SAE architecture being used.',
  },
  activation_fn_str: {
    label: 'Activation Function',
    description: '[activation_fn_str] Specifies the activation function used in the SAE.',
  },
  context_size: {
    label: 'Context Size',
    description:
      "[context_size] Defines the size of the context window. SAEs trained on small activations from small prompts often don't perform well on longer prompts.",
  },
  hook_name: {
    label: 'Hook Name',
    description: '[hook_name] Indicates the specific hook in the model where the SAE is applied.',
  },
  hook_point: {
    label: 'Hook Point',
    description: '[hook_point] Indicates the specific hook point in the model where the SAE is applied.',
  },
  hook_layer: {
    label: 'Hook Layer',
    description: '[hook_layer] Specifies the layer number where the hook is applied.',
  },
  hook_point_layer: {
    label: 'Hook Point Layer',
    description: '[hook_point_layer] Specifies the layer number where the hook point is applied.',
  },
  hook_head_index: {
    label: 'Hook Head Index',
    description: '[hook_head_index] Defines which attention head to hook into.',
  },
  prepend_bos: {
    label: 'Prepend BOS',
    description: '[prepend_bos] Determines whether to prepend the beginning-of-sequence token.',
  },
  dataset_path: {
    label: 'Dataset',
    description: '[dataset_path] Specifies the path to the dataset used for training or evaluation.',
  },
  dataset_trust_remote_code: {
    label: 'Dataset Trust Remote Code',
    description:
      '[dataset_trust_remote_code] Indicates whether to trust remote code (from HuggingFace) when loading the dataset.',
  },
  normalize_activations: {
    label: 'Normalize Activations',
    description: '[normalize_activations] Specifies how to normalize activations.',
  },
  dtype: {
    label: 'Data Type',
    description: '[dtype] Defines the data type for tensor operations.',
  },
  device: {
    label: 'Device',
    description: '[device] Specifies the computational device to use.',
  },
  sae_lens_training_version: {
    label: 'SAELens Training Version',
    description: '[sae_lens_training_version] Indicates the version of SAE Lens used for training.',
  },
  activation_fn_kwargs: {
    label: 'Activation Function Arguments',
    description:
      '[activation_fn_kwargs] Allows for additional keyword arguments for the activation function. This would be used if e.g. the activation_fn_str was set to topk, so that k could be specified.',
  },
  d_in: {
    label: 'Input Dimension',
    description: '[d_in] Defines the input dimension of the SAE.',
  },
  apply_b_dec_to_input: {
    label: 'Apply Decoder Bias to Input',
    description: '[apply_b_dec_to_input] Determines whether to apply the decoder bias to the input.',
  },
  finetuning_scaling_factor: {
    label: 'Finetuning Scaling Factor',
    description:
      '[finetuning_scaling_factor] Indicates whether to use a scaling factor to weight initialization and the forward pass. This is not usually used and was introduced to support a solution for shrinkage.',
  },
  model_name: {
    label: 'Model Name',
    description:
      '[model_name] Specifies the name of the model being used - should be a valid model in TransformerLens.',
  },
};

function ConfigItem({
  value,
  label,
  labelDescription,
}: {
  value: string | number;
  label: string;
  labelDescription: string;
}) {
  return (
    <div className="col-span-1 mb-1 flex w-full flex-col items-center justify-center text-xs">
      <div className="text-[11px] font-medium text-slate-400">
        <CustomTooltip
          trigger={
            <div className="group flex cursor-pointer flex-row items-center gap-x-1.5 rounded px-2 text-center hover:bg-slate-200">
              <div className="font-sans text-[9.5px] font-normal leading-none text-slate-400">{label}</div>
            </div>
          }
        >
          {labelDescription}
        </CustomTooltip>
      </div>
      <div className="w-full text-wrap break-all text-center font-mono text-[10.5px] font-bold text-slate-600">
        {value}
      </div>
    </div>
  );
}

export default function SaeLensConfigPane({
  sae,
  numPrompts,
  numTokensInPrompt,
  dashboardDataset,
  inSAEPage = false,
}: {
  sae: SourceWithPartialRelations;
  numPrompts?: number;
  numTokensInPrompt?: number;
  dashboardDataset?: string;
  inSAEPage?: boolean;
}) {
  const config = sae.saelensConfig ? (sae.saelensConfig as Prisma.JsonObject) : undefined;
  const configKeys = config ? Object.keys(config) : [];

  const hfDownloadLink =
    sae.hfRepoId && sae.hfFolderId ? `https://huggingface.co/${sae.hfRepoId}/tree/main/${sae.hfFolderId}` : undefined;

  const [showMoreConfig, setShowMoreConfig] = useState(false);

  if (!saeHasConfig(sae) && !(numPrompts && numTokensInPrompt) && !dashboardDataset) {
    return null;
  }

  return (
    <Card className="w-full bg-white">
      <CardHeader className="w-full pb-0 pt-0">
        {inSAEPage ? (
          <CardTitle className="flex flex-row items-center gap-x-2.5 pb-0 pt-5">
            Configuration
            <CustomTooltip trigger={<HelpCircle className="h-3.5 w-3.5 text-slate-500" />}>
              The configuration used to generate the dashboard activations, as well as the SAELens configuration used to
              train the SAE.
            </CustomTooltip>
          </CardTitle>
        ) : (
          <div className="flex flex-col pt-2">
            <div className="mb-0 flex w-full flex-row items-center justify-center gap-x-1 text-[10px] font-normal uppercase text-slate-400">
              Configuration{' '}
              <CustomTooltip trigger={<HelpCircle className="h-2.5 w-2.5" />}>
                The configuration used to generate the dashboard activations, as well as the SAELens configuration used
                to train the SAE.
              </CustomTooltip>
            </div>
          </div>
        )}

        <div className="flex w-full flex-row items-center justify-center gap-x-3 pb-3 pt-1">
          {hfDownloadLink && (
            <a
              href={hfDownloadLink}
              className={`flex ${
                inSAEPage ? 'max-w-full' : 'max-w-[220px]'
              } flex-row items-center justify-center gap-x-1 break-all text-center text-[11px] leading-snug text-sky-700 hover:underline`}
              target="_blank"
              rel="noopener noreferrer"
            >
              {sae.hfRepoId}/{sae.hfFolderId}
            </a>
          )}
          {sae.saelensSaeId && sae.saelensRelease && (
            <div className="flex">
              <CustomTooltip
                wide
                trigger={
                  <Button variant="outline" className="mb-0.5 h-7 whitespace-pre text-[10px] text-slate-500" size="sm">
                    How To Load
                  </Button>
                }
              >
                To load this SAE using SAELens:
                <br />
                <br />
                <code className="mb-2 mt-2 py-2.5 font-bold text-slate-600">from sae_lens import SAE</code>
                <br />
                <code className="mb-2 py-2.5 font-bold text-slate-600">
                  {`
                    sae, cfg_dict, sparsity = SAE.from_pretrained(release="${sae.saelensRelease}",
                         sae_id="${sae.saelensSaeId}",
                         device=device)
                  `}
                </code>
                <br />
                <Button
                  size="sm"
                  variant="outline"
                  className="my-2 cursor-pointer"
                  onClick={() => {
                    copy(
                      `from sae_lens import SAE\n\nsae, cfg_dict, sparsity = SAE.from_pretrained(release="${sae.saelensRelease}", sae_id="${sae.saelensSaeId}", device=device)`,
                    );
                    alert('Copied to clipboard!');
                  }}
                >
                  Copy Code
                </Button>
                <br />
                <br />
                <a
                  target="_blank"
                  rel="noreferrer"
                  className="flex flex-row items-center gap-x-1 font-semibold text-sky-700 hover:underline"
                  href="https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb"
                >
                  SAE Loading Tutorial <ArrowUpRight className="h-3.5 w-3.5" />
                </a>
              </CustomTooltip>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col items-start justify-between pb-4 pt-0">
        <div className={`grid w-full ${inSAEPage ? 'grid-cols-4' : 'grid-cols-2'} gap-x-2 gap-y-0`}>
          {numPrompts && numTokensInPrompt && (
            <ConfigItem
              value={`${numPrompts.toLocaleString()} prompts, ${numTokensInPrompt} tokens each`}
              label="Prompts (Dashboard)"
              labelDescription="Number of prompts and tokens per prompt used to generate the dashboard activations."
            />
          )}
          {dashboardDataset && (
            <ConfigItem
              value={dashboardDataset || ''}
              label="Dataset (Dashboard)"
              labelDescription="Dataset used to generate the dashboard activations."
            />
          )}
          {configKeys.length > 0 ? (
            configKeys
              .filter((key) => FEATURED_METADATA_KEYS.includes(key))
              .map((key) => (
                <ConfigItem
                  key={key}
                  value={
                    config
                      ? typeof config[key] === 'string'
                        ? (config[key] as string)
                        : typeof config[key] === 'number'
                          ? (config[key] as number).toLocaleString()
                          : JSON.stringify(config[key]).replace(/,/g, ', ')
                      : ''
                  }
                  label={Object.keys(KEY_TO_LABEL).includes(key) ? KEY_TO_LABEL[key].label : key}
                  labelDescription={Object.keys(KEY_TO_LABEL).includes(key) ? KEY_TO_LABEL[key].description : ''}
                />
              ))
          ) : (
            <div
              className={`${
                inSAEPage ? 'col-span-4' : 'hidden'
              } mb-3 mt-0 w-full text-center text-sm font-bold text-slate-300`}
            >
              No Configuration Found
            </div>
          )}
          {showMoreConfig
            ? configKeys
                .filter((key) => !FEATURED_METADATA_KEYS.includes(key))
                .map((key) => (
                  <ConfigItem
                    key={key}
                    value={
                      config
                        ? typeof config[key] === 'string'
                          ? (config[key] as string)
                          : typeof config[key] === 'number'
                            ? (config[key] as number).toLocaleString()
                            : JSON.stringify(config[key]).replace(/,/g, ', ')
                        : ''
                    }
                    label={Object.keys(KEY_TO_LABEL).includes(key) ? KEY_TO_LABEL[key].label : key}
                    labelDescription={Object.keys(KEY_TO_LABEL).includes(key) ? KEY_TO_LABEL[key].description : ''}
                  />
                ))
            : ''}
          <div
            className={`${inSAEPage ? 'col-span-4' : 'col-span-2'} mt-1.5 ${
              configKeys.length > 0 ? 'flex' : 'hidden'
            } w-full justify-center`}
          >
            <Button
              onClick={() => setShowMoreConfig(!showMoreConfig)}
              variant="outline"
              className="h-6 text-[10px] text-slate-400"
              size="sm"
            >
              {showMoreConfig ? 'Hide Extra' : 'Show All'}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
