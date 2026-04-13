import CustomTooltip from '@/components/custom-tooltip';
import { Metadata } from 'next';
import { Fragment } from 'react';
import { GraphInfoContent } from './content';

const TITLE = 'The Circuits Research Landscape: Results and Perspectives';
const DESCRIPTION = 'A multi-organization interpretability project to replicate and extend circuit tracing research.';

export async function generateMetadata(): Promise<Metadata> {
  const title = `${TITLE} - August 2025`;

  return {
    title,
    description: DESCRIPTION,
    openGraph: {
      title,
      description: DESCRIPTION,
      authors: ['Anthropic', 'EleutherAI', 'Goodfire AI', 'Google Deepmind', 'Decode Research'],
      type: 'article',
      publishedTime: '2025-08-05',
      modifiedTime: '2025-08-05',
      tags: ['interpretability', 'circuit tracing', 'attribution graphs'],
      url: `/graph/info`,
      images: [
        {
          url: 'https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets/graph/post/circuits-landscape.jpg',
          width: 1200,
          height: 630,
        },
      ],
    },
  };
}

export default function Page() {
  // randomize order of orgs displayed
  const tooltips = [
    <CustomTooltip
      key="decode"
      trigger={
        <a href="https://decoderesearch.org" target="_blank" rel="noopener noreferrer" className="hover:underline">
          Decode
        </a>
      }
      wide
      side="bottom"
    >
      <div className="flex flex-col gap-y-0.5 text-slate-700">
        <span>Curt Tigges*, Johnny Lin</span>
        <span className="text-[10px] italic text-slate-400">*Core Contributor</span>
      </div>
    </CustomTooltip>,
    <CustomTooltip
      key="eleuther"
      trigger={
        <a href="https://eleuther.ai" target="_blank" rel="noopener noreferrer" className="hover:underline">
          EleutherAI
        </a>
      }
      wide
      side="bottom"
    >
      <div className="flex flex-col gap-y-0.5 text-slate-700">
        <span>Stepan Shabalin*, Gonçalo Paulo</span>
        <span className="text-[10px] italic text-slate-400">*Core Contributor</span>
      </div>
    </CustomTooltip>,
    <CustomTooltip
      key="goodfire"
      trigger={
        <a href="https://goodfire.ai" target="_blank" rel="noopener noreferrer" className="hover:underline">
          Goodfire AI
        </a>
      }
      wide
      side="bottom"
    >
      <div className="flex flex-col gap-y-0.5 text-slate-700">
        <span>Owen Lewis*, Jack Merullo*, Connor Watts*, Liv Gorton, Elana Simon, Max Loeffler, Tom McGrath</span>
        <span className="text-[10px] italic text-slate-400">*Core Contributor</span>
      </div>
    </CustomTooltip>,
    <CustomTooltip
      key="deepmind"
      trigger={
        <a
          href="https://deepmindsafetyresearch.medium.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline"
        >
          Google Deepmind
        </a>
      }
      wide
      side="bottom"
    >
      <div className="flex flex-col gap-y-0.5 text-slate-700">
        <span>Neel Nanda*, Callum McDougall</span>
        <span className="text-[10px] italic text-slate-400">*Core Contributor</span>
      </div>
    </CustomTooltip>,
  ];

  const shuffled = [...tooltips].sort(() => Math.random() - 0.5);
  const withCommas = shuffled.map((tooltip, index) => (
    <Fragment key={tooltip.key}>
      {index > 0 && <span className="text-slate-500">, </span>}
      {tooltip}
    </Fragment>
  ));

  // put anthropic at the front
  withCommas.unshift(
    <Fragment key="anthropic">
      <CustomTooltip
        key="anthropic"
        trigger={
          <a
            href="https://www.anthropic.com/research#interpretability"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline"
          >
            Anthropic
          </a>
        }
        wide
        side="bottom"
      >
        <div className="flex flex-col gap-y-0.5 text-slate-700">
          <span>Mateusz Piotrowski†*, Michael Hanna†*, Emmanuel Ameisen*, Jack Lindsey*, Joshua Batson</span>
          <span className="text-[10px] italic text-slate-400">†Anthropic Fellow</span>
          <span className="text-[10px] italic text-slate-400">*Core Contributor</span>
        </div>
      </CustomTooltip>
      {}
      <span className="text-slate-500">, </span>
    </Fragment>,
  );

  return <GraphInfoContent title={TITLE} randomizedOrgs={withCommas} />;
}
