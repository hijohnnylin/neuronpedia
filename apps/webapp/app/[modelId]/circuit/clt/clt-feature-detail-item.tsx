import ActivationItem, { CENTER_ME_CLASSNAME } from '@/components/activation-item';
import { useEffect, useRef } from 'react';
import { CLTFeatureExample } from './clt-utils';

export default function CLTFeatureDetailItem({
  example,
  overallMaxActivationValue,
  itemKey,
}: {
  example: CLTFeatureExample | undefined;
  overallMaxActivationValue: number;
  itemKey: number | string; // Use the type you use for keys
}) {
  const activationItemWrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (example && activationItemWrapRef.current) {
      const item = activationItemWrapRef.current;
      const timer = setTimeout(() => {
        const centerElement = item.querySelector(`.${CENTER_ME_CLASSNAME}`);
        if (centerElement && item instanceof HTMLElement) {
          if (centerElement instanceof HTMLElement && item.scrollWidth > item.clientWidth) {
            const centerRect = centerElement.getBoundingClientRect();

            const centerElementCenter = centerRect.left + centerRect.width / 2 - item.getBoundingClientRect().left;
            const containerCenter = item.clientWidth / 2;

            const scrollAdjustment = centerElementCenter - containerCenter;

            item.scrollLeft += scrollAdjustment;
          }
        }
      }, 20);

      return () => clearTimeout(timer);
    }
    return () => {};
  }, [example]);

  if (!example) {
    return null; // Or some placeholder
  }

  return (
    <div key={itemKey} className="flex w-full flex-col items-center">
      <div ref={activationItemWrapRef} className="activation-item-wrap max-w-full overflow-x-auto overscroll-x-none">
        <ActivationItem
          enableExpanding={false}
          tokensToDisplayAroundMaxActToken={36}
          activation={{
            id: `example-${itemKey}`,
            tokens: example.tokens,
            values: example.tokens_acts_list,
            maxValueTokenIndex: example.tokens_acts_list?.indexOf(Math.max(...(example.tokens_acts_list || []))),
          }}
          overallMaxActivationValueInList={overallMaxActivationValue}
          centerAndBorderOnTokenIndex={example.train_token_ind}
          overrideTextSize="text-[10.5px] rounded-md leading-normal min-h-[19px]"
          className="w-max min-w-full whitespace-nowrap"
        />
      </div>
    </div>
  );
}
