import { DEFAULT_STEER_MODEL } from '@/lib/env';
import SteererSimple from '../../../components/steer/steerer-simple';

export default function Page() {
  return (
    <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white px-0">
      <SteererSimple initialModelId={DEFAULT_STEER_MODEL} excludedPresetNames={['👎 Negativity', '㊗️ Chinese']} />
    </div>
  );
}
