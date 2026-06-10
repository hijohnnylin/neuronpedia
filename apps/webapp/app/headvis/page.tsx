import { redirect } from 'next/navigation';

// /headvis is a stable entry point that lands on a preselected attention head with the head finder
// open, so users can immediately browse from there.
const PRESELECTED_MODEL_ID = 'qwen3.5-0.8b';
const PRESELECTED_LAYER_NUM = 4;
const PRESELECTED_HEAD_INDEX = 5;

export default function Page() {
  redirect(`/${PRESELECTED_MODEL_ID}/head/${PRESELECTED_LAYER_NUM}/${PRESELECTED_HEAD_INDEX}?headFinder=true`);
}
