import { redirect } from 'next/navigation';

// /headvis is a stable entry point that lands on a preselected attention head with the head finder
// open, so users can immediately browse from there.
const PRESELECTED_MODEL_ID = 'qwen3.6-27b';
const PRESELECTED_LAYER_NUM = 15;
const PRESELECTED_HEAD_INDEX = 22;

export default function Page() {
  redirect(`/${PRESELECTED_MODEL_ID}/head/${PRESELECTED_LAYER_NUM}/${PRESELECTED_HEAD_INDEX}?headFinder=true`);
}
