import { SPARSITY_COLOR_MAX, SPARSITY_COLOR_MIN, SPARSITY_COLORS } from '@/components/provider/umap-provider';

export const SEARCH_NOT_MATCHED_COLOR = '#e3e3e3';

export function getLogSparsityColorFromValue(value: number): string {
  // Define the value range
  const minValue = SPARSITY_COLOR_MIN;
  const maxValue = SPARSITY_COLOR_MAX;

  // Normalize the value to a 0-1 scale based on the value range
  let normalizedValue = (value - minValue) / (maxValue - minValue);

  // Ensure the normalized value is clamped between 0 and 1
  normalizedValue = Math.min(Math.max(normalizedValue, 0), 1);

  // Scale the normalized value to the range of indices
  const index = Math.round(normalizedValue * (SPARSITY_COLORS.length - 1));

  return SPARSITY_COLORS[index];
}
