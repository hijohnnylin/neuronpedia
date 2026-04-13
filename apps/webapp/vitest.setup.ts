// fixes canvas not found issues
import { vi } from 'vitest';
import 'vitest-canvas-mock';

// plotly issue with createObjectURL
window.URL.createObjectURL = vi.fn();
