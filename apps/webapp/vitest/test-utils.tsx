import { Providers } from '@/components/provider/providers';

import { render, RenderOptions } from '@testing-library/react';

function ProvidersWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Providers
      initialModels={{}}
      initialExplanationTypes={[]}
      initialExplanationModels={[]}
      initialReleases={[]}
      initialExplanationScoreTypes={[]}
      initialExplanationScoreModelTypes={[]}
    >
      {children}
    </Providers>
  );
}

const customRender = (ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) =>
  render(ui, { wrapper: ProvidersWrapper, ...options });

export * from '@testing-library/react';
export { customRender as render };
