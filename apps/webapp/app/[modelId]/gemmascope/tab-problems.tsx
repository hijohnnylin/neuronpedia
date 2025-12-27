import { Home, Rocket } from 'lucide-react';
import { useEffect, useRef } from 'react';

export default function TabProblems({ tabUpdater }: { tabUpdater: (tab: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);
  return (
    <div className="relative mt-0 flex h-full w-full max-w-screen-xl flex-col items-center justify-start bg-white pb-24 pt-1">
      <div ref={ref} className="pt-20 sm:pt-0" />

      <div className="mb-6 mt-5 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 sm:flex-row sm:px-7">
        <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
          💡 Purpose
        </span>

        <div className="flex w-full flex-col items-start justify-start text-left text-sm font-medium text-slate-500">
          <div className="leading-normal">
            Interpretability has many unsolved problems. If{' '}
            {`you'd like to try your hand at solving one of them, consider the following starting points.`} Once{' '}
            {`you're ready to share or collaborate, join us in the `} Open Source Mechanistic Interpretability{' '}
            <a
              href="https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3m2fulfeu-0LnVnF8yCrKJYQvWLuCQaQ"
              target="_blank"
              rel="noreferrer"
              className="text-gBlue"
            >
              Slack
            </a>{' '}
            under channel #neuronpedia. You can also engage with communities focused on interpretability, such as{' '}
            <a href="https://lesswrong.com" target="_blank" rel="noreferrer" className="text-gBlue">
              LessWrong
            </a>
            .
          </div>
        </div>
      </div>

      <div className="mb-5 flex w-full flex-row items-center justify-start px-2 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 pb-24 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            ❓ Problems
          </span>
          <div className="flex w-full flex-col items-start justify-start gap-y-1 text-sm font-medium text-slate-500">
            <div className="mt-0 font-bold text-slate-600">Understanding Sparse Autoencoders</div>
            <ul className="list-inside list-disc">
              <li>
                Exploring the structure and relationships between SAE features, as suggested in{' '}
                <a href="https://arxiv.org/abs/2407.14662" target="_blank" rel="noreferrer" className="text-gBlue">
                  Wattenberg et al
                </a>
              </li>
              <li>
                Comparisons of residual stream SAE features across layers, e.g., are there persistent features that one
                can {`"match up"`} across adjacent layers?
              </li>
              <li>
                Better understanding {`"feature splitting"`}, the phenomena where high-level features in a small SAE
                break apart into several, more fine-grained features in a wider SAE
              </li>
            </ul>
            <div className="mt-2 font-bold text-slate-600">
              Leveraging sparse autoencoders to improve performance on some objective measure/real-world task (and
              comparing to fair baselines!)
            </div>
            <ul className="list-inside list-disc">
              <li>Detecting or fixing jailbreaks</li>
              <li>
                Helping find new jailbreaks/red-teaming models (See{' '}
                <a href="https://arxiv.org/abs/2205.01663" target="_blank" rel="noreferrer" className="text-gBlue">
                  Ziegler et al
                </a>
                )
              </li>
              <li>
                Comparing{' '}
                <a href="https://arxiv.org/abs/2308.10248" target="_blank" rel="noreferrer" className="text-gBlue">
                  steering vectors
                </a>{' '}
                to{' '}
                <a
                  href="https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Activation_Steering_with_SAEs"
                  target="_blank"
                  rel="noreferrer"
                  className="text-gBlue"
                >
                  SAE feature steering
                </a>{' '}
                or{' '}
                <a
                  href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"
                  target="_blank"
                  rel="noreferrer"
                  className="text-gBlue"
                >
                  clamping
                </a>
              </li>
            </ul>
            <div className="mt-2 font-bold text-slate-600">Red-teaming sparse autoencoders</div>
            <ul className="list-inside list-disc">
              <li>Do they really find the {`"true"`} concepts in a model?</li>
              <li>
                How robust are claims about the interpretability of SAE features? (See{' '}
                <a href="https://arxiv.org/abs/2309.10312" target="_blank" rel="noreferrer" className="text-gBlue">
                  Huang et al
                </a>
                )
              </li>
              <li>
                Can we find computable, quantitative measures that are a useful proxy for how {`"interpretable"`} humans
                think a feature vector is? (See{' '}
                <a
                  href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html"
                  target="_blank"
                  rel="noreferrer"
                  className="text-gBlue"
                >
                  Bills et al
                </a>
                )
              </li>
              <li>
                {`Can we find "dark matter" of truly non-linear features (i.e.
                that don't lie in a low rank subspace)?`}
                <ul className="ml-4 list-inside list-disc">
                  <li>
                    Note that this is different to{' '}
                    <a href="https://arxiv.org/abs/2405.14860" target="_blank" rel="noreferrer" className="text-gBlue">
                      Engels et al
                    </a>
                    , which find circular features lying in an irreducible rank 2 subspace
                  </li>
                </ul>
              </li>
              <li>
                Do they learn compositions of independent features to improve sparsity (
                <a
                  href="https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy"
                  target="_blank"
                  rel="noreferrer"
                  className="text-gBlue"
                >
                  as shown to happen in toy models
                </a>
                ), and can we fix this?
              </li>
            </ul>
            <div className="mt-2 font-bold text-slate-600">
              Scalable circuit analysis: What interesting circuits can we find in these models?
            </div>
            <ul className="list-inside list-disc">
              <li>
                {`What's the learned algorithm for addition in Gemma 2 2B?`} (See{' '}
                <a href="https://arxiv.org/abs/2305.15054" target="_blank" rel="noreferrer" className="text-gBlue">
                  Stolfo et al
                </a>
                )
              </li>
              <li>
                How can we practically extend the SAE feature circuit finding algorithm in{' '}
                <a href="https://arxiv.org/abs/2403.19647" target="_blank" rel="noreferrer" className="text-gBlue">
                  Marks et al
                </a>{' '}
                to larger models?
              </li>
              <li>
                Can we use SAE-like techniques like{' '}
                <a href="https://arxiv.org/abs/2406.11944v1" target="_blank" rel="noreferrer" className="text-gBlue">
                  MLP transcoders
                </a>{' '}
                to find input independent, weights-based circuits?
              </li>
            </ul>
            <div className="mt-2 font-bold text-slate-600">
              Using sparse autoencoders as a tool to answer existing questions in interpretability
            </div>
            <ul className="list-inside list-disc">
              <li>
                {`What does finetuning do to a model's internals?`} (See{' '}
                <a href="https://arxiv.org/abs/2407.10264" target="_blank" rel="noreferrer" className="text-gBlue">
                  Jain et al
                </a>
                )
              </li>
              <li>What is actually going on when a model uses chain of thought?</li>
              <li>
                Is in-context learning true learning, or just promoting existing circuits? (See{' '}
                <a href="https://arxiv.org/abs/2310.15916" target="_blank" rel="noreferrer" className="text-gBlue">
                  Hendel et al
                </a>
                ,{' '}
                <a href="https://functions.baulab.info/" target="_blank" rel="noreferrer" className="text-gBlue">
                  Todd et al
                </a>
                )
              </li>
            </ul>
            <div className="mt-2 font-bold text-slate-600">Finding ways to improve sparse autoencoders</div>
            <ul className="list-inside list-disc">
              <li>
                Can they efficiently capture the circular features of{' '}
                <a href="https://arxiv.org/abs/2405.14860" target="_blank" rel="noreferrer" className="text-gBlue">
                  Engels et al
                </a>
                ?
              </li>
              <li>
                How can they deal efficiently with cross-layer superposition (a feature that is produced in
                superposition by neurons spread across layers)
              </li>
              <li>
                How much can SAEs be quantized without significant performance degradation? (Both for inference and
                training)
              </li>
            </ul>

            <div className="mt-5 flex flex-row gap-x-3">
              <button
                type="button"
                onClick={() => {
                  tabUpdater('learn');
                }}
                className="flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-gGreen bg-white px-5 py-2.5 text-sm font-medium text-gGreen shadow transition-all hover:scale-105 hover:bg-gGreen/20"
              >
                <Rocket className="h-5 w-5 text-gGreen" /> Do More
              </button>
              <button
                type="button"
                onClick={() => {
                  tabUpdater('main');
                }}
                className="flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-gBlue bg-white px-5 py-2 text-sm font-medium text-gBlue shadow transition-all hover:scale-105 hover:bg-gBlue/20"
              >
                <Home className="h-5 w-5 text-gBlue" /> Home
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
