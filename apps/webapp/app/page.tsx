import CustomTooltip from '@/components/custom-tooltip';
import JumpToSAE from '@/components/jump-to-sae';
import InferenceActivationAllProvider from '@/components/provider/inference-activation-all-provider';
import RandomFeatureLink from '@/components/random-feature-link';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { DEFAULT_MODELID, DEFAULT_SOURCE, DEMO_MODE, IS_LOCALHOST, NEXT_PUBLIC_URL } from '@/lib/env';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import {
  Blocks,
  BookOpenText,
  Computer,
  Folder,
  Github,
  Lightbulb,
  Map,
  Newspaper,
  Notebook,
  PictureInPicture,
  RocketIcon,
  Route,
  School,
  Scroll,
  Search,
  Slack,
  SmileIcon,
  Speech,
  SquareActivity,
  Wand,
  Youtube,
} from 'lucide-react';
import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import FeatureSelector from '../components/feature-selector/feature-selector';
import InferenceSearcher from '../components/inference-searcher/inference-searcher';
import { CAP_BLOG_URL, CAP_PAPER_URL } from './[modelId]/assistant-axis/shared';
import HomeModels from './home/home-models';
import HomeNewsletterSignup from './home/home-newsletter-signup';
import HomeReleases from './home/home-releases';

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export async function generateMetadata(): Promise<Metadata> {
  const description = 'Open Source Interpretability Platform';
  return {
    title: {
      template: '%s ｜ Neuronpedia',
      default: 'Neuronpedia',
    },
    metadataBase: new URL(NEXT_PUBLIC_URL),
    description,
    openGraph: {
      title: {
        template: '%s',
        default: 'Neuronpedia',
      },
      description,
      url: NEXT_PUBLIC_URL,
      siteName: 'Neuronpedia',
      locale: 'en_US',
      type: 'website',
    },
    manifest: '/site.webmanifest',
    icons: {
      icon: [{ url: '/favicon-32x32.png' }, new URL('/favicon-32x32.png', 'https://neuronpedia.org')],
      apple: [{ url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' }],
    },
  };
}

export default function Page() {
  return (
    <div className="flex w-full select-none flex-col items-center justify-center bg-slate-100 px-0 pt-8 sm:mt-0 sm:px-0">
      {IS_LOCALHOST && !DEMO_MODE && (
        <div className="mb-4 flex w-full max-w-screen-sm flex-col items-center justify-center gap-2 rounded-lg border bg-white px-8 py-4 shadow-sm">
          <div className="text-xs font-bold text-slate-400">You are running a local instance of Neuronpedia.</div>
          <div className="text-sm text-slate-700">Would you like to go to the Admin panel to import sources/SAEs?</div>
          <Link href="/admin">
            <Button className="gap-x-2">
              <Computer className="h-4 w-4" /> Admin
            </Button>
          </Link>
        </div>
      )}

      <div className="mb-5 mt-2 flex w-full max-w-screen-xl flex-col items-center justify-center gap-x-3 gap-y-5 px-3 sm:flex-row">
        <div className="flex basis-[55%] flex-col items-center justify-center text-center text-sm sm:text-base">
          <div className="text-lg font-medium text-slate-800 sm:text-[19px]">
            Neuronpedia is an{' '}
            <a
              href="https://github.com/hijohnnylin/neuronpedia#readme"
              className="transition-all hover:text-slate-900/70 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              open source
            </a>{' '}
            <CustomTooltip
              trigger={
                <span className="font-bold text-sky-700 transition-all hover:cursor-default hover:text-sky-600">
                  interpretability
                </span>
              }
            >
              The inner workings of modern AIs are a mystery. This is because AIs are language models that are grown,
              not designed. The science of understanding what happens inside AI is called interpretability.
            </CustomTooltip>{' '}
            platform.
          </div>
          <div className="mt-0 text-sm font-normal text-slate-600 sm:text-[14px]">
            Explore, visualize, and steer the internals of AI models.
          </div>
          <div className="mt-3 flex flex-row items-center justify-center gap-x-2.5 gap-y-2 sm:mt-2 sm:flex-row">
            <Link href="https://github.com/hijohnnylin/neuronpedia" target="_blank" rel="noreferrer">
              <Button
                variant="default"
                size="sm"
                className="flex w-[110px] max-w-[110px] flex-row gap-x-2 bg-slate-800 text-white transition-all hover:bg-slate-900"
              >
                <Github className="h-3.5 w-3.5" />
                <div className="flex flex-col gap-y-0.5">
                  <span className="text-xs leading-none">GitHub</span>
                </div>
              </Button>
            </Link>
            {/* <Link href="https://docs.neuronpedia.org" target="_blank" rel="noreferrer">
              <Button variant="default" size="sm" className="w-[165px] gap-x-2 bg-sky-600 text-white hover:bg-sky-700">
                <BookOpenText className="h-5 w-5" />
                <span>Get Started</span>
              </Button>
            </Link> */}
            <Link href="/gemma-scope">
              <Button
                variant="default"
                size="sm"
                className="flex w-[135px] max-w-[135px] flex-row gap-x-2 bg-sky-600 text-xs text-white transition-all hover:bg-sky-700"
              >
                <Lightbulb className="h-3.5 w-3.5" />
                <span>New to Interp?</span>
              </Button>
            </Link>
            <Link href="/explorer">
              <Button
                variant="default"
                size="sm"
                title="[BETA] Browse the latest tools, papers, replications, problems and more."
                className="hidden w-[135px] max-w-[135px] flex-row gap-x-2 bg-amber-500 text-xs text-white transition-all hover:bg-amber-600"
              >
                <Map className="h-3.5 w-3.5" />
                <span>Field Explorer</span>
              </Button>
            </Link>
          </div>
        </div>
        <HomeNewsletterSignup />
      </div>

      <div className="flex w-full flex-col items-center justify-center px-1.5 pb-10 sm:px-0">
        <div className="my-5 mb-1 w-full overflow-hidden rounded-3xl border border-slate-200 bg-white px-0 pb-3 pt-3 shadow sm:max-w-screen-xl">
          <div
            className="ml-4 w-36 rounded-full bg-emerald-600 py-[5px] text-center text-[10px] font-medium uppercase text-white"
            title="Fully vetted and stabilized projects and collaborations."
          >
            Featured Releases
          </div>

          <div className="flex flex-col items-center justify-center gap-x-4 gap-y-0 px-1 pt-2 sm:flex-row sm:gap-y-2 sm:px-4 sm:pt-0">
            <div className="relative z-0 mb-3 mt-1 flex w-full flex-1 flex-col items-center justify-center rounded-lg px-3 sm:mx-0 sm:mb-2 sm:mt-2 sm:h-[250px] sm:min-h-[250px] sm:px-0">
              <div className="relative z-0 mb-0 flex h-full w-full min-w-full max-w-screen-sm flex-1 flex-col items-center justify-center gap-x-8 gap-y-1 rounded-3xl bg-[#e5e4df] px-3 py-7 text-center text-slate-600 shadow-sm shadow-[#666663]/50 sm:gap-y-0 sm:px-5 sm:py-7">
                <div className="absolute left-0 top-0 cursor-help flex-row items-center justify-center rounded-br-3xl rounded-tl-3xl bg-[#666663]/20 px-7 py-[6px] text-[11px] font-semibold text-[#262625]/60">
                  Lu et al. 2026
                </div>
                <div className="content">
                  <div className="mt-2 text-[26px] font-bold sm:mt-3 sm:text-[32px]">
                    <span className="font-bold text-[#bf4d43]">Assistant Axis</span>
                  </div>
                  <div className="mt-1 text-[13.5px] font-medium leading-none text-[#666663] sm:mt-0">
                    Monitor and Stabilize the Character of an LLM
                  </div>

                  <div className="mt-4 flex flex-col items-center justify-center gap-y-3 sm:mt-6 sm:flex-row sm:gap-x-2.5">
                    <Link href="/llama3.3-70b-it/assistant-axis" className="">
                      <button
                        type="button"
                        className="h-16 min-h-16 w-32 min-w-32 transition-all hover:scale-105 sm:w-auto"
                      >
                        <div className="flex h-16 min-h-16 flex-row items-center justify-center gap-x-2 rounded-xl bg-[#CC785C] px-6 py-2 text-[#000000] shadow-sm shadow-[#666663]/60">
                          <SquareActivity className="h-7 w-7" style={{ transform: 'rotate(90deg)' }} />
                          <div className="text-[13.5px] font-semibold leading-snug">
                            Launch
                            <br />
                            Chat
                          </div>
                        </div>
                      </button>
                    </Link>
                    <Link href={CAP_BLOG_URL} className="">
                      <button
                        type="button"
                        className="h-14 min-h-14 w-28 min-w-28 transition-all hover:scale-105 sm:w-auto"
                      >
                        <div className="flex h-12 min-h-12 flex-row items-center justify-center gap-x-1.5 rounded-xl bg-[#D4A274] px-5 py-2 text-[#262625] shadow-sm shadow-[#666663]/60 sm:px-3">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            id="Anthropic-Icon--Streamline-Svg-Logos"
                            height="24"
                            width="24"
                            className="flex-shrink-0"
                          >
                            <desc>Anthropic Icon Streamline Icon: https://streamlinehq.com</desc>
                            <path
                              fill="#181818"
                              d="m13.788825 3.932 6.43325 16.136075h3.5279L17.316725 3.932H13.788825Z"
                              strokeWidth="0.25"
                            />
                            <path
                              fill="#181818"
                              d="m6.325375 13.682775 2.20125 -5.67065 2.201275 5.67065H6.325375ZM6.68225 3.932 0.25 20.068075h3.596525l1.3155 -3.3886h6.729425l1.315275 3.3886h3.59655L10.371 3.932H6.68225Z"
                              strokeWidth="0.25"
                            />
                          </svg>
                          <div className="text-[12px] font-semibold leading-tight">Read Post</div>
                        </div>
                      </button>
                    </Link>
                    <Link href={CAP_PAPER_URL} className="">
                      <button
                        type="button"
                        className="h-12 min-h-12 w-28 min-w-28 transition-all hover:scale-105 sm:w-auto"
                      >
                        <div className="flex h-12 min-h-12 flex-row items-center justify-center gap-x-1.5 rounded-xl bg-[#D4A274] px-5 py-2 text-[#262625] shadow-sm shadow-[#666663]/60 sm:px-3">
                          <Scroll className="h-5 w-5" />
                          <div className="text-[12px] font-semibold leading-tight">Read Paper</div>
                        </div>
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            </div>

            <div className="relative z-0 mb-3 mt-3 flex w-full flex-col items-center justify-center rounded-lg px-3 sm:mx-0 sm:mb-2 sm:h-[210px] sm:min-h-[210px] sm:w-[400px] sm:min-w-[400px] sm:px-0">
              <div className="relative z-0 mb-0 flex h-full w-full min-w-full max-w-screen-sm flex-1 flex-col items-center justify-center gap-x-8 gap-y-1 rounded-3xl bg-emerald-600/20 px-3 py-7 text-center text-emerald-700 shadow-sm shadow-emerald-700/40 sm:gap-y-0 sm:px-5 sm:py-7">
                <div className="absolute left-0 top-0 cursor-help flex-row items-center justify-center rounded-br-3xl rounded-tl-3xl bg-emerald-600/30 px-7 py-[6px] text-[11px] font-semibold text-emerald-700/80">
                  Multi-Org
                </div>
                <div className="mt-2 text-[24px] font-bold leading-snug sm:mt-5 sm:text-2xl">Circuit Tracer</div>
                <div className="mt-0 text-sm font-medium leading-tight sm:mt-0.5 sm:text-[13px]">
                  Trace the Internal Reasoning Steps of a Model
                </div>

                <div className="mt-3 flex w-full flex-col items-center justify-center gap-y-2 sm:mt-5 sm:flex-row sm:gap-x-2">
                  <a href="/gemma-2-2b/graph" className="flex-1 text-white">
                    <button
                      type="button"
                      className="h-16 min-h-16 w-full rounded-xl bg-emerald-800 shadow-sm shadow-emerald-700/40 transition-all hover:scale-105 hover:bg-emerald-900"
                    >
                      <div className="flex flex-row items-center justify-center px-5 py-2 font-bold leading-none">
                        <Route className="mr-2.5 h-6 w-6" />
                        <div className="text-[13.5px] font-semibold leading-snug text-white">
                          Launch
                          <br />
                          Tracer
                        </div>
                      </div>
                    </button>
                  </a>

                  <a
                    href="https://www.youtube.com/playlist?list=PL05yUGfKO5wP6S5_12z7LG30LZigRYx1e"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 text-white"
                  >
                    <button
                      type="button"
                      className="flex h-12 min-h-12 w-28 min-w-28 flex-row items-center justify-center rounded-xl bg-emerald-600 px-0 py-0 font-bold leading-none shadow-sm shadow-emerald-700/40 transition-all hover:scale-105 hover:bg-emerald-700"
                    >
                      <Youtube className="mr-2.5 h-6 w-6" />
                      <div className="text-[12px] font-bold leading-tight text-white">
                        Watch
                        <br />
                        Demo
                      </div>
                    </button>
                  </a>

                  <Link href="/graph/info" className="flex-1 text-white">
                    <button
                      type="button"
                      className="h-12 min-h-12 w-28 min-w-28 rounded-xl bg-emerald-600 shadow-sm shadow-emerald-700/40 transition-all hover:scale-105 hover:bg-emerald-700"
                    >
                      <div className="flex flex-row items-center justify-center px-0 py-0 font-bold leading-none">
                        <Newspaper className="mr-2 h-5 min-h-5 w-5 min-w-5" />
                        <div className="text-[12px] font-bold leading-tight text-white">Read Post</div>
                      </div>
                    </button>
                  </Link>
                </div>
              </div>
            </div>

            <div className="relative z-0 mb-3 mt-3 flex w-full flex-col items-center justify-center rounded-lg px-3 sm:mx-0 sm:mb-2 sm:h-[210px] sm:min-h-[210px] sm:w-[340px] sm:min-w-[340px] sm:px-0">
              <div className="relative z-0 mb-0 flex h-full w-full min-w-full max-w-screen-sm flex-1 flex-col items-center justify-center gap-x-8 gap-y-1 rounded-3xl bg-gBlue/10 px-3 py-7 text-center text-slate-600 shadow-sm shadow-gBlue/40 sm:gap-y-0 sm:px-5 sm:py-7">
                <div className="absolute left-0 top-0 cursor-help flex-row items-center justify-center rounded-br-3xl rounded-tl-3xl bg-gBlue/15 px-6 py-[6px] text-[11px] font-semibold text-gBlue/80">
                  Google Deepmind
                </div>
                <div className="content">
                  <div className="mt-2 text-[24px] font-bold sm:mt-1.5">
                    <span className="font-bold text-gBlue">Gemma Scope 2</span>
                  </div>
                  <div className="mt-1 text-[13px] font-medium leading-tight text-slate-500 sm:mb-1 sm:mt-0">
                    SAEs and Transcoders for Gemma 3
                  </div>

                  <div className="mt-4 flex flex-row items-center justify-center gap-3 gap-x-2 sm:mt-5">
                    <Link href="/gemma-scope-2" className="">
                      <button
                        type="button"
                        className="h-14 min-h-14 w-28 min-w-28 transition-all hover:scale-105"
                        title="Browse the Release on Neuronpedia"
                      >
                        <div className="flex h-14 min-h-14 flex-row items-center justify-center rounded-xl bg-gBlue px-3 py-2.5 text-white shadow">
                          <Folder className="mr-2 h-6 w-6" />
                          <div className="text-[12.5px] font-semibold leading-snug">
                            Browse
                            <br />
                            Data
                          </div>
                        </div>
                      </button>
                    </Link>

                    <a
                      href="https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className=""
                    >
                      <button
                        type="button"
                        className="h-12 max-h-12 min-h-12 w-12 min-w-12 max-w-12 transition-all hover:scale-105"
                        title="Read Google DeepMind Blog Post"
                      >
                        <div className="flex w-full flex-row items-center justify-center px-0 py-0 font-bold leading-none">
                          <div className="flex h-12 min-h-12 w-12 min-w-12 max-w-12 flex-row items-center justify-center rounded-xl bg-gYellow py-2.5 text-white shadow">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 24 24">
                              <path
                                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                                fill="#FFFFFF"
                              />
                              <path
                                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                                fill="#FFFFFF"
                              />
                              <path
                                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                                fill="#FFFFFF"
                              />
                              <path
                                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                                fill="#FFFFFF"
                              />
                              <path d="M1 1h22v22H1z" fill="none" />
                            </svg>
                          </div>
                        </div>
                      </button>
                    </a>
                    <a
                      href="https://huggingface.co/google/gemma-scope-2"
                      target="_blank"
                      rel="noopener noreferrer"
                      className=""
                      title="View Hugging Face Release"
                    >
                      <button
                        type="button"
                        className="h-12 max-h-12 min-h-12 w-12 min-w-12 max-w-12 transition-all hover:scale-105"
                      >
                        <div className="flex w-full flex-row items-center justify-center px-0 py-0 font-bold leading-none">
                          <div className="flex h-12 min-h-12 w-12 min-w-12 max-w-12 flex-row items-center justify-center rounded-xl bg-gRed py-2.5 text-white shadow">
                            <SmileIcon className="h-5 w-5" />
                          </div>
                        </div>
                      </button>
                    </a>

                    <a
                      href="https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r"
                      target="_blank"
                      rel="noopener noreferrer"
                      className=""
                    >
                      <button
                        type="button"
                        className="h-12 max-h-12 min-h-12 w-12 min-w-12 max-w-12 transition-all hover:scale-105"
                        title="Open Colab Tutorial Notebook"
                      >
                        <div className="flex w-full flex-row items-center justify-center px-0 py-0 font-bold leading-none">
                          <div className="flex h-12 min-h-12 w-12 min-w-12 max-w-12 flex-row items-center justify-center rounded-xl bg-gGreen py-2.5 text-white shadow">
                            <Notebook className="h-5 w-5" />
                          </div>
                        </div>
                      </button>
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid w-full grid-cols-2 items-center justify-center gap-x-12 gap-y-5 bg-white px-5 py-5 md:flex md:h-[95px] md:min-h-[95px] md:grid-cols-3 md:flex-row md:py-0">
        <a
          href="https://www.technologyreview.com/2024/11/14/1106871/google-deepmind-has-a-new-way-to-look-inside-an-ais-mind/"
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/mit.png"
            className="h-[40px] opacity-70 grayscale hover:opacity-100 hover:grayscale-0"
            alt="MIT Technology Review"
          />
        </a>
        <a
          href="https://www.anthropic.com/research/open-source-circuit-tracing"
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/anthropic.png"
            className="h-[17px] opacity-55 grayscale hover:opacity-100 hover:grayscale-0"
            alt="Anthropic"
          />
        </a>
        <a
          href={`${NEXT_PUBLIC_URL}/gemma-scope`}
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/deepmind.png"
            className="h-[30px] opacity-80 grayscale hover:opacity-100 hover:grayscale-0"
            alt="Google DeepMind"
          />
        </a>
        <a
          href="https://venturebeat.com/ai/stop-guessing-why-your-llms-break-anthropics-new-tool-shows-you-exactly-what-goes-wrong?utm_source=chatgpt.com"
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/vb.png"
            className="h-[50px] opacity-40 grayscale hover:opacity-100 hover:grayscale-0"
            alt="VentureBeat"
          />
        </a>
        <a
          href={`${NEXT_PUBLIC_URL}/llama-scope`}
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/fudan.jpg"
            className="h-[70px] opacity-70 grayscale hover:opacity-100 hover:grayscale-0"
            alt="OpenMOSS, Fudan University"
          />
        </a>
        <a
          href={`${NEXT_PUBLIC_URL}/llama3.1-8b-eleuther_gp`}
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/eleutherai2.png"
            className="h-[35px] opacity-40 grayscale hover:opacity-100 hover:grayscale-0"
            alt="EleutherAI"
          />
        </a>
        <a
          href={`${NEXT_PUBLIC_URL}/gpt2sm-apollojt`}
          target="_blank"
          className="flex flex-row items-center justify-center"
          rel="noreferrer"
        >
          <img
            src="/usedby/apolloresearch.png"
            className="hidden h-[35px] opacity-70 grayscale hover:opacity-100 hover:grayscale-0 sm:block"
            alt="Apollo Research"
          />
        </a>
        {/* <a href="#mats" className="flex flex-row items-center justify-center" rel="noreferrer" aria-label="MATS">
          <img
            src="/usedby/mats.png"
            className="h-[35px] opacity-60 grayscale hover:opacity-100 hover:grayscale-0"
            alt="MATS"
          />
        </a> */}
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center bg-sky-100 py-12 sm:py-16 sm:pt-14">
        <div className="flex max-w-screen-xl flex-1 flex-col items-center gap-x-5 rounded-xl px-2 sm:px-0">
          <div className="flex flex-col text-center">
            <div className="text-3xl font-black text-sky-800">Explore</div>
            <div className="mt-3 text-[15px] font-medium leading-relaxed text-sky-800">
              Browse five+ terabytes of activations, explanations, and metadata. <br className="hidden sm:block" />
              Neuronpedia supports probes,{' '}
              <a href="https://docs.neuronpedia.org/features" className="text-sky-600 underline" target="_blank">
                latents/features
              </a>
              , custom vectors,{' '}
              <a href="/axbench" className="text-sky-600 underline" target="_blank">
                concepts
              </a>
              , and more.
            </div>
          </div>
          <div className="flex w-full flex-1 flex-col gap-x-3 gap-y-3 pt-6 sm:flex-row">
            <Card className="flex flex-1 flex-col gap-x-3 bg-white">
              <CardHeader className="pb-3">
                <CardTitle className="flex flex-row gap-x-2 text-slate-800">
                  <div>Releases</div>
                  <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4" />}>
                    <div className="flex flex-col">
                      A {`"release"`} is the data (activations, explanations, vectors, etc) associated with a specific
                      paper or post. Each release can contain data for multiple models, layers, and {`"sources"/SAEs`}.
                      Releases are the broadest grouping of data on Neuronpedia.
                    </div>
                  </CustomTooltip>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-1 flex-row gap-x-3">
                <HomeReleases />
              </CardContent>
            </Card>
            <Card className="flex flex-1 flex-col gap-x-3 bg-white sm:max-w-[360px]">
              <CardHeader className="pb-3">
                <CardTitle className="flex flex-row gap-x-2 text-slate-800">
                  <div>Models</div>
                  <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4" />}>
                    <div className="flex flex-col">
                      Choose a model to view its releases and all associated data with it, including sources,
                      activations, explanations, and more.{' '}
                      {`You'll also be able to directly experiment with the model with tools such as steering or activation testing.`}
                    </div>
                  </CustomTooltip>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-1 flex-row gap-x-3">
                <HomeModels />
              </CardContent>
            </Card>

            <Card className="flex flex-1 flex-col gap-x-3 bg-white sm:max-w-[380px]">
              <CardHeader className="pb-3">
                <CardTitle className="flex flex-row gap-x-2 text-slate-800">
                  <div>Jump To</div>
                  <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4" />}>
                    <div className="flex flex-col">
                      A source is a group of latents/features/vectors/concepts associated with a specific model. For
                      example, the gemma-2-2b@20-gemmascope-res-16k source contains 16,384 latents from Gemma Scope
                      associated with the 20th layer for the residual stream hook. Sources are not always SAE
                      features/latents - for example, the AxBench sources are {`"concepts"`}.
                    </div>
                  </CustomTooltip>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-1 flex-col items-start justify-start gap-x-3 pl-10">
                <JumpToSAE modelId={DEFAULT_MODELID || ''} layer={DEFAULT_SOURCE || ''} modelOnSeparateRow />
                <div className="mt-4 flex w-full cursor-pointer flex-col items-start justify-start border-t border-b-slate-100 pt-4 text-sm font-medium text-sky-700 outline-none">
                  <div className="text-[10px] font-medium uppercase text-slate-500">Jump to Feature</div>
                  <FeatureSelector
                    showModel
                    openInNewTab={false}
                    defaultModelId={DEFAULT_MODELID || ''}
                    defaultSourceSet={getSourceSetNameFromSource(DEFAULT_SOURCE || '')}
                    defaultIndex="0"
                    filterToPublic
                    modelOnSeparateRow
                    autoFocus={false}
                  />
                </div>
                {DEFAULT_MODELID && DEFAULT_SOURCE && (
                  <div className="mt-4 flex w-full flex-col border-t pt-4">
                    <div className="mb-1 font-sans text-[9px] font-medium uppercase text-slate-500">Jump to Random</div>
                    <RandomFeatureLink modelId={DEFAULT_MODELID || ''} source={DEFAULT_SOURCE || ''} />
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center gap-x-3 gap-y-12 bg-slate-50 px-2 py-12 sm:px-8 sm:py-16">
        <div className="flex max-w-screen-xl flex-1 flex-col items-center justify-center gap-x-8 gap-y-8 rounded-xl px-2 sm:flex-row sm:px-0 sm:pb-0">
          <div className="flex flex-col sm:basis-1/3">
            <div className="text-3xl font-black text-slate-800">Graph</div>
            <div className="mt-3 text-[15px] font-medium text-slate-700">
              Visualize and trace the internal reasoning steps of a model with custom prompts, pioneered by Anthropic
              {`'`}s{' '}
              <a
                href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html"
                target="_blank"
                rel="noreferrer"
                className="text-sky-600 underline"
              >
                circuit tracing
              </a>{' '}
              papers.
            </div>
            <div className="mt-3 flex flex-col justify-center gap-x-2 gap-y-1.5 sm:justify-start">
              <Link href="https://www.neuronpedia.org/gemma-2-2b/graph" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <RocketIcon className="h-5 w-5" />
                  <span>Try It: Circuit Tracer</span>
                </Button>
              </Link>
              <Link
                href="https://www.youtube.com/playlist?list=PL05yUGfKO5wP6S5_12z7LG30LZigRYx1e"
                target="_blank"
                rel="noreferrer"
              >
                <Button variant="default" size="lg" className="gap-x-2">
                  <Youtube className="h-5 w-5" />
                  <span>YouTube: Guided Demo</span>
                </Button>
              </Link>
              <Link href="https://neuronpedia.org/graph/info" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <Scroll className="h-5 w-5" />
                  <span>Post: Research Landscape</span>
                </Button>
              </Link>
            </div>
          </div>
          <a
            href="https://www.neuronpedia.org/gemma-2-2b/graph"
            target="_blank"
            rel="noreferrer"
            className="w-full flex-1 overflow-hidden rounded-xl border border-slate-200 bg-white p-2 pt-2 shadow transition-all duration-300 hover:ring-4 hover:ring-blue-400 hover:ring-opacity-50 sm:flex-initial sm:basis-2/3"
          >
            <Image
              src="https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets/blog/tracer.png"
              alt="Attribution graph example with Dallas gemma-2-2b"
              className="rounded-md"
              width={1736}
              height={998}
            />
          </a>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center gap-x-3 gap-y-12 bg-sky-100 px-2 py-12 sm:px-8 sm:py-16">
        <div className="flex max-w-screen-xl flex-1 flex-col items-center justify-center gap-x-8 gap-y-8 rounded-xl px-2 sm:flex-row sm:px-0 sm:pb-0">
          <div className="flex flex-col sm:basis-1/3">
            <div className="text-3xl font-black text-sky-800">Steer</div>
            <div className="mt-3 text-[15px] font-medium text-sky-700">
              Modify model behavior by steering its activations using latents or custom vectors. Steering supports
              instruct (chat) and reasoning models, and has fully customizable temperature, strength, seed, etc.
            </div>
            <div className="mt-3 flex flex-row justify-center gap-x-2 sm:justify-start">
              <Link
                href="https://www.neuronpedia.org/gemma-2-9b-it/steer?saved=cm7cp63af00jx1q952neqg6e5"
                target="_blank"
                rel="noreferrer"
              >
                <Button variant="default" size="lg" className="gap-x-2">
                  <Wand className="h-5 w-5" />
                  <span>Try It: Gemma 2 - Cat Steering</span>
                </Button>
              </Link>
            </div>
          </div>
          <a
            href="https://www.neuronpedia.org/gemma-2-9b-it/steer?saved=cm7cp63af00jx1q952neqg6e5"
            target="_blank"
            rel="noreferrer"
            className="w-full flex-1 overflow-hidden rounded-xl border border-slate-200 bg-white p-2 pt-2 shadow transition-all duration-300 hover:ring-4 hover:ring-blue-400 hover:ring-opacity-50 sm:flex-initial sm:basis-2/3"
          >
            <Image
              src="/steering-example.png"
              alt="Steering example with a cat feature"
              className="rounded-md"
              width={1736}
              height={998}
            />
          </a>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center gap-x-3 gap-y-12 bg-slate-50 px-2 py-12 sm:px-8 sm:py-16">
        <div className="flex max-w-screen-xl flex-1 flex-col items-center gap-x-8 gap-y-8 rounded-xl px-2 sm:flex-row sm:px-0 sm:pb-0">
          <div className="flex flex-col sm:basis-1/3">
            <div className="text-3xl font-black text-slate-800">Search</div>
            <div className="mt-3 text-[15px] font-medium text-slate-700">
              Search over 50,000,000 latents/vectors, either by semantic similarity to explanation text, or by running
              custom text via inference through a model to find top matches.{' '}
            </div>
            <div className="mt-3 flex flex-col justify-center gap-x-2 gap-y-1.5 sm:justify-start">
              <Link href="/search-explanations" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <Search className="h-5 w-5" />
                  <span>Try It: Search by Explanation</span>
                </Button>
              </Link>
              <Link href="https://docs.neuronpedia.org/search" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <BookOpenText className="h-5 w-5" />
                  <span>Docs: Search via Inference</span>
                </Button>
              </Link>
            </div>
          </div>
          <div className="w-full flex-1 sm:flex-initial sm:basis-2/3">
            <Card className="flex flex-1 flex-col gap-x-3 bg-white">
              <CardHeader className="pb-3">
                <CardTitle className="flex flex-row gap-x-2 text-slate-800">
                  <div>Search via Inference</div>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <InferenceActivationAllProvider>
                  <InferenceSearcher showSourceSets />
                </InferenceActivationAllProvider>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center gap-x-3 gap-y-12 bg-sky-100 px-2 py-12 sm:px-8 sm:py-16">
        <div className="flex max-w-screen-xl flex-1 flex-col items-center justify-center gap-x-8 gap-y-8 rounded-xl px-2 sm:flex-row sm:px-0 sm:pb-0">
          <div className="flex flex-col sm:basis-1/3">
            <div className="text-3xl font-black text-sky-800">API + Libraries</div>
            <div className="mt-3 text-[15px] font-medium text-sky-700">
              Neuronpedia hosts the {`world's first interpretability API (March 2024)`} - and all functionality is
              available by API or Python/TypeScript libraries. Most endpoints have an OpenAPI spec and interactive docs.
            </div>
            <div className="mt-3 flex flex-row justify-center gap-x-2 sm:justify-start">
              <Link href="/api-doc" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <Blocks className="h-5 w-5" />
                  <span>API Playground</span>
                </Button>
              </Link>
            </div>
          </div>
          <a
            href="/api-doc"
            target="_blank"
            rel="noreferrer"
            className="w-full flex-1 overflow-hidden rounded-xl border border-slate-200 bg-white p-2 pt-2 shadow transition-all duration-300 hover:ring-4 hover:ring-blue-400 hover:ring-opacity-50 sm:flex-initial sm:basis-2/3"
          >
            <Image
              src="/search-screenshot.png"
              alt="Steering example with a cat feature"
              className="rounded-md"
              width={1726}
              height={1000}
            />
          </a>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center gap-x-3 gap-y-12 bg-slate-50 px-2 py-12 sm:px-8 sm:py-16">
        <div className="flex w-full max-w-screen-xl flex-1 flex-col items-center gap-x-8 gap-y-8 rounded-xl px-2 sm:flex-row sm:px-0 sm:pb-0">
          <div className="flex flex-1 flex-col sm:basis-1/3">
            <div className="text-3xl font-black text-slate-800">Inspect</div>
            <div className="mt-3 text-[15px] font-medium text-slate-700">
              Go in depth on each probe/latent/feature with top activations, top logits, activation density, and live
              inference testing. All dashboards have unique links, can be compiled into sharable lists, and supports
              IFrame embedding, as demonstrated here.{' '}
            </div>
            <div className="mt-2 flex flex-row justify-center gap-x-2 sm:justify-start">
              <Link href="https://docs.neuronpedia.org/lists" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <BookOpenText className="h-5 w-5" />
                  <span>Docs: Lists</span>
                </Button>
              </Link>
              <Link href="https://docs.neuronpedia.org/embed-iframe" target="_blank" rel="noreferrer">
                <Button variant="default" size="lg" className="gap-x-2">
                  <PictureInPicture className="h-5 w-5" />
                  <span>Docs: Embed</span>
                </Button>
              </Link>
            </div>
          </div>
          <div className="flex w-full flex-1 sm:basis-2/3 sm:px-0">
            <iframe
              title="Jedi Feature"
              src="https://neuronpedia.org/gpt2-small/0-res-jb/14057?embed=true&embedexplanation=true&embedplots=true"
              style={{ width: '100%', height: '540px' }}
              scrolling="no"
              className="overflow-hidden rounded-lg border"
            />
          </div>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center bg-slate-100 sm:flex-row sm:px-10">
        <div className="flex w-full max-w-screen-xl flex-col items-center justify-center gap-y-5 px-2 py-12 sm:flex-row sm:px-8 sm:py-16">
          <div className="flex flex-1 flex-col items-center justify-center gap-x-5 bg-slate-100">
            <div className="text-2xl font-black text-slate-700">Who We Are</div>
            <div className="mt-3 text-base font-medium leading-normal text-slate-700">
              Neuronpedia was created by{' '}
              <a href="https://johnnylin.co" target="_blank" rel="noreferrer" className="text-sky-600">
                Johnny Lin
              </a>
              , an ex-Apple engineer who previously founded a privacy startup. Neuronpedia is supported by Decode
              Research, Open Philanthropy, the Long Term Future Fund, AISTOF, Anthropic, Manifund, and others.
            </div>
          </div>

          <div className="flex flex-1 flex-col items-center gap-x-5 bg-slate-100 text-left">
            <div className="text-2xl font-black text-slate-700">Get Involved</div>
            <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4 text-base font-medium leading-snug text-amber-100 sm:mt-3 sm:gap-x-3 sm:gap-y-3">
              <a
                href="https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3m2fulfeu-0LnVnF8yCrKJYQvWLuCQaQ"
                target="_blank"
                rel="noreferrer"
                className=""
              >
                <Button className="h-14 w-[170px] gap-x-2 sm:w-[200px]" size="lg">
                  <Slack className="h-5 w-5" />
                  <span className="flex-1">Community</span>
                </Button>
              </a>
              <a href="https://github.com/hijohnnylin/neuronpedia" target="_blank" rel="noreferrer">
                <Button className="h-14 w-[170px] gap-x-2 sm:w-[200px]" size="lg">
                  <Github className="h-5 w-5" />
                  <span className="flex-1">GitHub</span>
                </Button>
              </a>
              <a href="/contact" target="_blank" rel="noreferrer">
                <Button className="h-14 w-[170px] gap-x-2 sm:w-[200px]" size="lg">
                  <Speech className="h-5 w-5" />
                  <span className="flex-1">Contact</span>
                </Button>
              </a>
              <a href="https://arena.education" target="_blank" rel="noreferrer">
                <Button className="h-14 w-[170px] gap-x-2 sm:w-[200px]" size="lg">
                  <School className="h-5 w-5" />
                  <span className="flex-1">Upskill</span>
                </Button>
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="flex w-full flex-1 flex-col items-center justify-center bg-white py-16">
        <div className="mt-0 text-2xl font-black text-slate-700">Citation</div>
        <div className="mt-4 flex max-w-[320px] flex-row items-start justify-start overflow-x-scroll text-[10px] font-medium leading-normal text-slate-700 sm:max-w-[100%] sm:text-sm">
          <pre className="flex cursor-text select-text flex-row justify-start whitespace-pre-wrap text-left font-mono">
            {`@misc{neuronpedia,
    title = {Neuronpedia: Interactive Reference and Tooling for Analyzing Neural Networks},
    year = {2023},
    note = {Software available from neuronpedia.org},
    url = {https://www.neuronpedia.org},
    author = {Lin, Johnny}
}`}
          </pre>
        </div>
      </div>
    </div>
  );
}
