<p align="center">
  <a href="https://github.com/hijohnnylin/neuronpedia">
    <img src="https://github.com/user-attachments/assets/9bcea0bf-4fa9-401d-bb7a-d031a4d12636" alt="Splash GIF"/>
  </a>

<h3 align="center"><a href="https://neuronpedia.org">neuronpedia.org 🧠🔍</a></h3>

  <p align="center">
    open source interpretability platform
    <br />
    <sub>
    <strong>api · steering · activations · autointerp · scoring · inference · search · filter · dashboards · benchmarks · cossim · umap · embeds · probes · saes · lists · exports · uploads</strong>
    </sub>
  </p>
</p>

<p align="center" style="color: #cccccc;">
  <a href="https://github.com/hijohnnylin/neuronpedia/blob/main/LICENSE"><img height="20px" src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="MIT"></a>
  <a href="https://status.neuronpedia.org"><img height="20px" src="https://uptime.betterstack.com/status-badges/v2/monitor/1roih.svg" alt="Uptime"></a>
  <a href="https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-2o756ku1c-_yKBeUQMVfS_p_qcK6QLeA"><img height="20px" src="https://img.shields.io/badge/slack-purple?logo=slack&logoColor=white" alt="Slack"></a>
  <a href="mailto:johnny@neuronpedia.org"><img height="20px" src="https://img.shields.io/badge/contact-blue.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiPjwvZz48ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjwvZz48ZyBpZD0iU1ZHUmVwb19pY29uQ2FycmllciI+IDxwYXRoIGQ9Ik00IDcuMDAwMDVMMTAuMiAxMS42NUMxMS4yNjY3IDEyLjQ1IDEyLjczMzMgMTIuNDUgMTMuOCAxMS42NUwyMCA3IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48L3BhdGg+IDxyZWN0IHg9IjMiIHk9IjUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxNCIgcng9IjIiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiPjwvcmVjdD4gPC9nPjwvc3ZnPg==" alt="Email"></a>
  <a href="https://neuronpedia.org/blog"><img height="20px" src="https://img.shields.io/badge/blog-10b981.svg" alt="blog"></a>
  <a href="https://neuronpedia.org"><img height="20px" src="https://img.shields.io/badge/website-gray.svg" alt="website"></a>
  <a href="https://www.every.org/decode-research"><img height="20px" src="https://img.shields.io/badge/sponsor-green.svg" alt="Sponsor"></a>

</p>

- [about neuronpedia](#about-neuronpedia)
- [instant start - vercel deploy](#instant-start---vercel-deploy)
- [quick start - local webapp + demo environment](#quick-start---local-webapp--demo-environment)
- [setting up your local environment](#setting-up-your-local-environment)
  - ["i want to use a local database / import more neuronpedia data"](#i-want-to-use-a-local-database--import-more-neuronpedia-data)
  - ["i want to do webapp (frontend + api) development"](#i-want-to-do-webapp-frontend--api-development)
  - ["i want to run/develop inference locally"](#i-want-to-rundevelop-inference-locally)
  - ['i want to run/develop autointerp locally\`](#i-want-to-rundevelop-autointerp-locally)
  - ['i want to do high volume autointerp explanations'](#i-want-to-do-high-volume-autointerp-explanations)
  - ['i want to generate my own dashboards/data and add it to neuronpedia'](#i-want-to-generate-my-own-dashboardsdata-and-add-it-to-neuronpedia)
- [architecture](#architecture)
  - [requirements](#requirements)
  - [services](#services)
    - [services are standalone apps](#services-are-standalone-apps)
    - [service-specific documentation](#service-specific-documentation)
  - [openapi schema](#openapi-schema)
  - [monorepo directory structure](#monorepo-directory-structure)
- [security](#security)
- [contact / support](#contact--support)
- [contributing](#contributing)
- [appendix](#appendix)
    - ['make' commands reference](#make-commands-reference)
    - [import data into your local database](#import-data-into-your-local-database)
    - [why an openai api key is needed for search explanations](#why-an-openai-api-key-is-needed-for-search-explanations)

<!-- # ultra-quick start: one-click deploy on vercel
TODO, after making repo public -->

# about neuronpedia

check out our [blog post](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source) about Neuronpedia, why we're open sourcing it, and other details. there's also a [tweet thread](https://x.com/neuronpedia/status/1906793456879775745) with quick demos.

**feature overview**

a diagram showing the main features of neuronpedia as of march 2025.
![neuronpedia-features](https://github.com/user-attachments/assets/13e07a93-e046-4e1c-b670-2d26d251d55d)

# instant start - vercel deploy

click the `Deploy` button to instantly deploy a custom neuronpedia. a [free vercel account](https://vercel.com/signup) is required.

<p align="left">
  <a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fhijohnnylin%2Fneuronpedia&env=NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY&envDescription=***Your%20Custom%20Website%20Name.%20For%20example%3A%20PuppyNeurons***&root-directory=apps/webapp&build-command=npx%20prisma%20generate%20%26%26%20npm%20run%20build%3Ademo&project-name=my-neuronpedia&repository-name=my-neuronpedia&demo-title=Neuronpedia&demo-description=Deploy%20your%20own%20custom%20Neuronpedia%20%F0%9F%9A%80%F0%9F%A7%A0%F0%9F%A7%90&demo-url=https%3A%2F%2Fneuronpedia.org">
    <img src="https://vercel.com/button" width="160" alt="Deploy with Vercel"/>
  </a>
</p>

here's how easy it is to deploy a "PuppyNeurons" fork of Neuronpedia:

https://github.com/user-attachments/assets/707deaed-02b4-442b-8c1f-bf44d193b9fa

# quick start - local webapp + demo environment

#### what this does

this sets up the webapp (frontend + api) locally, and connects to a public remote demo database and public inference servers

#### what you'll get

after following the quick start, you will be able to use neuronpedia for some sources/SAEs we have preloaded in `gpt2-small` and `gemma-2-2b/-it`.

> ⚠️ **warning:** since you are connecting to a public, read-only demo database, you will not be able to add new data immediately. you will need to follow [subsequent steps](#i-want-to-use-my-own-database--import-more-neuronpedia-data) to configure your own database that you can write to.

#### steps

1. install [docker desktop (UI)](https://docs.docker.com/desktop/) or [docker engine (no UI)](https://docs.docker.com/engine/), and launch it.
2. generate your local `.env`
   ```
   make init-env
   ```
3. build the webapp (this will take ~10 min the first time)
   ```
   make webapp-demo-build
   ```
4. bring up the webapp
   ```
   make webapp-demo-run
   ```
5. once everything is up, open [localhost:3000](http://localhost:3000) to load the home page.
6. your local instance is connected to the remote demo database and inference servers, with the following SAEs/sources data available:

| model                          | source/sae                            | comment                                        |
| ------------------------------ | ------------------------------------- | ---------------------------------------------- |
| `gpt2-small`                   | `res-jb`, all layers                  | a small starter SAE set                        |
| `gemma-2-2b` / `gemma-2-2b-it` | `gemmascope-res-16k`, all layers      | the SAEs used in the Gemma Scope demo          |
| `deepseek-r1-distill-llama-8b` | `llamascope-slimpj-res-32k`, layer 15 | SAE for a reasoning model, trained by OpenMOSS |

7. example things you can do (links work after `make webapp-demo-run`)

   i. steering - [steer gpt2-small on cats](http://localhost:3000/gpt2-small/steer?source=10-res-jb&index=16899&strength=40)

   ii. activation tests/search - [test activation for a gemma-2-2b feature](http://localhost:3000/gemma-2-2b/20-gemmascope-res-16k/502?defaulttesttext=what's%20the%20deal%20with%20airplane%20food%3F)

   iii. search by explanation, [if you configured](<(#why-an-openai-api-key-is-needed-for-search-explanations)>) an `OPENAI_API_KEY` - [search for parrots features](http://localhost:3000/search-explanations/?q=parrots)

   iv. browse dashboards - [a parrot feature](http://localhost:3000/gpt2-small/11-res-jb/23687)

   v. run the [gemma-scope demo](http://localhost:3000/gemma-scope#main)

8. now that we've set up a local webapp that's usable, this is a good time to quickly review neuronpedia's [simple architecture](#architecture) and its [individual services](#services), so that you can get a better understanding of what you'll set up later. then, keep going to [setting up your local environment](#setting-up-your-local-environment).

> 🔥 **pro-tip:** see all the available `make` commands by running `make help`

# setting up your local environment

once you've played around with the demo, you will start running into limitations, like having a limited number of models/SAEs to use, or not being able to generate new explanations. this is because the public demo database is read-only.

ideally, you will probably eventually want to do all of the sub-sections below, so you can have everything running locally. however, you may only be interested in specific parts of neuronpedia to start:

1. if you want to jump into developing webapp frontend or api with the demo environment, follow [webapp dev](#i-want-to-do-webapp-frontend--api-development)
2. if you want to start loading more sources/data and relying on your own local database, follow [local database](#i-want-to-use-a-local-database--import-more-neuronpedia-data)

> 🔥 **pro-tip:** neuronpedia is configured for AI agent development. here's an example using a [single prompt](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/experiments/steerify/README.md#claude-code-prompt) to build a custom app (Steerify) using Neuronpedia's inference server as a backend:

https://github.com/user-attachments/assets/bc82f88b-8155-4c1d-948a-ea5d987ae0f8

## "i want to use a local database / import more neuronpedia data"

#### what this does + what you'll get

relying on the demo environment means you are limited to read-only access to a specific set of SAEs. these steps show you how to configure and connect to your own local database. you can then download sources/SAEs of your choosing:

https://github.com/user-attachments/assets/d7fbb46e-8522-4f98-aa08-21c6529424af

> ⚠️ **warning:** your database will start out empty. you will need to use the admin panel to [import sources/data](#import-data-into-your-local-database) (activations, explanations, etc).

> ⚠️ **warning:** the local database environment does not have any inference servers connected, so you won't be able to do activation testing, steering, etc initially. you will need to [configure a local inference instance]().

#### steps

1. build the webapp
   ```
   make webapp-localhost-build
   ```
2. bring up the webapp
   ```
   make webapp-localhost-run
   ```
3. go to [localhost:3000](http://localhost:3000) to see your local webapp instance, which is now connected to your local database
4. see the `warnings` above for caveats, and `next steps` to finish setting up

#### next steps

1. [click here](#import-data-into-your-local-database) for how to import data into your local database (activations, explanations, etc), because your local database will be empty to start
2. [click here](#i-want-to-rundevelop-inference-locally) for how to bring up a local `inference` service for the model/source/SAE you're working with

## "i want to do webapp (frontend + api) development"

#### what this does

the webapp builds you've been doing so far are _production builds_, which are slow to build, and fast to run. since they are slow to build and don't have debug information, they are not ideal for development.

this subsection installs the development build on your local machine (not docker), then mounts the build inside your docker instance.

#### what you'll get

once you do this section, you'll be able to do local development and quickly see changes that are made, as well as see more informative debug/errors. if you are purely interested in doing frontend/api development for neuronpedia, you don't need to set up anything else!

#### steps

1. install [nodejs](https://nodejs.org) via [node version manager](https://github.com/nvm-sh/nvm)
   ```
   make install-nodejs
   ```
2. install the webapp's dependencies
   ```
   make webapp-localhost-install
   ```
3. run the development instance
   ```
   make webapp-localhost-dev
   ```
4. go to [localhost:3000](http://localhost:3000) to see your local webapp instance

#### doing local webapp development

- **auto-reload**: when you change any files in the `apps/webapp` subdirectory, the `localhost:3000` will automatically reload
- **install commands**: you do not need to run `make install-nodejs` again, and you only need to run `make webapp-localhost-install` if dependencies change

## "i want to run/develop inference locally"

#### what this does + what you'll get

once you start using a local environment, you won't be connected to the demo environment's inference instances. this subsection shows you how to run an inference instance locally so you can do things like steering, activation testing, etc on the sources/SAEs you've downloaded.

> ⚠️ **warning:** for the local environment, we only support running one inference server at a time. this is because you are unlikely to be running multiple models simultaneously on one machine, as they are memory and compute intensive.

#### steps

1. ensure you have [installed poetry](https://python-poetry.org/docs/#installation)
2. install the inference server's dependencies
   ```
   make inference-localhost-install
   ```
3. build the image, picking the correct command based on if the machine has CUDA or not:
   ```
   # CUDA
   make inference-localhost-build-gpu
   ```
   ```
   # no CUDA
   make inference-localhost-build
   ```
4. run the inference server, using the `MODEL_SOURCESET` argument to specify the `.env.inference.[model_sourceset]` file you're loading from. for this example, we will run `gpt2-small`, and load the `res-jb` sourceset/SAE set, which is configured in the `.env.inference.gpt2-small.res-jb` file. you can see the other [pre-loaded inference configs](#pre-loaded-inference-server-configurations) or [create your own config](#making-your-own-inference-server-configurations) as well.

   ```
   # CUDA
   make inference-localhost-dev-gpu \
        MODEL_SOURCESET=gpt2-small.res-jb

   # no CUDA
   make inference-localhost-dev \
        MODEL_SOURCESET=gpt2-small.res-jb
   ```

5. wait for it to load (first time will take longer). when you see `Initialized: True`, the local inference server is now ready on `localhost:5002`

#### using the inference server

to interact with the inference server, you have a few options - note that this will only work for the model / selected source you have loaded:

1.  load the webapp with the [local database setup](#i-want-to-use-a-local-database--import-more-neuronpedia-data), then using the model / selected source as you would normally do on neuronpedia.
2.  use the pre-generated inference python client at `packages/python/neuronpedia-inference-client` (set environment variable `INFERENCE_SERVER_SECRET` to `public`, or whatever it's set to in `.env.localhost` if you've changed it)
3.  use the openapi spec, located at `schemas/openapi/inference-server.yaml` to make calls with any client of your choice.
4.  [TODO #1](https://github.com/hijohnnylin/neuronpedia/issues/1): Use a documentation generator to make a simple tester-server that can be activated with `make doc-inference-localhost`

#### pre-loaded inference server configurations

we've provided some pre-loaded inference configs as examples of how to load a specific model and sourceset for inference. view them by running `make inference-list-configs`:

```
$ make inference-list-configs

Available Inference Configurations (.env.inference.*)
================================================

deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k
    Model: meta-llama/Llama-3.1-8B
    Source/SAE Sets: '["llamascope-slimpj-res-32k"]'
    make inference-localhost-dev MODEL_SOURCESET=deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k

gemma-2-2b-it.gemmascope-res-16k
    Model: gemma-2-2b-it
    Source/SAE Sets: '["gemmascope-res-16k"]'
    make inference-localhost-dev MODEL_SOURCESET=gemma-2-2b-it.gemmascope-res-16k

gpt2-small.res-jb
    Model: gpt2-small
    Source/SAE Sets: '["res-jb"]'
    make inference-localhost-dev MODEL_SOURCESET=gpt2-small.res-jb
```

#### making your own inference server configurations

look at the `.env.inference.*` files for examples on how to make these inference server configurations.

the `MODEL_ID` is the model id from the [transformerlens model table](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html) and each of `SAE_SETS` is the text after the layer number and hyphen in a neuronpedia source ID - for example, if you have a neuronpedia feature at url `http://neuronpedia.org/gpt2-small/0-res-jb/123`, the `0-res-jb` is the source ID, and the item in the `SAE_SETS` is `res-jb`. This example matches the `.env.inference.gpt2-small.res-jb` file exactly.

you can find neuronpedia source IDs in the saelens [pretrained saes yaml file](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml) or by clicking into models in the [neuronpedia datasets exports](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/) directory.

**using models not officially supported by transformerlens**
look at the `.env.inference.deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k` to see an example of how to load a model not officially supported by transformerlens. this is mostly for swapping in weights of a distilled/fine-tuned model.

**loading non-saelens sources/SAEs**

- [TODO #2](https://github.com/hijohnnylin/neuronpedia/issues/2) document how to load SAEs/sources that are not in saelens pretrained yaml

#### doing local inference development

- **schema-driven development**: to add new endpoints or change existing endpoints, you will need to start by updating the openapi schemas, then generating clients from that, then finally updating the actual inference and webapp code. for details on how to do this, see the [openapi readme: making changes to the inference server](schemas/README.md#making-changes-to-the-inference-server)
- **no auto-reload**: when you change any files in the `apps/inference` subdirectory, the inference server will _NOT_ automatically reload, because server reloads are slow: they reload the model and all sources/SAEs. if you want to enable autoreload, then append `AUTORELOAD=1` to the `make inference-localhost-dev` call, like so:
  ```
  make inference-localhost-dev \
       MODEL_SOURCESET=gpt2-small.res-jb \
       AUTORELOAD=1
  ```

## 'i want to run/develop autointerp locally`

this section is under construction.

- check out the [autointerp readme](apps/autointerp/README.md)
- [TODO](https://github.com/hijohnnylin/neuronpedia/issues/5) instructions for setting up autointerp server locally
- TODO - look at the `autointerp` service in [docker-compose.yaml](docker-compose.yaml)
- schema-driven development: [openapi readme: making changes to the autointerp server](schemas/README.md#making-changes-to-the-autointerp-server)

## 'i want to do high volume autointerp explanations'

this section is under construction.

- use EleutherAI's [Delphi library](https://github.com/EleutherAI/delphi)
- for OpenAI's autointerp, use [utils/neuronpedia_utils/batch-autointerp.py](utils/neuronpedia_utils/batch-autointerp.py)

## 'i want to generate my own dashboards/data and add it to neuronpedia'

this section is under construction.

[TODO: simplify generation + upload of data to neuronpedia](https://github.com/hijohnnylin/neuronpedia/issues/46)

[TODO: neuronpedia-utils should use poetry](https://github.com/hijohnnylin/neuronpedia/issues/43)

in this example, we will generate dashboards/data for an [SAELens](https://github.com/jbloomAus/SAELens)-compatible SAE, and upload it to our own Neuronpedia instance.

1. ensure you have [Poetry installed](https://python-poetry.org/docs/)
2. [upload](https://github.com/jbloomAus/SAELens/blob/main/tutorials/uploading_saes_to_huggingface.ipynb) your SAELens-compatible source/SAE to HuggingFace.
   > Example
   > ➡️ [https://huggingface.co/chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40](https://huggingface.co/chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40)
3. clone SAELens locally.
   ```
   git clone https://github.com/jbloomAus/SAELens.git
   ```
4. open your cloned SAELens and edit the file `sae_lens/pretrained_saes.yaml`. add a new entry at the bottom, based on the template below (see comments for how to fill it out):
   > Example
   > ➡️ [https://github.com/jbloomAus/SAELens/pull/455/files](https://github.com/jbloomAus/SAELens/pull/455/files)
   ```
   gemma-2-2b-res-matryoshka-dc:                 # a unique ID for your set of SAEs
     conversion_func: null                       # null if your SAE config is already compatible with SAELens
     links:                                      # optional links
       model: https://huggingface.co/google/gemma-2-2b
     model: gemma-2-2b                           # transformerlens model id - https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
     repo_id: chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40  # the huggingface repo path
     saes:
     - id: blocks.0.hook_resid_post                 # an id for this SAE
       path: standard/blocks.0.hook_resid_post      # the path in the repo_id to the SAE
       l0: 40.0
       neuronpedia: gemma-2-2b/0-matryoshka-res-dc  # what you expect the neuronpedia URI to be - neuronpedia.org/[this_slug]. should be [model_id]/[layer]-[identical_slug_for_this_sae_set]
     - id: blocks.1.hook_resid_post                 # more SAEs in this SAE set
       path: standard/blocks.1.hook_resid_post
       l0: 40.0
       neuronpedia: gemma-2-2b/1-matryoshka-res-dc  # note that this is identical to the entry above, except 1 instead of 0 for the layer
     - [...]
   ```
5. clone [SAEDashboard](https://github.com/jbloomAus/SAEDashboard.git) locally.
   ```
   git clone https://github.com/jbloomAus/SAEDashboard.git
   ```
6. configure your cloned `SAEDashboard` to use your cloned modified `SAELens`, instead of the one in production
   ```
   cd SAEDashboard                    # set directory
   poetry lock && poetry install      # install dependencies
   poetry remove sae-lens             # remove production dependency
   poetry add PATH/TO/CLONED/SAELENS  # set local dependency
   ```
7. generate dashboards for the SAE. this will take from 30 min to a few hours, depending on your hardware and size of model.

   ```
   cd SAEDashboard                    # set directory
   rm -rf cached_activations          # clear old cached data

   # start the generation. details for each argument (full details: https://github.com/jbloomAus/SAEDashboard/blob/main/sae_dashboard/neuronpedia/neuronpedia_runner_config.py)
   #     - sae-set = should match the unique ID for the set from pretrained_saes.yaml
   #     - sae-path = should match the id for the sae in from pretrained_saes.yaml
   #     - np-set-name = should match the [identical_slug_for_this_sae_set] for the sae.neuronpedia from pretrained_saes.yaml
   #     - dataset-path = the huggingface dataset to use for generating activations. usually you want to use the same dataset the model was trained on.
   #     - output-dir = the output directory of the dashboard data
   #     - n-prompts = number of activation texts to test from the dataset
   #     - n-tokens-in-prompt, n-features-per-batch, n-prompts-in-forward-pass = keep these at 128
   poetry run neuronpedia-runner \
        --sae-set="gemma-2-2b-res-matryoshka-dc" \
        --sae-path="blocks.12.hook_resid_post" \
        --np-set-name="matryoshka-res-dc" \
        --dataset-path="monology/pile-uncopyrighted" \
        --output-dir="neuronpedia_outputs/" \
        --sae_dtype="float32" \
        --model_dtype="bfloat16" \
        --sparsity-threshold=1 \
        --n-prompts=24576 \
        --n-tokens-in-prompt=128 \
        --n-features-per-batch=128 \
        --n-prompts-in-forward-pass=128
   ```

8. convert these dashboards for import into neuronpedia
   ```
   cd neuronpedia/utils/neuronpedia-utils          # get into this current repository's util directory
   python convert-saedashboard-to-neuronpedia.py   # start guided conversion script. follow the steps.
   ```
9. once dashboard files are generated for neuronpedia, upload these to the global Neuronpedia S3 bucket - currently you need to [contact us](mailto:johnny@neuronpedia.org) to do this.
10. from a localhost instance, [import your data](#i-want-to-use-a-local-database--import-more-neuronpedia-data)

# architecture

here's how the services/scripts connect in neuronpedia. it's easiest to read this diagram by starting at the image of the laptop ("User").

![architecture diagram](architecture.png)

## requirements

you can run neuronpedia on any cloud and on any modern OS. neuronpedia is designed to avoid vendor lock-in. these instructions were written for and tested on macos 15 (sequoia), so you may need to repurpose commands for windows/ubuntu/etc. at least 16GB ram is recommended.

## services

| name       | description                                                                                                                                                  | powered by                            |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| webapp     | serves the neuronpedia.org frontend and [the api](neuronpedia.org/api-doc)                                                                                   | [next.js](https://nextjs.org) / react |
| database   | stores features, activations, explanations, users, lists, etc                                                                                                | postgres                              |
| inference  | [support server] steering, activation testing, search via inference, topk, etc. a separate instance is required for each model you want to run inference on. | python / torch                        |
| autointerp | [support server] auto-interp explanations and scoring, using eleutherAI's [delphi](https://github.com/EleutherAI/delphi) (formerly `sae-auto-interp`)        | python                                |

### services are standalone apps

by design, each service can be run independently as a standalone app. this is to enable extensibility and forkability.

for example, if you like the neuronpedia webapp frontend but want to use a different API for inference, you can do that! just ensure your alternative inference server supports the `schema/openapi/inference-server.yaml` spec, and/or that you modify the neuronpedia calls to inference under `apps/webapp/lib/utils`.

### service-specific documentation

there are draft `README`s for each specific app/service under `apps/[service]`, but they are heavily WIP. you can also check out the `Dockerfile` under the same directory to build your own images.

## openapi schema

for services to communicate with each other in a typed and consistent way, we use openapi schemas. there are some exceptions - for example, streaming is not offically supported by the openapi spec. however, even in that case, we still try our best to define a schema and use it.

especially for inference and autointerp server development, it is critical to understand and use the instructions under the [openapi readme](schemas/README.md).

openapi schemas are located under `/schemas`. we use openapi generators to generate clients in both typescript and python.

## monorepo directory structure

`apps` - the three neuronpedia services: webapp, inference, and autointerp. most of the code is here.
`schemas` - the openapi schemas. to make changes to inference and autointerp endpoints, first make changes to their schemas - see details in the [openapi readme](schemas/README.md).
`packages` - clients generated from the `schemas` using generator tools. you will mostly not need to manually modify these files.
`utils` - various utilities that help do offline processing, like high volume autointerp, or generating dashboards, or exporting data.

# security

please report vulnerabilities to [johnny@neuronpedia.org](mailto:johnny@neuronpedia.org).

we don't currently have an official bounty program, but we'll try our best to give compensation based on the severity of the vulnerability - though it's likely we will not able able to offer awards for any low-severity vulnerabilities.

# contact / support

- slack: [join #neuronpedia](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-2o756ku1c-_yKBeUQMVfS_p_qcK6QLeA)
- email: [johnny@neuronpedia.org](mailto:johnny@neuronpedia.org)
- issues: [github issues](https://github.com/hijohnnylin/neuronpedia/issues)

# contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

# appendix

### 'make' commands reference

you can view all available `make` commands and brief descriptions of them by running `make help`

### import data into your local database

if you set up your own database, it will start out empty - no features, explanations, activations, etc. to load this data, there's a built-in `admin panel` where you can download this data for SAEs (or "sources") of your choosing.

> ⚠️ **warning:** the admin panel is finicky and does not currently support resuming imports. if an import is interrupted, you must manually click `re-sync`. the admin panel currently does not check if your download is complete or missing parts - it is up to you to check if the data is complete, and if not, to click `re-sync` to re-download the entire dataset.

> ℹ️ **recommendation:** when importing data, start with just one source (like `gpt2-small`@`10-res-jb`) instead of downloading everything at once. This makes it easier to verify the data imported correctly and lets you start using neuronpedia faster.

the instructions below demonstrate how to download the `gpt2-small`@`10-res-jb` SAE data.

1. navigate to [localhost:3000/admin](http://localhost:3000/admin).
2. scroll down to `gpt2-small`, and expand `res-jb` with the `▶`.
3. click `Download` next to `10-res-jb`.
4. wait patiently - this can be a _LOT_ of data, and depending on your connection/cpu speed it can take up to 30 minutes or an hour.
5. once it's done, click `Browse` or use the navbar to try it out: `Jump To`/`Search`/`Steer`.
6. repeat for other SAE/source data you wish to download.

### why an openai api key is needed for search explanations

in the webapp, the `search explanations` feature requires you to set an `OPENAI_API_KEY`. otherwise you will get no search results.

this is because the `search explanations` functionality searches for features by semantic similarity. if you search `cat`, it will also return `feline`, `tabby`, `animal`, etc. to do this, it needs to calculate the embedding for your input `cat`. we use openai's embedding api (specifically, `text-embedding-3-large` with `dimension: 256`) to calculate the embeddings.
