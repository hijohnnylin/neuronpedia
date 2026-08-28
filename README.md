<p align="center">
  <a href="https://github.com/hijohnnylin/neuronpedia">
    <img src="https://github.com/user-attachments/assets/9bcea0bf-4fa9-401d-bb7a-d031a4d12636" alt="Splash GIF"/>
  </a>

<h3 align="center"><a href="https://neuronpedia.org">neuronpedia.org 🧠🔍</a></h3>

  <p align="center">
    Open source interpretability platform
    <br />
    <sub>
    <strong>api · steering · activations · circuits/graphs · natural language autoencoders · jacobian lens · autointerp · scoring · inference · search · filter · dashboards · benchmarks · cossim · umap · embeds · probes · saes · lists · exports · uploads</strong>
    </sub>
  </p>
</p>

<p align="center" style="color: #cccccc;">
  <a href="https://github.com/hijohnnylin/neuronpedia/blob/main/LICENSE"><img height="20px" src="https://img.shields.io/badge/license-Apache%202.0-yellow.svg" alt="Apache 2.0"></a>
  <a href="https://status.neuronpedia.org"><img height="20px" src="https://uptime.betterstack.com/status-badges/v2/monitor/1roih.svg" alt="Uptime"></a>
  <a href="https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3z9o0hxjl-MDX9pbATO2qESOazNDLpdQ"><img height="20px" src="https://img.shields.io/badge/slack-purple?logo=slack&logoColor=white" alt="Slack"></a>
  <a href="mailto:johnny@neuronpedia.org"><img height="20px" src="https://img.shields.io/badge/contact-blue.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiPjwvZz48ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjwvZz48ZyBpZD0iU1ZHUmVwb19pY29uQ2FycmllciI+IDxwYXRoIGQ9Ik00IDcuMDAwMDVMMTAuMiAxMS42NUMxMS4yNjY3IDEyLjQ1IDEyLjczMzMgMTIuNDUgMTMuOCAxMS42NUwyMCA3IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48L3BhdGg+IDxyZWN0IHg9IjMiIHk9IjUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxNCIgcng9IjIiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiPjwvcmVjdD4gPC9nPjwvc3ZnPg==" alt="Email"></a>
  <a href="https://neuronpedia.org/blog"><img height="20px" src="https://img.shields.io/badge/blog-10b981.svg" alt="blog"></a>
  <a href="https://neuronpedia.org"><img height="20px" src="https://img.shields.io/badge/website-gray.svg" alt="website"></a>

</p>

- [About Neuronpedia](#about-neuronpedia)
- [Setting Up Your Local Environment](#setting-up-your-local-environment)
  - ["I Want to Use a Local Database / Import More Neuronpedia Data"](#i-want-to-use-a-local-database--import-more-neuronpedia-data)
  - ["I Want to Do Webapp (Frontend + API) Development"](#i-want-to-do-webapp-frontend--api-development)
  - ["I Want to Run/Develop Inference Locally"](#i-want-to-rundevelop-inference-locally)
  - ['I Want to Run/Develop the Graph Server Locally'](#i-want-to-rundevelop-the-graph-server-locally)
  - ['I Want to Run/Develop Autointerp Locally'](#i-want-to-rundevelop-autointerp-locally)
  - ['I Want to Do High Volume Autointerp Explanations'](#i-want-to-do-high-volume-autointerp-explanations)
  - ['I Want to Generate My Own Dashboards/Data and Add It to Neuronpedia'](#i-want-to-generate-my-own-dashboardsdata-and-add-it-to-neuronpedia)
- [Architecture](#architecture)
  - [Requirements](#requirements)
  - [Services](#services)
    - [Services Are Standalone Apps](#services-are-standalone-apps)
    - [Service-Specific Documentation](#service-specific-documentation)
  - [OpenAPI](#openapi)
  - [Monorepo Directory Structure](#monorepo-directory-structure)
- [Security](#security)
- [Contact / Support](#contact--support)
- [Contributing](#contributing)
- [Appendix](#appendix)
  - ['Make' Commands Reference](#make-commands-reference)
  - [Import Data Into Your Local Database](#import-data-into-your-local-database)
  - [Why an OpenAI API Key Is Needed for Search Explanations](#why-an-openai-api-key-is-needed-for-search-explanations)

# About Neuronpedia

Check out our [blog post](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source) about Neuronpedia, why we're open sourcing it, and other details. There's also a [tweet thread](https://x.com/neuronpedia/status/1906793456879775745) with quick demos.

**Feature Overview**

A diagram showing the main features of Neuronpedia as of March 2025.
![neuronpedia-features](https://github.com/user-attachments/assets/13e07a93-e046-4e1c-b670-2d26d251d55d)

# Setting Up Your Local Environment

Every Neuronpedia service runs directly on your machine. Start by setting up your [local database](#i-want-to-use-a-local-database--import-more-neuronpedia-data).

## "I Want to Use a Local Database / Import More Neuronpedia Data"

#### What This Does + What You'll Get

These steps show you how to configure and connect to your own local database. You can then download sources/SAEs of your choosing:

https://github.com/user-attachments/assets/d7fbb46e-8522-4f98-aa08-21c6529424af

> ⚠️ **Warning:** Your database will start out empty. You will need to use the admin panel to [import sources/data](#import-data-into-your-local-database) (activations, explanations, etc).

> ⚠️ **Warning:** The local database environment does not have any inference servers connected, so you won't be able to do activation testing, steering, etc initially. You will need to [configure a local inference instance](#i-want-to-rundevelop-inference-locally).

#### Steps

1. Install Postgres 16+ along with the [pgvector](https://github.com/pgvector/pgvector) extension, which Neuronpedia uses to search explanations by meaning.
   ```
   # macos (homebrew)
   brew install postgresql@16 pgvector && brew services start postgresql@16
   ```
   ```
   # debian / ubuntu
   sudo apt install postgresql-16 postgresql-16-pgvector && sudo systemctl start postgresql
   ```
   For other platforms, see the [pgvector installation notes](https://github.com/pgvector/pgvector#installation).
2. Check that Neuronpedia can reach your database
   ```
   make db-check
   ```
   > ➡️ Connection details live in `apps/webapp/.env.localhost` and default to user `postgres`, password `postgres`, and database `postgres` on port `5432`. If yours differ, edit `POSTGRES_PRISMA_URL` and `POSTGRES_URL_NON_POOLING` there.
3. Create the tables and seed the initial rows
   ```
   make db-init
   ```
4. Bring up the webapp by following [webapp development](#i-want-to-do-webapp-frontend--api-development) below, then go to [localhost:3000](http://localhost:3000) to see your local instance connected to your local database
5. See the `warnings` above for caveats, and `next steps` to finish setting up

#### Next Steps

1. [Click here](#import-data-into-your-local-database) for how to import data into your local database (activations, explanations, etc), because your local database will be empty to start
2. [Click here](#i-want-to-rundevelop-inference-locally) for how to bring up a local `inference` service for the model/source/SAE you're working with

## "I Want to Do Webapp (Frontend + API) Development"

#### What You'll Get

The webapp serves the frontend and the API. Running it in development mode gives you fast reloads on every change and more informative debug output. If you are purely interested in doing frontend/api development for Neuronpedia, you don't need to set up anything else!

#### Steps

1. Install [Node.js](https://nodejs.org) via [Node Version Manager](https://github.com/nvm-sh/nvm)
   ```
   make install-nodejs
   ```
2. Install the webapp's dependencies
   ```
   make webapp-install
   ```
3. Run the development server
   ```
   make webapp-dev
   ```
4. Go to [localhost:3000](http://localhost:3000) to see your local webapp instance

#### Doing Local Webapp Development

- **Auto-reload**: When you change any files in the `apps/webapp` subdirectory, the `localhost:3000` will automatically reload
- **Install commands**: You do not need to run `make install-nodejs` again, and you only need to run `make webapp-install` if dependencies change
- **Production build**: `make webapp-build` followed by `make webapp-run` serves an optimized build instead - slower to build, faster to run, and without debug information

## "I Want to Run/Develop Inference Locally"

#### What This Does + What You'll Get

This subsection shows you how to run an inference instance locally so you can do things like steering, activation testing, etc on the sources/SAEs you've downloaded.

> ⚠️ **Warning:** For the local environment, we only support running one inference server at a time. This is because you are unlikely to be running multiple models simultaneously on one machine, as they are memory and compute intensive.

#### Steps

1. Ensure you have [installed uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install the inference server's dependencies
   ```
   make inference-install
   ```
3. Run the inference server, using the `MODEL_SOURCESET` argument to specify the `.env.inference.[model_sourceset]` file you're loading from. For this example, we will run `gpt2-small`, and load the `res-jb` sourceset/SAE set, which is configured in the `.env.inference.gpt2-small.res-jb` file. You can see the other [pre-loaded inference configs](#pre-loaded-inference-server-configurations) or [create your own config](#making-your-own-inference-server-configurations) as well.

   ```
   make inference-dev MODEL_SOURCESET=gpt2-small.res-jb
   ```

   > ➡️ The server picks its own backend and device: vLLM on CUDA where the architecture supports it, otherwise eager PyTorch. Models are read from your normal Hugging Face cache at `~/.cache/huggingface`, so weights you've already downloaded are reused.

4. Wait for it to load (first time will take longer). When you see `Initialized: True`, the local inference server is now ready on `localhost:5002`
5. Tell the webapp about it. The webapp looks up every GPU server in the `ComputeHost` table, so a server it has never been told about is invisible to it:

   ```
   make host-add SERVICE=INFERENCE MODEL=gpt2-small URL=http://127.0.0.1:5002 SOURCES=6-res-jb
   ```

   Leave `SOURCES` off to say "this host can serve anything for the model", which is what jlens and steering with vectors need. `make host-list` shows what is registered and `make host-remove` takes one away.

#### Using the Inference Server

To interact with the inference server, you have a few options - note that this will only work for the model / selected source you have loaded:

1.  Load the webapp with the [local database setup](#i-want-to-use-a-local-database--import-more-neuronpedia-data), then using the model / selected source as you would normally do on Neuronpedia.
2.  Use the OpenAPI spec at `apps/inference/openapi.json` to make calls with any client of your choice, or to generate one. You can get a Swagger interactive spec at `/docs` after the server starts up. See the `apps/inference/README.md` for details. (Set environment variable `INFERENCE_SERVER_SECRET` to `localhost-secret`, or whatever it's set to in `apps/webapp/.env.localhost` if you've changed it.)

#### Pre-Loaded Inference Server Configurations

We've provided some pre-loaded inference configs as examples of how to load a specific model and sourceset for inference. View them by running `make inference-list-configs`:

```
$ make inference-list-configs

Available Inference Configurations (.env.inference.*)
================================================

deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k
    Model: meta-llama/Llama-3.1-8B
    Source/SAE Sets: '["llamascope-slimpj-res-32k"]'
    make inference-dev MODEL_SOURCESET=deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k

gemma-2-2b-it.gemmascope-res-16k
    Model: gemma-2-2b-it
    Source/SAE Sets: '["gemmascope-res-16k"]'
    make inference-dev MODEL_SOURCESET=gemma-2-2b-it.gemmascope-res-16k

gpt2-small.res-jb
    Model: gpt2-small
    Source/SAE Sets: '["res-jb"]'
    make inference-dev MODEL_SOURCESET=gpt2-small.res-jb
```

#### Making Your Own Inference Server Configurations

Look at the `.env.inference.*` files for examples on how to make these inference server configurations.

The `MODEL_ID` is the Hugging Face repo id of the model (`openai-community/gpt2`, `google/gemma-2-2b`), which is what the weights load from. Each of `SAE_SETS` is the text after the layer number and hyphen in a Neuronpedia source ID - for example, if you have a Neuronpedia feature at url `http://neuronpedia.org/gpt2-small/0-res-jb/123`, the `0-res-jb` is the source ID, and the item in the `SAE_SETS` is `res-jb`. This example matches the `.env.inference.gpt2-small.res-jb` file exactly.

You can find Neuronpedia source IDs in the SAELens [pretrained SAEs YAML file](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml) or by clicking into models in the [Neuronpedia datasets exports](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/) directory.

**Using Models Not Officially Supported by TransformerLens**
Look at the `.env.inference.deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k` to see an example of how to load a model not officially supported by TransformerLens. This is mostly for swapping in weights of a distilled/fine-tuned model.

**Loading Non-SAELens Sources/SAEs**

- [TODO #2](https://github.com/hijohnnylin/neuronpedia/issues/2) Document how to load SAEs/sources that are not in SAELens pretrained YAML

#### Doing Local Inference Development

- **The Pydantic models are the spec**: To add or change an endpoint, edit the models under `apps/inference/neuronpedia_inference/schemas/`, then run `make inference-openapi` and `make webapp-openapi` to refresh the committed `openapi.json` and the webapp's TypeScript types. There is no schema file to edit first. See [OpenAPI](#openapi) below and the "Cross-server APIs" section of [AGENTS.md](AGENTS.md).
- **No auto-reload**: When you change any files in the `apps/inference` subdirectory, the inference server will _NOT_ automatically reload, because server reloads are slow: they reload the model and all sources/SAEs. If you want to enable autoreload, then append `AUTORELOAD=1` to the `make inference-dev` call, like so:
  ```
  make inference-dev \
       MODEL_SOURCESET=gpt2-small.res-jb \
       AUTORELOAD=1
  ```

## 'I Want to Run/Develop the Graph Server Locally'

#### What This Does + What You'll Get

The graph server powers the attribution graph generation functionality, built on top of [circuit-tracer](https://github.com/safety-research/circuit-tracer) by Piotrowski & Hanna. This service handles the backend processing when you create new graphs through the [Neuronpedia Circuit Tracer](https://www.neuronpedia.org/gemma-2-2b/graph) interface.

#### Steps

1. Ensure you have [installed uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install the graph server's dependencies
   ```
   make graph-install
   ```
3. Within the `apps/graph` directory, create a `.env` file with `HF_TOKEN` (see `apps/graph/.env.example`)
   - Make sure your `HF_TOKEN` has access to the [Gemma-2-2B model](https://huggingface.co/google/gemma-2-2b) on Hugging Face.
   - The server secret passed in the `x-secret-key` request header defaults to `localhost-secret`. Override it with `make graph-dev LOCALHOST_SECRET=your-secret`, and set `GRAPH_SERVER_SECRET` in `apps/webapp/.env.localhost` to match.
4. Run the graph server:

   ```
   make graph-dev
   ```

5. Wait for it to load. The graph server is then ready on `localhost:5004`
6. Register it with the webapp, which routes graph requests by source set:

   ```
   make host-add SERVICE=GRAPH MODEL=gemma-2-2b URL=http://127.0.0.1:5004 SOURCE_SETS=gemmascope-transcoder-16k
   ```

For example requests, see the [Graph Server README](apps/graph/README.md#example-request---output-graph-json-directly).

## 'I Want to Run/Develop Autointerp Locally'

#### What This Does + What You'll Get

The autointerp server provides automatic interpretation and scoring of neural network features. It uses EleutherAI's [Delphi](https://github.com/EleutherAI/delphi) for generating explanations and scoring.

> ⚠️ **Warning:** The Eleuther embedding scorer uses an embedding model only supported on CUDA (it won't work on Mac MPS or CPU)

#### Steps

1. Ensure you have [installed uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install the autointerp server's dependencies
   ```
   make autointerp-install
   ```
3. Run the autointerp server:

   ```
   make autointerp-dev
   ```

4. Wait for it to load. The autointerp server is then ready on `localhost:5003`

#### Using the Autointerp Server

To interact with the autointerp server, you have a few options:

1. Use the OpenAPI spec at `apps/autointerp/openapi.json` to make calls with any client of your choice, or to generate one. You can get a Swagger interactive spec at `/docs` after the server starts up. (Set environment variable `AUTOINTERP_SERVER_SECRET` to `localhost-secret`, or whatever it's set to in `apps/webapp/.env.localhost` if you've changed it.)

#### Doing Local Autointerp Development

- **The Pydantic models are the spec**: To add or change an endpoint, edit the models under `apps/autointerp/neuronpedia_autointerp/schemas/`, then run `make autointerp-openapi` and `make webapp-openapi` to refresh the committed `openapi.json` and the webapp's TypeScript types. There is no schema file to edit first. See [OpenAPI](#openapi) below and the "Cross-server APIs" section of [AGENTS.md](AGENTS.md).
- **No auto-reload**: When you change any files in the `apps/autointerp` subdirectory, the autointerp server will _NOT_ automatically reload. Restart `make autointerp-dev` to pick up changes.

## 'I Want to Do High Volume Autointerp Explanations'

This section is under construction.

- Use EleutherAI's [Delphi library](https://github.com/EleutherAI/delphi)
- For OpenAI's autointerp, use [utils/neuronpedia_utils/batch-autointerp.py](utils/neuronpedia-utils/neuronpedia_utils/batch-autointerp.py)

## 'I Want to Generate My Own Dashboards/Data and Add It to Neuronpedia'

This section is under construction.

[TODO: Simplify generation + upload of data to Neuronpedia](https://github.com/hijohnnylin/neuronpedia/issues/46)

[TODO: neuronpedia-utils should use Poetry](https://github.com/hijohnnylin/neuronpedia/issues/43)

In this example, we will generate dashboards/data for an [SAELens](https://github.com/jbloomAus/SAELens)-compatible SAE, and upload it to our own Neuronpedia instance.

1. Ensure you have [Poetry installed](https://python-poetry.org/docs/)
2. [Upload](https://github.com/jbloomAus/SAELens/blob/main/tutorials/uploading_saes_to_huggingface.ipynb) your SAELens-compatible source/SAE to Hugging Face.
   > Example
   > ➡️ [https://huggingface.co/chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40](https://huggingface.co/chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40)
3. Clone SAELens locally.
   ```
   git clone https://github.com/jbloomAus/SAELens.git
   ```
4. Open your cloned SAELens and edit the file `sae_lens/pretrained_saes.yaml`. Add a new entry at the bottom, based on the template below (see comments for how to fill it out):
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
       neuronpedia: gemma-2-2b/0-matryoshka-res-dc  # what you expect the Neuronpedia URI to be - neuronpedia.org/[this_slug]. should be [model_id]/[layer]-[identical_slug_for_this_sae_set]
     - id: blocks.1.hook_resid_post                 # more SAEs in this SAE set
       path: standard/blocks.1.hook_resid_post
       l0: 40.0
       neuronpedia: gemma-2-2b/1-matryoshka-res-dc  # note that this is identical to the entry above, except 1 instead of 0 for the layer
     - [...]
   ```
5. Clone [SAEDashboard](https://github.com/jbloomAus/SAEDashboard.git) locally.
   ```
   git clone https://github.com/jbloomAus/SAEDashboard.git
   ```
6. Configure your cloned `SAEDashboard` to use your cloned modified `SAELens`, instead of the one in production
   ```
   cd SAEDashboard                    # set directory
   poetry lock && poetry install      # install dependencies
   poetry remove sae-lens             # remove production dependency
   poetry add PATH/TO/CLONED/SAELENS  # set local dependency
   ```
7. Generate dashboards for the SAE. This will take from 30 min to a few hours, depending on your hardware and size of model.

   ```
   cd SAEDashboard                    # set directory
   rm -rf cached_activations          # clear old cached data

   # start the generation. details for each argument (full details: https://github.com/jbloomAus/SAEDashboard/blob/main/sae_dashboard/neuronpedia/neuronpedia_runner_config.py)
   #     - sae-set = should match the unique ID for the set from pretrained_saes.yaml
   #     - sae-path = should match the id for the sae in from pretrained_saes.yaml
   #     - np-set-name = should match the [identical_slug_for_this_sae_set] for the sae.Neuronpedia from pretrained_saes.yaml
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

8. Convert these dashboards for import into Neuronpedia
   ```
   cd neuronpedia/utils/neuronpedia-utils          # get into this current repository's util directory
   python convert-saedashboard-to-neuronpedia.py   # start guided conversion script. follow the steps.
   ```
9. Once dashboard files are generated for Neuronpedia, upload these to the global Neuronpedia S3 bucket - currently you need to [contact us](mailto:johnny@neuronpedia.org) to do this.
10. From a localhost instance, [import your data](#i-want-to-use-a-local-database--import-more-neuronpedia-data)

# Architecture

Here's how the services/scripts connect in Neuronpedia. It's easiest to read this diagram by starting at the image of the laptop ("User").

![architecture diagram](architecture.png)

## Requirements

You can run Neuronpedia on any cloud and on any modern OS. Neuronpedia is designed to avoid vendor lock-in. These instructions were written for and tested on macOS 15 (Sequoia), so you may need to repurpose commands for Windows/Ubuntu/etc. At least 16GB RAM is recommended.

Each service runs directly on the host, so install the toolchain for the ones you plan to work on:

| Service          | Needs                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| webapp           | [Node.js](https://nodejs.org) 22+ (`make install-nodejs`)                                                                  |
| database         | Postgres 16+ with [pgvector](https://github.com/pgvector/pgvector)                                                         |
| inference, graph | [uv](https://docs.astral.sh/uv/getting-started/installation/), and a CUDA GPU for anything larger than the smallest models |
| autointerp       | [uv](https://docs.astral.sh/uv/getting-started/installation/), and a CUDA GPU for the Eleuther embedding scorer            |
| nla, sparsity    | [uv](https://docs.astral.sh/uv/getting-started/installation/), and a CUDA GPU                                              |

## Services

| Name       | Port | Description                                                                                                                                                  | Powered by                                                                                                                                             |
| ---------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| webapp     | 3000 | Serves the neuronpedia.org frontend and [the API](https://neuronpedia.org/api-doc)                                                                           | [Next.js](https://nextjs.org) / React                                                                                                                  |
| database   | 5432 | Stores features, activations, explanations, users, lists, etc                                                                                                | Postgres                                                                                                                                               |
| inference  | 5002 | [Support server] Steering, activation testing, search via inference, topk, etc. A separate instance is required for each model you want to run inference on. | Python / Torch                                                                                                                                         |
| autointerp | 5003 | [Support server] Auto-interp explanations and scoring, using EleutherAI's [Delphi](https://github.com/EleutherAI/delphi) (formerly `sae-auto-interp`)        | Python                                                                                                                                                 |
| graph      | 5004 | [Support server] Builds attribution graphs (circuit traces) for a prompt                                                                                     | Python / [circuit-tracer](https://github.com/safety-research/circuit-tracer) or [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) |
| sparsity   | 5005 | [Support server] Analyzes MLP neuron connections in sparse circuit models                                                                                    | Python / [circuit_sparsity](https://github.com/openai/circuit_sparsity)                                                                                |
| nla        | 5009 | [Support server] Natural Language Autoencoders: turns activation vectors into natural language descriptions, and back                                        | Python / Torch                                                                                                                                         |

### Services Are Standalone Apps

By design, each service can be run independently as a standalone app. This is to enable extensibility and forkability.

For example, if you like the Neuronpedia webapp frontend but want to use a different API for inference, you can do that! Just ensure your alternative inference server matches the `apps/inference/openapi.json` spec, and/or that you modify the Neuronpedia calls to inference under `apps/webapp/lib/utils`.

### Service-Specific Documentation

There are draft `README`s for each specific app/service under `apps/[service]`, but they are heavily WIP. Each service's `pyproject.toml` or `package.json` under the same directory lists its dependencies if you want to run or package it yourself.

## OpenAPI

For services to communicate with each other in a typed and consistent way, we generate types from OpenAPI — in one direction, from the Python out.

Each Python service's Pydantic models are the source of truth. `make <app>-openapi` dumps that server's route table to `apps/<app>/openapi.json`, and `make webapp-openapi` turns every one of those specs into `apps/webapp/lib/api/<app>.d.ts` for the webapp to compile against. Both artifacts are committed, and both are guarded: a drift test in each service's own suite catches a stale `openapi.json`, and `.github/workflows/openapi-drift.yml` catches stale TypeScript.

So there is no spec file to hand-edit and nothing to publish by hand. To change a wire format, change the Pydantic model and run `make openapi`, which does both halves for every service you have installed — regenerating the spec but not the TypeScript is the easy mistake, and it surfaces in CI rather than locally.

The `neuronpedia-{inference,autointerp}-client` packages on npm and PyPI are still published for callers outside this repo, but nothing here imports them and they are committed nowhere. `.github/workflows/openapi-publish.yml` rebuilds them from the same committed `openapi.json`, so they are downstream of a wire-format change rather than a step in making one. `make sdk-dry-run SERVICE=inference` runs that build locally without uploading.

The one thing this cannot cover is streaming: SSE and NDJSON frames are not response bodies, so they never reach a spec. Those are pinned by contract tests instead — see `apps/inference/tests/unit/test_lens_frame_contract.py` and `apps/nla/tests/test_frame_contract.py`.

For the full workflow, including which servers are camelCase on the wire and which are deliberately snake_case, see the "Cross-server APIs" section of [AGENTS.md](AGENTS.md).

## Monorepo Directory Structure

`apps` - The six Neuronpedia services: webapp, inference, autointerp, graph, nla, and sparsity. Most of the code is here.
`utils` - Various utilities that help do offline processing, like high volume autointerp, or generating dashboards, or exporting data.
`webapp-python-client` - The hand-written Python SDK for the public API, published to PyPI as `neuronpedia`.

The interpretability engine that inference, graph and nla run on — hooking, capture, steering, the vLLM backend — is [`interp-engine`](https://github.com/decoderesearch/interp-engine), a separate repository published to PyPI. Those three apps pin a release of it; see "interp-engine is a dependency" in [AGENTS.md](AGENTS.md) for working on both at once.

# Security

Please report vulnerabilities to [johnny@neuronpedia.org](mailto:johnny@neuronpedia.org).

We don't currently have an official bounty program, but we'll try our best to give compensation based on the severity of the vulnerability - though it's likely we will not able able to offer awards for any low-severity vulnerabilities.

# Contact / Support

- Slack: [join #neuronpedia](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3z9o0hxjl-MDX9pbATO2qESOazNDLpdQ)
- Email: [johnny@neuronpedia.org](mailto:johnny@neuronpedia.org)
- Issues: [GitHub issues](https://github.com/hijohnnylin/neuronpedia/issues)

# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

The checks that gate a pull request are the ones in `make python-lint` and `npm run lint`. You can
have the fast ones run on each commit, over the files you changed, with `make githooks-install` -
`make webapp-install` enables the same `.githooks/` hook for you through npm. It is per checkout and
optional: see ["Checks Before You Commit"](CONTRIBUTING.md#checks-before-you-commit).

# Appendix

### 'Make' Commands Reference

You can view all available `make` commands and brief descriptions of them by running `make help`

### Import Data Into Your Local Database

If you set up your own database, it will start out empty - no features, explanations, activations, etc. To load this data, there's a built-in `admin panel` where you can download this data for SAEs (or "sources") of your choosing.

> ⚠️ **Warning:** The admin panel is finicky and does not currently support resuming imports. If an import is interrupted, you must manually click `re-sync`. The admin panel currently does not check if your download is complete or missing parts - it is up to you to check if the data is complete, and if not, to click `re-sync` to re-download the entire dataset.

> ℹ️ **Recommendation:** When importing data, start with just one source (like `gpt2-small`@`10-res-jb`) instead of downloading everything at once. This makes it easier to verify the data imported correctly and lets you start using Neuronpedia faster.

The instructions below demonstrate how to download the `gpt2-small`@`10-res-jb` SAE data.

1. Navigate to [localhost:3000/admin](http://localhost:3000/admin).
2. Scroll down to `gpt2-small`, and expand `res-jb` with the `▶`.
3. Click `Download` next to `10-res-jb`.
4. Wait patiently - this can be a _LOT_ of data, and depending on your connection/CPU speed it can take up to 30 minutes or an hour.
5. Once it's done, click `Browse` or use the navbar to try it out: `Jump To`/`Search`/`Steer`.
6. Repeat for other SAE/source data you wish to download.

### Why an OpenAI API Key Is Needed for Search Explanations

In the webapp, the `Search Explanations` feature requires you to set an `OPENAI_API_KEY`. Otherwise you will get no search results.

This is because the `search explanations` functionality searches for features by semantic similarity. If you search `cat`, it will also return `feline`, `tabby`, `animal`, etc. To do this, it needs to calculate the embedding for your input `cat`. We use OpenAI's embedding API (specifically, `text-embedding-3-large` with `dimension: 256`) to calculate the embeddings.
