#### neuronpedia 🧠🔍 openapi schemas

- [what this is](#what-this-is)
- [install openapi generator cli](#install-openapi-generator-cli)
- [making changes to the inference server](#making-changes-to-the-inference-server)
- [making changes to the autointerp server](#making-changes-to-the-autointerp-server)

## what this is

neuronpedia's [services](../README.md#architecture) has a webapp (nextjs) in **typescript** and separate **python** servers for [inference](../apps/inference/README.md) and [autointerp](../apps/autointerp/README.md). to ensure that our servers communicate with each other in a predictable and typesafe manner, the requests to the inference and autointerp servers are defined by [openapi schemas](https://www.openapis.org).

having openapi schemas allows us to use [openapi client generators](https://openapi-generator.tech/docs/installation/) to automatically create clients that make and parse requests.

this readme documents how to update the schemas and re-generate clients when we want to make additions or changes to the api.

## install openapi generator cli

install the [openapi generator cli](https://openapi-generator.tech/docs/installation/#npm) here.

## making changes to the inference server

here's how to create or update an endpoint in the inference server.

1. spec out your new addition(s) or change(s) under [openapi/inference-server.yaml](openapi/inference-server.yaml) and the [openapi/inference](openapi/inference/) subdirectory. for example, you might create a new endpoint under [openapi/inference/paths](openapi/inference/paths/), then add that new endpoint path to the `inference-server.yaml` file
2. run the command, replacing `BUMPED_SEMANTIC_VERSION`
   ```
   make setup-all-inference VERSION=BUMPED_SEMANTIC_VERSION
   ```
3. update the code under [apps/inference/neuronpedia_inference](../apps/inference/neuronpedia_inference/) for your changes. for example, if you were creating a new endpoint of `POST [host]/v1/util/action`, you would create a new `action.py` file under [apps/inference/neuronpedia_inference/util](apps/inference/neuronpedia_inference/util).
4. update your webapp code, starting by changing [../apps/webapp/lib/utils/inference.ts](../apps/webapp/lib/utils/inference.ts) to use the new/updated endpoints in the `neuronpedia-inference-client`
5. bring up the webapp and inference server locally (see the main [readme](../README.md) for running dev instances), and test the calls between the two servers.
6. once you're satisfied with the results, commit all your changes.
7. [highly encouraged] make a PR to the official [neuronpedia repo](https://github.com/hijohnnylin/neuronpedia) and we'll review it ASAP!

## making changes to the autointerp server

the instructions are _mostly_ the same as [making changes to the inference server](#making-changes-to-the-inference-server), except names are changed from `neuronpedia-inference` to `neuronpedia-autointerp`, etc.

1. spec out your new addition(s) or change(s) under [openapi/autointerp-server.yaml](openapi/autointerp-server.yaml) and the [openapi/autointerp](openapi/autointerp/) subdirectory.
2. run the command, replacing `BUMPED_SEMANTIC_VERSION`
   ```
   make setup-all-autointerp VERSION=BUMPED_SEMANTIC_VERSION
   ```
3. update the code under [apps/autointerp/neuronpedia_autointerp](../apps/autointerp/neuronpedia_autointerp/) for your changes.
4. update your webapp code, starting by changing [../apps/webapp/lib/utils/autointerp.ts](../apps/webapp/lib/utils/autointerp.ts) to use the new/updated endpoints in the `neuronpedia-autointerp-client`
5. bring up the webapp and autointerp server locally (see the main [readme](../README.md) for running dev instances), and test the calls between the two servers.
6. once you're satisfied with the results, commit all your changes.
7. [highly encouraged] make a PR to the official [neuronpedia repo](https://github.com/hijohnnylin/neuronpedia) and we'll review it ASAP!
