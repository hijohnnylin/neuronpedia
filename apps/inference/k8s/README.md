#### neuronpedia 🧠🔍 inference server - k8s

we currently run our inference servers on kubernetes (k8s) on google cloud. the k8s configs/customizations are in this directory.

the following are instructions for how we set up this up. you'll need to customize these depending on your own setup.

> ⚠️ **warning:** this is _draft_ documentation. we expect to make this better soon - for example, with more general instructions for non-gcloud infrastructure. probably we should have some scripts and ci/cd as well!

- [install + configure google cloud](#install--configure-google-cloud)
- [create k8s cluster + pools](#create-k8s-cluster--pools)
- [configure + connect to k8s cluster](#configure--connect-to-k8s-cluster)
- [deploy instances + apply config changes](#deploy-instances--apply-config-changes)
- [restart instances to pick up new docker image](#restart-instances-to-pick-up-new-docker-image)
- [misc k8s commands](#misc-k8s-commands)
  - [delete instance](#delete-instance)
  - [updating node pool configs](#updating-node-pool-configs)
  - [preview changes for config](#preview-changes-for-config)
  - ["ssh" into the container](#ssh-into-the-container)
  - [describe container](#describe-container)

you'll probably want to follow the instructions in order, up until and including [deploy instances](#deploy-instances--apply-config-changes)

## install + configure google cloud

1. install [google cloud cli](https://cloud.google.com/sdk/docs/install)
2. install the kubernetes plugin
   ```
   gcloud components install gke-gcloud-auth-plugin
   ```
3. config and create a default google cloud project

   ```
   # change this to whatever zone/region you prefer
   gcloud config set compute/zone us-central1-c
   gcloud config set compute/region us-central1

   gcloud projects create [your_new_project_id]
   gcloud config set project [your_new_project_id]
   ```

## create k8s cluster + pools

1. create cluster with a cpu node pool. we use this one for `gpt2-small`.

   ```
   # gke's c4 instance types don't support a specific type of storage (pd-balanced) so we use c3
   gcloud container clusters create neuronpedia-inference \
    --num-nodes=1 \
    --machine-type=c3-highmem-8 \
    --addons=GcePersistentDiskCsiDriver \
    --disk-size=250 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=2
   ```

2. add a gpu node pool to the cluster with a100s: `a2-highgpu-1g`
   > ⚠️ **warning:** this is expensive and you will need to [request quota](https://cloud.google.com/compute/resource-usage) to use gpus on google cloud
   ```
   gcloud container node-pools create gpu-pool \
    --cluster=neuronpedia-inference \
    --machine-type=a2-highgpu-1g \
    --disk-size=250 \
    --enable-autoscaling \
    --num-nodes=1 \
    --min-nodes=1 \
    --max-nodes=16
   ```
   if you prefere a cheaper gpu (less ram), you can use nvidia L4's instead: `g2-standard-8`
   ```
   gcloud container node-pools create gpu-lite-pool \
    --cluster=neuronpedia-inference \
    --machine-type=g2-standard-8 \
    --disk-size=250 \
    --enable-autoscaling \
    --num-nodes=1 \
    --min-nodes=1 \
    --max-nodes=10
   ```

## configure + connect to k8s cluster

1. ensure you have `kubectl` installed. if you installed docker desktop, you have it already. otherwise, [install kubectl](https://kubernetes.io/docs/tasks/tools/).
2. load google cloud credentials into kubectl
   ```
   gcloud container clusters get-credentials neuronpedia-inference
   ```
3. set secrets in gcloud kubectl
   ```
   kubectl create secret generic server-secret --from-literal=SECRET='your-secret-value'
   kubectl create secret generic hf-token --from-literal=HF_TOKEN='your-secret-value'
   ```

## deploy instances + apply config changes

if we're making a config/customization change, or if we want to deploy a new inference instance, we use `kubectl apply -k [path_to_kustomization_dir]`, like so:

```
# sets the project_id to fetch the correct image during the kubectl commands later
export PROJECT_ID=$(gcloud config get-value project)

kubectl apply -k k8s/overlays/cpu/gpt2-small && \
kubectl apply -k k8s/overlays/cpu/gpt2-small-public && \
kubectl apply -k k8s/overlays/gpu/gemma-2-2b-it-a && \
kubectl apply -k k8s/overlays/gpu/gemma-2-2b-it-b && \
kubectl apply -k k8s/overlays/gpu/gemma-2-2b-public && \
kubectl apply -k k8s/overlays/gpu/gemma-2-2b-it-public && \
kubectl apply -k k8s/overlays/gpu/gemma-2-9b-it-a && \
kubectl apply -k k8s/overlays/gpu/gemma-2-9b-it-b && \
kubectl apply -k k8s/overlays/gpu/gemma-2-2b && \
kubectl apply -k k8s/overlays/gpu/gemma-2-9b && \
kubectl apply -k k8s/overlays/gpu/deepseek-r1-llama-8b-a && \
kubectl apply -k k8s/overlays/gpu/deepseek-r1-llama-8b-b && \
kubectl apply -k k8s/overlays/gpu/deepseek-r1-llama-8b-public-a && \
kubectl apply -k k8s/overlays/gpu/deepseek-r1-llama-8b-public-b && \
kubectl apply -k k8s/overlays/gpu/llama-31-8b
```

## restart instances to pick up new docker image

if we have made a new code change and created a new docker image (see `apps/inference/README.md#setup--run---docker`), we use `kubectl rollout restart deployment [deployment_id]` to do a rolling deploy of the new image, like so:

```
kubectl rollout restart deployment gpt2-small-cpu-neuronpedia-inference && \
kubectl rollout restart deployment gpt2-small-public-cpu-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-2b-it-a-gpu-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-2b-it-b-gpu-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-2b-public-gpu-lite-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-2b-it-public-gpu-lite-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-9b-it-a-gpu-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-9b-it-b-gpu-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-2b-gpu-neuronpedia-inference && \
kubectl rollout restart deployment gemma-2-9b-gpu-neuronpedia-inference && \
kubectl rollout restart deployment deepseek-r1-llama-8b-a-gpu-lite-neuronpedia-inference && \
kubectl rollout restart deployment deepseek-r1-llama-8b-b-gpu-lite-neuronpedia-inference && \
kubectl rollout restart deployment deepseek-r1-llama-8b-public-a-gpu-lite-neuronpedia-inference && \
kubectl rollout restart deployment deepseek-r1-llama-8b-public-b-gpu-lite-neuronpedia-inference && \
kubectl rollout restart deployment llama-31-8b-gpu-neuronpedia-inference
```

## misc k8s commands

### delete instance

example deleting the `gemma-2-2b-it-public` instance

```
kubectl delete -k k8s/overlays/gpu/gemma-2-2b-it-public
```

### updating node pool configs

various config change examples

```
gcloud container node-pools update default-pool \
 --cluster=neuronpedia-inference \
 --min-nodes=1 \
 --max-nodes=2

gcloud container node-pools update gpu-pool \
 --cluster=neuronpedia-inference \
 --max-nodes=16
```

### preview changes for config

```
kubectl diff -k k8s/overlays/gpu/gemma-2-2b-it-a
```

### "ssh" into the container

```
kubectl exec -it deployment/gemma-2-2b-it-a-gpu-neuronpedia-inference -- bash
```

### describe container

this is useful for for debugging launch errors

```
kubectl describe pod gpt2-small-cpu-neuronpedia-inference
```
