# Kubernetes Deployment Guide for Neuronpedia

This guide will help you deploy Neuronpedia on Kubernetes using `kubectl` and `kustomize`. We've designed a modular component-based architecture that makes it easy to customize your deployment for different environments and models.

## How This Works

Instead of managing monolithic configuration files, we break everything into reusable "components" that you can mix and match. Think of components as building blocks - you pick the ones you need and combine them in an "overlay" to create your specific deployment.

For example, you might combine:
- A Harbor registry component (to pull images from your private registry)
- A GPU-standard resource component (for models that need more compute)
- A DeepSeek model component (with model-specific settings)
- Your organization's config component (with your API keys and settings)

## Directory Structure

```
k8s/
├── base/                           # Core Kubernetes manifests (deployments, services, etc.)
│   ├── config/                     # Base configuration files
│   ├── deployments/                # Application pod definitions
│   ├── jobs/                       # Database setup jobs
│   ├── networking/                 # Ingress rules for external access
│   ├── services/                   # Internal service definitions
│   └── statefulsets/               # PostgreSQL database
├── components/                     # Reusable building blocks you can mix and match
│   ├── config/
│   │   ├── app/                    # Webapp settings and API keys
│   │   ├── db/                     # Database connection info
│   │   └── inference/              # Inference service secrets
│   ├── corporate-ca-certs/         # For corporate networks (optional)
│   ├── models/                     # Model-specific configurations
│   │   ├── deepseek*/              # DeepSeek model variants
│   │   ├── llama*/                 # Llama model variants
│   │   ├── gemma*/                 # Gemma model variants
│   │   └── gpt2*/                  # GPT-2 variants
│   ├── registry/                   # Docker registry configurations
│   │   ├── dockerhub/              # For pulling from DockerHub
│   │   └── harbor/                 # For private Harbor registries
│   └── resources/                  # Hardware resource profiles
│       ├── inference-gpu-lite/     # Lighter GPU requirements
│       └── inference-gpu-standard/ # Standard GPU requirements
└── overlays/                       # Your actual deployments (combines components)
    └── your-custom-overlay/
```

## Quick Start

Once you've configured everything (see steps below), deploying is straightforward:

```bash
kubectl apply -k k8s/overlays/your-custom-overlay/
```

Need help with kubectl commands? Check out our [K8s Operations Cheat Sheet](../docs/K8S.md) for common operations and troubleshooting.

## Understanding the Components

### Config Components: Your Organization's Settings

These components contain the environment variables and secrets that your organization will use across all deployments. You'll typically set these once and reuse them everywhere.

- **`config/app`**: Contains webapp settings like your domain URL, API keys for OpenAI/Anthropic, OAuth credentials, and other application secrets
- **`config/db`**: Database connection details and credentials
- **`config/inference`**: Secrets needed by the inference service, like Hugging Face tokens

### Key Configuration Objects

Understanding how configuration is organized helps you know where to put different types of settings. Anything with `*_API_KEY`, `*_SECRET`, or OAuth credentials should go in `app-secrets`, while public configuration and feature flags belong in `app-config`:

| Name | Type | Purpose | Key Variables |
|------|------|---------|---------------|
| `app-config` | ConfigMap | Non-sensitive webapp configuration | `NEXT_PUBLIC_*`, `USE_LOCALHOST_*`, feature flags |
| `inference-config` | ConfigMap | Model-specific inference parameters | `MODEL_ID`, `SAE_SETS`, `MAX_LOADED_SAES` |
| `db-credentials` | Secret | PostgreSQL connection strings | `POSTGRES_*` variables |
| `app-secrets` | Secret | API keys and authentication secrets | `*_API_KEY`, `*_SECRET`, OAuth credentials |

### Model Components: Model-Specific Configuration

Each model component handles two important things:
1. **Naming and labels**: Ensures you can run multiple models in the same Kubernetes namespace without conflicts
2. **Model configuration**: Sets the specific environment variables each model needs (like which Hugging Face model to load, memory limits, etc.)

For example, the `deepseek-r1-distill-llama-8b` component will name your inference service something like `deepseek-r1-distill-llama-8b-inference` and configure it to load the DeepSeek model with appropriate settings.

### Registry Components: Where to Pull Docker Images

These components tell Kubernetes where to find the Neuronpedia Docker images. You'll need one of these because the base manifests use generic image names like `neuronpedia-webapp`, but your actual images are stored in a specific registry.

- **`registry/harbor`**: For private Harbor registries (common in enterprise environments)
- **`registry/dockerhub`**: For public DockerHub images

### Resource Components: Hardware Requirements

Different models need different amounts of CPU, memory, and GPU resources. These components define resource profiles:

- **`resources/inference-gpu-lite`**: For smaller models or when GPU resources are limited
- **`resources/inference-gpu-standard`**: For larger models that need more compute power

### Corporate CA Certs Component: For Corporate Networks

If you're deploying behind a corporate firewall, you might need to add your organization's CA certificates to the containers. This optional component handles that by mounting your CA bundle into the database containers.

## Setting Up Your Deployment

### Step 1: Configure Database Initialization

The database initialization job needs to know how to reach PostgreSQL. Open [k8s/base/jobs/db-init.yaml](./base/jobs/db-init.yaml) and update the namespace in this line:

```yaml
# Change "neuronpedia" to match your actual Kubernetes namespace
until nc -z postgres.neuronpedia.svc.cluster.local 5432; do
```

### Step 2: Set Your Domain Name

Edit [k8s/base/networking/ingress.yaml](./base/networking/ingress.yaml) to use your actual domain:

```yaml
spec:
  rules:
  - host: your-actual-domain.com  # Replace neuronpedia.org with your domain
```

This tells Kubernetes how external users will reach your Neuronpedia instance.

### Step 3: Update Your Organization's Configuration

This is where you'll spend most of your configuration time. Update the config components with your organization's specific settings:

**In `k8s/components/config/app/kustomization.yaml`:**

You'll need to update several important settings:
- **URLs**: Change `NEXT_PUBLIC_URL` and `NEXTAUTH_URL` from `http://localhost:3000` to your actual domain
- **Service connections**: Set `USE_LOCALHOST_INFERENCE` and `USE_LOCALHOST_AUTOINTERP` to `false` for Kubernetes deployment
- **Contact info**: Update `CONTACT_EMAIL_ADDRESS` to your admin email
- **Security**: Replace `NEXTAUTH_SECRET` with a secure random string
- **API keys**: Fill in any API keys you plan to use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- **OAuth**: Add your GitHub, Google, or Apple OAuth credentials if you want social login

**In `k8s/components/config/db/kustomization.yaml`:**

Replace the default password with something secure:
- Change `POSTGRES_PASSWORD` from `org-secure-password` to your actual password
- Update the connection strings to use your new password

**In `k8s/components/config/inference/kustomization.yaml`:**

Add your Hugging Face token:
- Replace `your-huggingface-token` with your actual token from huggingface.co

### Step 4: Set Up Corporate CA Certificates (If Needed)

If you're deploying in a corporate environment with custom CA certificates, you'll need to update the mount path in the [k8s/components/corporate-ca-certs patches](./base/components/corporate-ca-certs/kustomization.yaml). Look for lines like:

```yaml
mountPath: /usr/local/share/ca-certificates  # Update this to your preferred path
```

Most users can skip this step.

### Step 5: Configure Your Docker Registry

You need to tell Kubernetes where to find the Neuronpedia Docker images. Create a new registry component (or modify an existing one) in [k8s/components/registry/](./components/registry/):

```yaml
# k8s/components/registry/my-registry/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1alpha1
kind: Component

images:
  - name: neuronpedia-inference
    newName: my-registry.example.com/neuronpedia/inference
  - name: neuronpedia-webapp
    newName: my-registry.example.com/neuronpedia/webapp
  - name: neuronpedia-db-init
    newName: my-registry.example.com/neuronpedia/webapp  # db-init uses the webapp image
  - name: neuronpedia-autointerp
    newName: my-registry.example.com/neuronpedia/autointerp
```

Replace `my-registry.example.com` with your actual registry URL.

## Creating Your Deployment

Now comes the fun part - creating your actual deployment by combining components. Create a new directory under `k8s/overlays/` and add a `kustomization.yaml` file:

```yaml
# k8s/overlays/my-production-deployment/kustomization.yaml
---
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Start with the base Neuronpedia manifests
resources:
  - ../../base

# Add the components you need
components:
  - ../../components/corporate-ca-certs        # Only if you need corporate CA certs
  - ../../components/registry/my-registry      # Your registry component
  - ../../components/config/app                # Your org's app config
  - ../../components/config/db                 # Your org's database config
  - ../../components/config/inference          # Your org's inference config
  - ../../components/resources/inference-gpu-standard  # GPU profile for your model
  - ../../components/models/deepseek-r1-distill-llama-8b  # The model you want

# Override any settings specific to this deployment
configMapGenerator:
  - name: app-config
    behavior: merge
    literals:
      - NEXT_PUBLIC_URL=https://my-production-deployment.example.com
      - NEXTAUTH_URL=https://my-production-deployment.example.com
```

The beauty of this approach is that you can easily create variations. Want a staging environment with a different domain? Just copy the overlay and change the URLs. Want to try a different model? Swap out the model component.

## Deploying Specific Services

Sometimes you don't want to deploy everything at once. Kubernetes label selectors let you deploy just the parts you need:

```bash
# Deploy only the inference service (great for testing new models)
kubectl apply -k k8s/overlays/my-deployment/ --selector="app=inference"

# Deploy the web interface and inference, but skip the database
kubectl apply -k k8s/overlays/my-deployment/ --selector="app in (inference,webapp)"

# Deploy everything except the database initialization job
kubectl kustomize k8s/overlays/my-deployment/ | \
  kubectl apply -f - --field-selector="metadata.name!=db-init"
```

This is particularly useful when you want to test individual components or when you're managing the database separately.

## What Gets Deployed

Your Neuronpedia deployment includes several key services:

- **webapp**: The main Next.js web interface that users interact with
- **inference**: The service that runs your AI models and generates responses  
- **autointerp**: Provides automatic interpretation of model behaviors
- **postgres**: The PostgreSQL database (with pgvector extension for embeddings)
- **db-init**: A one-time job that sets up the database schema and initial data

All these services communicate with each other internally through Kubernetes' built-in service discovery. The ingress controller routes external traffic to the appropriate service based on the URL path.

One neat feature: you can run multiple inference services with different models in the same namespace. The model components handle the naming and labeling to prevent conflicts.

## Security Best Practices

Before deploying to production, make sure you:

- Replace all default passwords and API keys with secure values
- Use your cluster's secret management system for sensitive data
- Consider integrating with external secret management tools like HashiCorp Vault
- Review the network policies and ingress rules for your security requirements

The development tokens and passwords included in the base configuration are meant for testing only.

## Getting Help

If you run into issues, the [K8s Operations Cheat Sheet](../docs/K8S.md) has common kubectl commands and troubleshooting steps. For Neuronpedia-specific questions, check the main project documentation or reach out to the community.