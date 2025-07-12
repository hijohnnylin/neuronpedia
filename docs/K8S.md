# Neuronpedia K8S Project - kubectl Cheat Sheet

This document provides a quick reference for common `kubectl` commands.

All examples will assume a namespace ID of `neuronpedia`. Note that your namespace may be named differently.

## Namespace Management

```bash
# Create the namespace if it doesn't exist
kubectl create namespace neuronpedia --dry-run=client -o yaml | kubectl apply -f -

# Set default namespace for current context
kubectl config set-context --current --namespace=neuronpedia

# List resources in the namespace
kubectl get all -n neuronpedia
```

## Deployment Management

```bash
# Deploy using kustomize (use appropriate overlay)
kubectl apply -k k8s/overlays/development/
kubectl apply -k k8s/overlays/development-gemma/
kubectl apply -k k8s/overlays/development-gpt2/
kubectl apply -k k8s/overlays/development-deepseek/

# Get status of all deployments
kubectl get deployments -n neuronpedia

# Describe a specific deployment
kubectl describe deployment webapp -n neuronpedia

# Restart a deployment
kubectl rollout restart deployment webapp -n neuronpedia

# Scale a deployment
kubectl scale deployment webapp --replicas=2 -n neuronpedia

# Delete a deployment
kubectl delete deployment webapp -n neuronpedia
```

## Database Operations

```bash
# Check postgres StatefulSet
kubectl get statefulset postgres -n neuronpedia

# Check postgres PVC
kubectl get pvc -n neuronpedia

# Delete postgres StatefulSet (preserving PVCs)
kubectl delete statefulset postgres -n neuronpedia --cascade=orphan

# Apply updated StatefulSet
kubectl apply -f k8s/base/statefulsets/postgres.yaml

# Check postgres logs
kubectl logs postgres-0 -n neuronpedia

# Execute command in postgres pod
kubectl exec -it postgres-0 -n neuronpedia -- psql -U postgres
```

## Secret & ConfigMap Management

```bash
# List all secrets
kubectl get secrets -n neuronpedia

# View a secret's contents
kubectl get secret db-credentials -n neuronpedia -o yaml

# List all configmaps
kubectl get configmaps -n neuronpedia

# Create db credentials secret manually
kubectl create secret generic postgres-credentials \
  --from-literal=POSTGRES_USER=postgres \
  --from-literal=POSTGRES_PASSWORD=postgres123 \
  --from-literal=POSTGRES_DB=postgres \
  --from-literal=POSTGRES_URL_NON_POOLING="postgres://postgres:postgres123@postgres:5432/postgres" \
  --from-literal=POSTGRES_PRISMA_URL="postgres://postgres:postgres123@postgres:5432/postgres?pgbouncer=true&connect_timeout=15" \
  -n neuronpedia

# Update a configmap
kubectl create configmap app-config \
  --from-literal=USE_LOCALHOST_INFERENCE=false \
  --from-literal=USE_LOCALHOST_AUTOINTERP=false \
  -n neuronpedia \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Troubleshooting

```bash
# View pod logs
kubectl logs webapp-xxxxx-yyyyy -n neuronpedia

# View container logs in a multi-container pod
kubectl logs webapp-xxxxx-yyyyy -c webapp -n neuronpedia

# View previous container logs (if container restarted)
kubectl logs webapp-xxxxx-yyyyy -p -n neuronpedia

# Describe pod for events and status
kubectl describe pod webapp-xxxxx-yyyyy -n neuronpedia

# Get detailed info about a resource
kubectl get pod webapp-xxxxx-yyyyy -n neuronpedia -o yaml

# Check events in the namespace
kubectl get events -n neuronpedia --sort-by='.lastTimestamp'

# Port-forward to a service for local testing
kubectl port-forward svc/webapp 3000:3000 -n neuronpedia

# Shell into a container
kubectl exec -it webapp-xxxxx-yyyyy -n neuronpedia -- /bin/bash
```

## Job Management

```bash
# List all jobs
kubectl get jobs -n neuronpedia

# Get job details
kubectl describe job db-init -n neuronpedia

# Delete a job (important for jobs with immutable specs)
kubectl delete job db-init -n neuronpedia --ignore-not-found=true

# View job logs
kubectl logs job/db-init -n neuronpedia

# Create a job from YAML file
kubectl apply -f k8s/base/jobs/db-init.yaml

# Force delete stuck job
kubectl patch job db-init -n neuronpedia -p '{"spec":{"ttlSecondsAfterFinished":0}}' --type=merge
```

## Service Management

```bash
# List all services
kubectl get svc -n neuronpedia

# Test service resolution (from another pod)
kubectl exec -it webapp-xxxxx-yyyyy -n neuronpedia -- nslookup postgres.neuronpedia.svc.cluster.local

# Describe a service
kubectl describe svc postgres -n neuronpedia

# Check service endpoints
kubectl get endpoints postgres -n neuronpedia
```

## Cleanup Operations

```bash
# Delete everything in the namespace
kubectl delete all --all -n neuronpedia

# Delete specific resource types
kubectl delete deployments,statefulsets,jobs --all -n neuronpedia

# Delete configmaps and secrets
kubectl delete configmaps,secrets --all -n neuronpedia

# Delete entire namespace (be careful!)
kubectl delete namespace neuronpedia
```

## Resource Force Deletion

```bash
# Force delete stuck pod
kubectl delete pod webapp-xxxxx-yyyyy -n neuronpedia --force --grace-period=0

# Remove finalizers from a stuck resource
kubectl patch pod webapp-xxxxx-yyyyy -n neuronpedia -p '{"metadata":{"finalizers":null}}' --type=merge
```

## Applying Changes with Kustomize

```bash
# View what would be applied without actually applying
kubectl kustomize k8s/overlays/development/

# Apply with kustomize
kubectl apply -k k8s/overlays/development/

# Delete with kustomize
kubectl delete -k k8s/overlays/development/
```