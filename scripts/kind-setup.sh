#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-ml-platform}"
IMAGE_NAME="${IMAGE_NAME:-ml-deployment-platform:latest}"

echo "=== ML Deployment Platform — Kind Cluster Setup ==="

if ! command -v kind &> /dev/null; then
    echo "Error: 'kind' is not installed. Install from https://kind.sigs.k8s.io/"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "Error: 'kubectl' is not installed."
    exit 1
fi

if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "Cluster '${CLUSTER_NAME}' already exists. Deleting..."
    kind delete cluster --name "${CLUSTER_NAME}"
fi

echo "Creating Kind cluster: ${CLUSTER_NAME}"
kind create cluster --name "${CLUSTER_NAME}" --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
        protocol: TCP
      - containerPort: 443
        hostPort: 443
        protocol: TCP
  - role: worker
  - role: worker
EOF

echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=120s

echo "Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml 2>/dev/null || true

echo "Waiting for ingress controller..."
kubectl wait --namespace ingress-nginx \
  --for=condition=Ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s 2>/dev/null || echo "Ingress controller may still be starting..."

if docker image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "Loading Docker image into Kind cluster..."
    kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"
else
    echo "Warning: Docker image '${IMAGE_NAME}' not found locally. Build it first."
fi

echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

echo "Waiting for deployment rollout..."
kubectl rollout status deployment/ml-serving-api --timeout=120s 2>/dev/null || echo "Deployment may still be rolling out..."

echo ""
echo "=== Setup Complete ==="
echo "Cluster: ${CLUSTER_NAME}"
echo "API endpoint: http://ml-serving.local (add to /etc/hosts: 127.0.0.1 ml-serving.local)"
echo ""
echo "Useful commands:"
echo "  kubectl get pods"
echo "  kubectl logs -l app=ml-serving-api"
echo "  kind delete cluster --name ${CLUSTER_NAME}"
