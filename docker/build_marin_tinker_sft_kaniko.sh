#!/usr/bin/env bash
set +x
set -euo pipefail

: "${GITSHA:?}"
: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"
[[ "$GITSHA" =~ ^[0-9a-f]{40}$ ]] || { echo "GITSHA must be a full lowercase commit" >&2; exit 2; }
[ "$(uname -m)" = x86_64 ] || { echo "The reproduction image requires an x86_64 builder" >&2; exit 2; }
[ -f /app/docker/Dockerfile-marin-tinker-sft ] || { echo "The Axolotl checkout must be mounted at /app" >&2; exit 2; }

IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-ghcr.io/marin-community/axolotl}"
REGISTRY_HOST="${IMAGE_REPOSITORY%%/*}"
[ "$REGISTRY_HOST" != "$IMAGE_REPOSITORY" ] || { echo "IMAGE_REPOSITORY must include a registry host" >&2; exit 2; }

cd /tmp
curl -fsSL \
  https://github.com/google/go-containerregistry/releases/download/v0.20.2/go-containerregistry_Linux_x86_64.tar.gz \
  -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane
crane export --platform linux/amd64 gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true
test -x /kaniko/executor

DOCKER_CONFIG=/kaniko/.docker
install -d -m 0700 "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$REGISTRY_USER" "$REGISTRY_TOKEN" | base64 | tr -d '\n')
printf '{"auths":{"%s":{"auth":"%s"}}}\n' "$REGISTRY_HOST" "$AUTH" > "$DOCKER_CONFIG/config.json"
chmod 0600 "$DOCKER_CONFIG/config.json"
unset AUTH REGISTRY_TOKEN

exec env DOCKER_CONFIG="$DOCKER_CONFIG" /kaniko/executor \
  --context dir:///app \
  --dockerfile /app/docker/Dockerfile-marin-tinker-sft \
  --build-arg GITSHA="$GITSHA" \
  --cache=true \
  --cache-repo="${IMAGE_REPOSITORY}/cache-tinker-sft" \
  --image-fs-extract-retry=3 \
  --image-download-retry=3 \
  --push-retry=3 \
  --destination "${IMAGE_REPOSITORY}:tinker-sft-${GITSHA}"
