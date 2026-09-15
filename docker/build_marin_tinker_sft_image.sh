#!/usr/bin/env bash
# Append the Marin recipe to the pinned upstream image without downloading or
# rebuilding its large CUDA layers.
set +x
set -euo pipefail

: "${GITSHA:?}"
: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"
[[ "$GITSHA" =~ ^[0-9a-f]{40}$ ]] || { echo "GITSHA must be a full lowercase commit" >&2; exit 2; }

BASE_IMAGE=docker.io/axolotlai/axolotl-uv@sha256:ee66d1b20b1f308996857e3a08b8b20903f9c8df827bc8b50e37ccb7b9215fbd
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-ghcr.io/marin-community/axolotl}"
IMAGE_WORKDIR=workspace/axolotl
REGISTRY_HOST="${IMAGE_REPOSITORY%%/*}"
[ "$REGISTRY_HOST" != "$IMAGE_REPOSITORY" ] || { echo "IMAGE_REPOSITORY must include a registry host" >&2; exit 2; }

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LAYER_ROOT=$(mktemp -d)
CRANE_ROOT=$(mktemp -d)
trap 'rm -rf "$LAYER_ROOT" "$CRANE_ROOT"' EXIT

install -D -m 0644 \
  "$REPOSITORY_ROOT/examples/marin/tinker-openthoughts3-sft.yaml" \
  "$LAYER_ROOT/$IMAGE_WORKDIR/examples/marin/tinker-openthoughts3-sft.yaml"
install -D -m 0644 \
  "$REPOSITORY_ROOT/scripts/__init__.py" \
  "$LAYER_ROOT/$IMAGE_WORKDIR/scripts/__init__.py"
while IFS= read -r source; do
  relative=${source#"$REPOSITORY_ROOT/"}
  install -D -m 0644 "$source" "$LAYER_ROOT/$IMAGE_WORKDIR/$relative"
done < <(find "$REPOSITORY_ROOT/scripts/marin_experiments" -type f -name '*.py' -print | sort)

tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -czf "$CRANE_ROOT/recipe-layer.tar.gz" -C "$LAYER_ROOT" "${IMAGE_WORKDIR%%/*}"
curl -fsSL \
  https://github.com/google/go-containerregistry/releases/download/v0.20.2/go-containerregistry_Linux_x86_64.tar.gz \
  -o "$CRANE_ROOT/crane.tgz"
tar -xzf "$CRANE_ROOT/crane.tgz" -C "$CRANE_ROOT" crane
chmod 0755 "$CRANE_ROOT/crane"

export DOCKER_CONFIG="$CRANE_ROOT/docker-config"
printf '%s' "$REGISTRY_TOKEN" | "$CRANE_ROOT/crane" auth login "$REGISTRY_HOST" \
  --username "$REGISTRY_USER" --password-stdin
unset REGISTRY_TOKEN

exec "$CRANE_ROOT/crane" mutate --platform linux/amd64 "$BASE_IMAGE" \
  --append "$CRANE_ROOT/recipe-layer.tar.gz" \
  --env "AXOLOTL_SOURCE_COMMIT=$GITSHA" \
  --label "org.opencontainers.image.source=https://github.com/marin-community/axolotl" \
  --label "org.opencontainers.image.revision=$GITSHA" \
  --label "org.opencontainers.image.title=Marin Tinker SFT reproduction" \
  --workdir "/$IMAGE_WORKDIR" \
  --tag "${IMAGE_REPOSITORY}:tinker-sft-${GITSHA}"
