#!/usr/bin/env bash
# Build the portable wheel and publish it as a GitHub Release on
# deepinfra/kv-local-indexer.
#
# Prereqs: docker + network, and `gh` logged in (run `gh auth login` once; a
# private repo's release assets need a token, an SSH key is not enough).
# The wheel is portable, so this needs nothing from localhost:30500.
#
# Usage: release.sh <version-tag>      e.g.  release.sh v0.1.0
set -euo pipefail

VER="${1:?usage: release.sh <version-tag>   e.g.  release.sh v0.1.0}"
REPO="deepinfra/kv-local-indexer"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${HERE}/build_wheel.sh"
WHEEL="$(ls -1t "${HERE}/dist"/*.whl | head -1)"

echo "Publishing ${WHEEL##*/} as ${VER} on ${REPO}"
if gh release view "${VER}" --repo "${REPO}" >/dev/null 2>&1; then
  gh release upload "${VER}" "${WHEEL}" --repo "${REPO}" --clobber
else
  gh release create "${VER}" "${WHEEL}" --repo "${REPO}" \
    --title "${VER}" --notes "kv_local_indexer ${VER} (portable cp312 wheel)"
fi
echo "Done. Verify: gh release view ${VER} --repo ${REPO}"
