#!/usr/bin/env bash
set -euo pipefail

# Provision standalone MSGFPlus.jar and free2bind inside dependencies/ when not using Bioconda.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${REPO_ROOT}/dependencies"

MSGF_VERSION="${MSGF_VERSION:-v2024.03.26}"
MSGF_URL="https://github.com/MSGFPlus/msgfplus/releases/download/${MSGF_VERSION}/MSGFPlus_v20240326.zip"

mkdir -p "${DEPS_DIR}/MSGFPlus"

if [[ ! -f "${DEPS_DIR}/MSGFPlus/MSGFPlus.jar" ]]; then
  echo "Downloading MS-GF+ (${MSGF_VERSION})..."
  TMP_ZIP="$(mktemp /tmp/msgfplus.XXXXXX.zip)"
  curl -fsSL "${MSGF_URL}" -o "${TMP_ZIP}"
  python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
    "${TMP_ZIP}" "${DEPS_DIR}/MSGFPlus"
  rm -f "${TMP_ZIP}"
fi

echo "MSGFPlus.jar ready at: ${DEPS_DIR}/MSGFPlus/MSGFPlus.jar"
echo "Tip: export UPROTEINS_MSGF_JAR=${DEPS_DIR}/MSGFPlus/MSGFPlus.jar or use conda env create -f environment.yml"
