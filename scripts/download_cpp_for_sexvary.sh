#!/usr/bin/env bash
set -euo pipefail

CPP_RELEASE_TAG="${CPP_RELEASE_TAG:-v3.2}"
DEST_DIR="${1:-data/sources/nih/cpp/release_v3_2}"
BASE_URL="https://github.com/ljlasker/CollaborativePerinatalProject/releases/download/${CPP_RELEASE_TAG}"

mkdir -p "${DEST_DIR}"

FILES=(
  "cpp_clean_v1.csv"
  "cpp_cognitive_scores.csv"
  "cpp_g_factors.csv"
  "cpp_weights.csv"
  "cpp_clean_v1_codebook.csv"
  "README.md"
  "RELEASE_NOTES.md"
  "METHODOLOGY.md"
  "cpp_growth_trajectories.csv"
  "cpp_birthweight_zscores.csv"
  "cpp_kinship_links.csv"
  "cpp_twin_zygosity.csv"
  "cpp_weights_codebook.csv"
  "CPP_Codebook.csv"
)

echo "CPP release tag: ${CPP_RELEASE_TAG}"
echo "Downloading to: ${DEST_DIR}"

for file in "${FILES[@]}"; do
  url="${BASE_URL}/${file}"
  out="${DEST_DIR}/${file}"
  echo "-> ${file}"
  curl -L --fail --retry 3 --retry-delay 2 -o "${out}" "${url}"
done

inventory="${DEST_DIR}/FILE_INVENTORY.csv"
{
  echo "filename,url"
  for file in "${FILES[@]}"; do
    echo "${file},${BASE_URL}/${file}"
  done
} > "${inventory}"

cat > "${DEST_DIR}/SOURCE.md" <<EOF
# CPP source notes

Downloaded from the public CPP data release assets.

Base release URL:
${BASE_URL}

The release tag was set by:
CPP_RELEASE_TAG=${CPP_RELEASE_TAG}

If the public site has moved to a newer release tag, update the variable and rerun.
EOF

echo "Done."
echo "Inventory written to: ${inventory}"
