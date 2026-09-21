#!/usr/bin/env bash
set -euo pipefail

if ! command -v npx >/dev/null 2>&1; then
  echo "Error: npx is required but not found on PATH." >&2
  exit 1
fi

has_session_flag="false"
has_filename_flag="false"
help_requested="false"
version_requested="false"
boolean_option=""
skip_value="false"
literal_args="false"
command_name=""
for arg in "$@"; do
  if [[ -n "${boolean_option}" ]]; then
    if [[ "${arg}" == "true" || "${arg}" == "false" ]]; then
      case "${boolean_option}" in
        help) help_requested="${arg}" ;;
        version) version_requested="${arg}" ;;
      esac
      boolean_option=""
      continue
    fi
    boolean_option=""
  fi
  if [[ "${skip_value}" == "true" ]]; then
    skip_value="false"
    continue
  fi
  if [[ "${literal_args}" == "true" ]]; then
    if [[ -z "${command_name}" ]]; then command_name="${arg}"; fi
    continue
  fi
  case "$arg" in
    --session|-s)
      has_session_flag="true"
      skip_value="true"
      ;;
    --session=*|-s=*) has_session_flag="true" ;;
    --filename) has_filename_flag="true"; skip_value="true" ;;
    --filename=*) has_filename_flag="true" ;;
    --depth) skip_value="true" ;;
    --json|--raw|--boxes) boolean_option="other" ;;
    --help|-h) help_requested="true"; boolean_option="help" ;;
    --version|-v) version_requested="true"; boolean_option="version" ;;
    --help=true) help_requested="true" ;;
    --help=false) help_requested="false" ;;
    --version=true) version_requested="true" ;;
    --version=false) version_requested="false" ;;
    --) literal_args="true" ;;
    -*) ;;
    *) if [[ -z "${command_name}" ]]; then command_name="${arg}"; fi ;;
  esac
done

cmd=(npx --yes --prefer-offline --package @playwright/cli@0.1.21 playwright-cli)
if [[ "${has_session_flag}" != "true" && -n "${PLAYWRIGHT_CLI_SESSION:-}" ]]; then
  cmd+=(--session "${PLAYWRIGHT_CLI_SESSION}")
fi
# Unlike navigation/actions, bare `snapshot` prints the entire tree inline.
# Use the documented filename option to keep tool output bounded while retaining
# the full snapshot and element refs for selective read_file calls. mktemp works
# on both macOS and Linux and prevents snapshots overwriting one another.
if [[ "${command_name}" == "snapshot" && "${has_filename_flag}" != "true" && "${help_requested}" != "true" && "${version_requested}" != "true" ]]; then
  mkdir -p output/playwright
  snapshot_file="$(mktemp output/playwright/snapshot-XXXXXXXX)"
  # Put the option before caller arguments so an explicit `--` cannot hide it.
  cmd+=("--filename=${snapshot_file}")
fi
cmd+=("$@")

exec "${cmd[@]}"
