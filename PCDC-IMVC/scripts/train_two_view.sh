#!/usr/bin/env bash
set -euo pipefail

python -m pcdc_imvc.cli.train --mode two "$@"