#!/usr/bin/env bash
set -u
source /mnt/d/疲劳寿命/scripts/_env.sh
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
python -u data/_probe_fq_root.py 2>&1
