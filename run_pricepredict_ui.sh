#!/usr/bin/bash

# Import LLM keys
. ~/.ssh/LLM_envs.sh

export PYTHONPATH=lib:${PYTHONPATH}
export PYTHONUNBUFFERED=1

streamlit run dgs_pred_ui.py
