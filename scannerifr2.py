"""Compat entrypoint for Streamlit Cloud.

Streamlit Cloud in this repo is configured to run `scannerifr2.py` as the main module.
The real app lives in `ifr2_app.py`, so we import it here.
"""

import streamlit as st
st.write("DEBUG: scannerifr2.py entrypoint reached")

from ifr2_app import *  # noqa: F401,F403
