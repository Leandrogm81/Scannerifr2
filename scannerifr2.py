"""Compat entrypoint for Streamlit Cloud.

Streamlit Cloud in this repo is configured to run `scannerifr2.py` as the main module.
The real app lives in `ifr2_app.py`, so we import it here.
"""

from ifr2_app import *  # noqa: F401,F403
