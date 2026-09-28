"""Manuscript figure entry points and shared rendering code."""

import logging


# Matplotlib logs one message for every text element when a requested font is
# unavailable. Keep its normal font fallback behavior, but suppress that noisy
# diagnostic in manuscript rendering jobs.
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
