"""
Figure v4: identical to v2 but with white background.
Outputs: results/figure_prefetch_timeline_v4.pdf and .png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_prefetch_timeline_v2 import main

if __name__ == "__main__":
    main(suffix="v4", dark_bg=False)
