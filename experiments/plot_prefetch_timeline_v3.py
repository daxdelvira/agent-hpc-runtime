"""
Figure v3: identical to v2 but with larger font sizes throughout.
Outputs: results/figure_prefetch_timeline_v3.pdf and .png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_prefetch_timeline_v2 import main

if __name__ == "__main__":
    main(
        suffix="v3",
        font_sizes=dict(title=15, label=13, tick=12, annot=11.5, small=10.5),
        figsize=(17, 11),
    )
