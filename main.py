# region Imports

import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')
# Configure logging to completely disable speechbrain logging
logging.getLogger('speechbrain').setLevel(logging.CRITICAL)
logging.getLogger('speechbrain').propagate = False
logging.getLogger('speechbrain').disabled = True
# Suppress unwanted messages
logging.getLogger("pyannote").setLevel(logging.CRITICAL)
logging.getLogger("nicegui").setLevel(logging.CRITICAL)
logging.getLogger("watchfiles").setLevel(logging.CRITICAL)

from src.gui_maingui import MainGUI

# endregion Imports


# region Runtime

if __name__ == "__main__":
    gui = MainGUI()
    gui.run()

# endregion Runtime
