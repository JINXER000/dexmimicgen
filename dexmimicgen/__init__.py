# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import sys
robosuite_path = "/home/user/yzchen_ws/imitation_learning/robosuite/"
sys.path.append(robosuite_path)  # add robosuite root to path


from dexmimicgen.environments.two_arm_box_cleanup import TwoArmBoxCleanup
from dexmimicgen.environments.two_arm_can_sort import (
    TwoArmCanSortBlue,
    TwoArmCanSortRandom,
    TwoArmCanSortRed,
)
from dexmimicgen.environments.two_arm_coffee import TwoArmCoffee
from dexmimicgen.environments.two_arm_drawer_cleanup import (
    TwoArmDrawerCleanup,
)
from dexmimicgen.environments.two_arm_lift_tray import TwoArmLiftTray
from dexmimicgen.environments.two_arm_pouring import TwoArmPouring
from dexmimicgen.environments.two_arm_threading import TwoArmThreading
from dexmimicgen.environments.two_arm_three_piece_assembly import (
    TwoArmThreePieceAssembly,
)
from dexmimicgen.environments.two_arm_transport import TwoArmTransport



__version__ = "0.1.0"
