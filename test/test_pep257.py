# Copyright 2015 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    pkg_root = os.path.join(os.path.dirname(__file__), '..')
    excludes = [
        os.path.abspath(os.path.join(pkg_root, p)) for p in [
            'trajectory_following_ros2/old',
            'trajectory_following_ros2/cvxpy',
            'trajectory_following_ros2/purepursuit/old',
            'trajectory_following_ros2/utils/autoware_auto_mpc_utils.py',
            'trajectory_following_ros2/utils/autoware_universe_mpc_utils.py',
            'trajectory_following_ros2/utils/TrajectoryOld.py',
            'trajectory_following_ros2/utils/Trajectory.py',
            'main.py',
            'temp.py',
        ]
    ]
    extra_ignore = (
        'D200,D202,D204,D205,D208,D209,D210,'
        'D213,D214,D300,D400,D401,D402,D403,'
        'D406,D407,D409,D411,D413,D414,D415,D417'
    )
    rc = main(argv=['--add-ignore', extra_ignore, '--exclude'] + excludes + ['.', 'test'])
    assert rc == 0, 'Found code style errors / warnings'
