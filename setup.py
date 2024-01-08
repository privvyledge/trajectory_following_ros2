from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_following_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Boluwatife Olabiran',
    maintainer_email='humble@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coupled_kinematic_casadi = trajectory_following_ros2.coupled_kinematic_casadi:main',
            'coupled_kinematic_do_mpc = trajectory_following_ros2.coupled_kinematic_do_mpc:main',
            'coupled_kinematic_acados = trajectory_following_ros2.coupled_kinematic_acados:main',
            'kinematic_dompc_simulator = trajectory_following_ros2.simulator.do_mpc_simulator_node:main',
            'twist_to_ackermann = trajectory_following_ros2.twist_to_ackermann_drive:main',
            'waypoint_recorder = trajectory_following_ros2.waypoint_recorder:main',
            'waypoint_loader = trajectory_following_ros2.waypoint_loader:main',
        ],
    },
)