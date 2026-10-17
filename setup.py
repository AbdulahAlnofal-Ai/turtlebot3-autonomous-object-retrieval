from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        #(os.path.join('share', package_name, 'models/my_maze'), glob('models/my_maze/*')),
        (os.path.join('share', package_name, 'models/object_block'), glob('models/object_block/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hassan',
    maintainer_email='hassan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'wall_detector = pkg.wall_detector:main','wall_detector_2 = pkg.wall_detector_2:main',
            'mission_manager = pkg.mission_manager:main',
            'object_detector = pkg.object_detector:main',
            'camera_object_detector = pkg.camera_object_detector:main',
            'explorer = pkg.explorer:main',
        ],
    },
)
