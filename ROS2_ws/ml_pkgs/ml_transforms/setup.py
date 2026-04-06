from setuptools import find_packages, setup

package_name = 'ml_transforms'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Newton Campbell',
    maintainer_email='newtonh20@ieee.org',
    description=(
        'Minimal SE3 / quaternion transform library for BAM. '
        'ROS2 Jazzy ament_python package.'
    ),
    license='NASA Open Source Agreement 1.3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
