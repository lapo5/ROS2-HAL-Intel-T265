from setuptools import setup

package_name = 'hal_intel_t265'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marco',
    maintainer_email='marco.lapolla5@gmail.com',
    description='HAL for Intel Realsense T265',
    license='BSD',
    entry_points={
        'console_scripts': [
                'hal_intel_t265 = hal_intel_t265.ros2_hal_intel_t265:main',
        ],
},
)
