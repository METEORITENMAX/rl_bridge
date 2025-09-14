from setuptools import find_packages, setup

package_name = 'rl_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='m.kirsch@fh-aachen.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gen3_rl_jointPos = rl_bridge.gen3_rl_jointPos:main',
            'test_ik = rl_bridge.test_ik:main',
            'test_jointPos = rl_bridge.test_jointPos:main'
        ],
    },
)
