from setuptools import find_packages, setup

package_name = 'coordinate_converter'

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
    maintainer='gianluca',
    maintainer_email='gianlucaeremita.03@gmali.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "home_joint_final = coordinate_converter.home_joint_final:main",
            "home_joint_trajectory = coordinate_converter.home_joint_trajectory:main",
            "spline_separata = coordinate_converter.spline_separata:main",
            "spline_separata_finale = coordinate_converter.spline_separata_finale:main"
        ],
    },
)
