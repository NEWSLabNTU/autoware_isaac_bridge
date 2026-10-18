from glob import glob

from setuptools import find_packages, setup

package_name = "aib"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/scripts", glob("scripts/*.sh")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NEWSLab NTU",
    maintainer_email="jerry73204@gmail.com",
    description="An Autoware ego on Isaac Sim: vehicle, sensors, vehicle interface; spec generator",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "aib_generate = aib.gen:main",
        ],
    },
)
