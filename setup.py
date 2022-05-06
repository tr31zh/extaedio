import pathlib
from setuptools import find_packages, setup
import sys

if sys.platform == "win32":
    script_ = "scripts/launch.bat"
else:
    script_ = "scripts/launch.sh"


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="extaedio",
    version="0.9.2.134",
    description="Dataframe plotter",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/tr31zh/ask_me_polotly",
    author="Felicia Antoni Maviane Macia",
    author_email="maviane@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    platforms=["Any"],
    keywords="plot dataframe visualization",
    packages=find_packages(exclude=("doc")),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "dash",
        "sklearn",
        "watchdog",
    ],
    entry_points={
        "console_scripts": [
            "extaedio=extaedio:extaedio",
        ]
    },
)
