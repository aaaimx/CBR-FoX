from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements(filename):
    return [
        line.strip()
        for line in Path(filename).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="CBR-FoX",
    version="0.3.0",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    install_requires=parse_requirements("requirements.txt"),
    packages=find_packages(),
)