from setuptools import setup, find_packages

with open("README.MD", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="Resemblyzer",
    version="0.0.1-dev",
    packages=find_packages(),
    package_data={
        "resemblyzer": ["resemblyzer/pretrained.pt"]
    },
    install_requires=requirements,
    author="Corentin Jemine",
    author_email="corentin@resemble.ai",
    description="Analyze and compare voices with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/resemble-ai/Resemblyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
