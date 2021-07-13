from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6",
                'spacy',
                'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz']

setup(
    name="e2e Ml",
    version="0.0.1",
    author="Thomas Meißner",
    author_email="meissnercorporation@gmx.de",
    description="An end to end solution for automl. ",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasMeissnerDS/e2e_ml",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)