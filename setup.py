from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6",
                "boostaroota==1.3",
                "category_encoders==2.2.2",
                "imblearn==0.0",
                "ipython>=7.10.0",
                "jupyter_core==4.7.0",
                "lightgbm==3.2.1",
                "matplotlib==3.3.4",
                "ngboost==0.3.10",
                "nltk==3.6.2",
                "numpy==1.19.4",
                "optuna==2.5.0",
                "pandas==1.1.5",
                "pip==21.1.3",
                "plotly==5.1.0",
                "psutil==5.8.0",
                "seaborn==0.11.1",
                "scikit-learn==0.23.2",
                "scipy==1.6.3",
                "setuptools >= 51.1.0",
                "shap==0.39.0",
                "spacy==3.0.6",
                "wheel==0.36.2",
                "xgboost==1.3.3",
                ]

setup(
    name="e2eml",
    version="0.9.6",
    author="Thomas Meißner",
    author_email="meissnercorporation@gmx.de",
    description="An end to end solution for automl.",
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