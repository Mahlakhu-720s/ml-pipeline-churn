from setuptools import setup, find_packages

setup(
    name="ml-pipeline-churn",
    version="1.0.0",
    description="End-to-end MLOps pipeline for customer churn prediction",
    author="AI Engineering",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "mlflow>=2.10",
        "fastapi>=0.111",
        "pydantic>=2.0",
    ],
)
