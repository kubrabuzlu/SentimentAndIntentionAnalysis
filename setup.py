from setuptools import setup, find_packages

setup(
    name="sentiment-intention-analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "configparser",
        "transformers",
        "fastapi",
        "uvicorn",
        "pydantic"
    ],
    entry_points={
        'console_scripts': [
            'sentiment-intention-analysis=SentimentIntentionAnalysis.src.api:app'
        ]
    }
)
