from setuptools import setup, find_packages

setup(
    name='sentiment_intention_analysis',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'transformers',
        'torch',
        'configparser',
        'json'
    ],
    entry_points={
        'console_scripts': [
            'sentiment_intention_analysis=your_module_name_here.main:main',
        ],
    },
    author='Kubra Buzlu',
    author_email='kubraa.buzlu@email.com',
    description='Sentiment and Intention Analysis Package',
    url='https://github.com/kubrabuzlu/SentimentAndIntentionAnalysis',
)
