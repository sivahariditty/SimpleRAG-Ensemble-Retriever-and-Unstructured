from setuptools import find_packages, setup

setup(
    name="python-rag",
    version="1.0.0",
    author="lodgify.com",
    author_email="sohravb@lodgify.com, denis.ergashbaev@lodgify.com",
    packages=find_packages(),
    test_suite="test",
    install_requires=[
        "wheel",
        "pandas==2.2.*",
        "transformers==4.43.*",
        "ragas==0.1.*",
        "pypdf==4.3.*",
        "python-dotenv==1.0.*",
        "langchain==0.2.*",
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
        "pytest-timeout",
    ],
    extras_require={
        'test': [
            "pytest",
            "pytest-timeout",
        ],
    },
)