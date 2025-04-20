from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from __init__.py
with open(os.path.join("abstractllm", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="abstractllm",
    version=version,
    author="Laurent-Philippe Albou",
    author_email="lpalbou@gmail.com",
    description="A unified interface for interacting with multiple LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lpalbou/abstractllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
        "pillow>=8.0.0",
        "aiohttp>=3.8.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "huggingface": [
            "transformers>=4.36.0",
            "torch>=2.0.0",
            "huggingface-hub>=0.20.0",
        ],
        "tools": [
            "docstring-parser>=0.15",
            "pydantic>=2.0.0",
            "jsonschema>=4.0.0",
        ],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "transformers>=4.36.0",
            "torch>=2.0.0",
            "huggingface-hub>=0.20.0",
            "docstring-parser>=0.15",
            "pydantic>=2.0.0",
            "jsonschema>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "abstractllm=abstractllm.cli:main",
        ],
    },
) 