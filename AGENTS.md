
# Agent Instructions

This file provides instructions for agents on how to set up the environment and run tests for this project.

## Environment Setup

1. **Install Dependencies**: This project uses Poetry for dependency management. To install the required dependencies, run the following command:

   ```bash
   pip install -e .[dev]
   ```

   If you encounter network issues, you may need to install the dependencies manually. The required dependencies are listed in the `pyproject.toml` file.

2. **Set `PYTHONPATH`**: To ensure that the `parsfet` module can be found, you must set the `PYTHONPATH` environment variable to include the `src` directory:

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   ```

## Running Tests

Once the environment is set up, you can run the tests using the following command:

```bash
python3 -m unittest discover tests
```
