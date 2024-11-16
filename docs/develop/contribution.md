# Contributing to the Project

We welcome contributions to this project and appreciate your interest in improving the library! Please follow these guidelines to ensure a smooth and productive collaboration.

---

## How to Contribute

- **Fork the Repository**  
   Create a fork of the repository to your GitHub account.

- **Clone Your Fork**  
   Clone your fork to your local machine:

  ```bash
  git clone https://github.com/kyrylo-gr/ffit.git
  cd ffit
  ```

- **Create a Branch**  
   Create a new branch for your feature or bugfix:

  ```bash
  git checkout -b feature-or-bugfix-name
  ```

- **Install Dependencies**  
   Set up the development environment by installing the required dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Run Linters and Formatters

Before submitting your changes, ensure your code follows the project's style and quality standards:

- Format code using [Black](https://github.com/psf/black):
  ```bash
  black .
  ```
- Lint code using [Ruff](https://github.com/charliermarsh/ruff):
  ```bash
  ruff .
  ```
- Additionally, run Flake8 with the following configuration:
  ```bash
  flake8 . --count --max-complexity=10 --max-line-length=127 --ignore=E731,E741,E203,E265,E226,C901,W504,W503,E704
  ```

## Run Tests

Ensure all tests pass:

```bash
pytest
```

## Create Documentation

The documentation is generated automatically from the docstrings in the code. It should be updated if any new functions are added. To update the documentation, follow these steps:

```bash
cd docs
python create_functions_doc.py
```

## Submit Changes

- **Update the version**
  The version of the package should be increased by 0.0.1 for each new feature or bugfix. It's saved in the `ffit/__config__.py` file.
- **Commit Changes**  
   Commit your changes with a clear and concise message:

  ```bash
  git commit -m "Add feature/fix issue: description of the change"
  ```

- **Push Changes**  
   Push your branch to your fork:

  ```bash
  git push origin feature-or-bugfix-name
  ```

- **Ensure your branch is up to date** with the main branch:

  ```bash
  git fetch upstream
  git rebase upstream/main
  ```

- **Submit a Pull Request**
  Go to the original repository on GitHub and open a pull request. Provide a detailed description of the changes and reference any related issues.

## Communication

- For questions, issues, or feedback, open a GitHub issue.
- Before opening a PR, it's can be helpful to discuss your changes in an issue first.

---

Thank you for contributing! 🎉
