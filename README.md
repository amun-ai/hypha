![PyPI](https://img.shields.io/pypi/v/imjoy.svg?style=popout)
# Hypha

A serverless application framework for large-scale data management and AI model serving.

## Installation
```
pip install -U hypha[server-apps]
playwright install
```

## Usage
To start the hypha server, run the following command:
```
python -m hypha.server --port=9527 --enable-server-apps
```

## Development

- We use [`black`](https://github.com/ambv/black) for code formatting.

```
  git clone git@github.com:imjoy-team/hypha.git
  # Enter directory.
  cd hypha
  # Install all development requirements and package in development mode.
  pip3 install -r requirements_dev.txt
```

- Run `tox` to run all tests and lint, including checking that `black` doesn't change any files.
