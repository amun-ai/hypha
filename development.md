# Development

- We use [`black`](https://github.com/ambv/black) for code formatting.

```
  git clone git@github.com:amun-ai/hypha.git
  # Enter directory.
  cd hypha
  # Install all development requirements and package in development mode.
  pip3 install -r requirements_dev.txt
```

- Run `tox` to run all tests and lint, including checking that `black` doesn't change any files.
