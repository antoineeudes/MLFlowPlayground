# MLFlowPlayground

## Prerequisites
 - docker
 - docker-compose

## Installation
```bash
make build
```

## Usage
To start the `mlflow` UI interface,
```bash
make start
```
Browse http://localhost:5000/ and voilà! :tada:

### Run an experiment
```bash
make run-experiment
```

The parameters, the metric, the model and the artifacts can be vizualized on the interface.

## Uninstallation
```bash
make uninstall
```