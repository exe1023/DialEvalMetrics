# FED Dockerized Server


## Build Docker Image
```
sh build_docker.sh
```

## Run Docker Container
```
sh run_docker.sh
```

## Test the Running Docker 
```
sh test_server.sh
```

## Code Structure

Entry Point: `fed_server.py`

Main API: `fed.py`

## Change Batch Size / Maximum Query Length

Change the parameter when calling score_batch in [here](https://github.com/exe1023/dialogue_metrics_docker/blob/ff4eb42718972a76cd3df734715fcf08ec956a87/fed/fed.py#L134) and [here](https://github.com/exe1023/dialogue_metrics_docker/blob/ff4eb42718972a76cd3df734715fcf08ec956a87/fed/fed.py#L201)
