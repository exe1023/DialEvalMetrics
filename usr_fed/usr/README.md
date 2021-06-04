# USR Dockerized Server

## Build Docker Image
```
sh build_docker.sh
```

## Run Docker Container
1. Please refer to [Google Drive](https://drive.google.com/drive/folders/1sxaSIpAh6XOcmWd6dm__96DCamN-lCFX?usp=sharing) to download the pretrained models. Updated: You can also download data using seperate links [ctx](https://drive.google.com/file/d/1jkUeqUG0WFzSCmisbo1xTClRlCo8JPF3/view?usp=sharing), [roberta](https://drive.google.com/file/d/1YkXrkUdCFldl0EJXoMBy_uiu71wz1rgE/view?usp=sharing), and [uk](https://drive.google.com/file/d/1KLB3NSDjNv-ZX1I8pz4IxVRbvD3Bzbyk/view?usp=sharing).
2. Create a directory named `pretrained_models` and unzip the model folders to it. You should end up with the `pretrained_models/roberta_ft`, `pretrainedd_models/roberta_uk`, and `pretrained_models/ctx`
3.  Run 

```
nvidia-docker run -p 8888:8888 -v /path/to/pretrained_models:/workspace/pretrained_models --rm usr-test
```
or you can choose to modify `run_server.sh` to the command above.

## Test the Running Docker 
```
sh test_server.sh
```

## Code Structure

Entry Point: `usr_server.py`

Main API: `usr.py`

Retrieval Dialogue Metrics API: `dr_api.py`

Masked Language Model API: `mlm_api.py`

Arguments: `arguments.py`

