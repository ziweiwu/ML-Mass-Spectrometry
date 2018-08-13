#!/usr/bin/env bash
docker run --rm -p 8900:89 -e JUPYTER_ENABLE_LAB=yes -v "${pwd}:/home/jovyan/work" jupyter/scipy-notebook:latest
