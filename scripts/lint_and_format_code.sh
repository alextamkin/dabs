#!/bin/bash

pylint *
yapf --recursive --in-place --exclude outputs/ .
isort --skip outputs/ .
