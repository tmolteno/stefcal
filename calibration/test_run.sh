#!/bin/sh
uv run tart-stefcal run --tart-name $1 \
    --archive --duration 90 --n 15 \
    --upload --pw $2
