# Mirrors manage.py for Linux/macOS. manage.py is the source of truth.
PY ?= python3

.PHONY: setup seed run test demo incident eval lint

setup:    ; $(PY) manage.py setup
seed:     ; $(PY) manage.py seed
run:      ; $(PY) manage.py run
test:     ; $(PY) manage.py test
demo:     ; $(PY) manage.py demo
incident: ; $(PY) manage.py incident
eval:     ; $(PY) manage.py eval
lint:     ; $(PY) manage.py lint
