PYTHON ?= python3

.PHONY: ask csv kb eval eval-judge pack

IN ?= data/demo_questions.csv
OUT ?= answers.csv
LIMIT ?=
DIST := dist
STAGE := $(DIST)/stage

ask:
	@test -n "$(Q)" || { echo 'Usage: make ask Q="your question"'; exit 1; }
	$(PYTHON) solution.py --query "$(Q)"

csv:
	$(PYTHON) solution.py --input $(IN) --output $(OUT)

kb:
	$(PYTHON) -m kb_build

# STRICT=1 → --strict-gates (exit 1 when quality gates FAIL)
STRICT ?=
STRICT_FLAG := $(if $(STRICT),--strict-gates,)

eval:
	$(PYTHON) -m eval --split dev --no-judge $(STRICT_FLAG) $(if $(LIMIT),--limit $(LIMIT),)

eval-judge:
	$(PYTHON) -m eval --split dev $(STRICT_FLAG) $(if $(LIMIT),--limit $(LIMIT),)

pack:
	@test -f build/chunks.jsonl || $(PYTHON) -m kb_build
	rm -rf $(STAGE)
	mkdir -p $(STAGE)/agent $(STAGE)/build
	cp solution.py requirements.txt $(STAGE)/
	cp agent/*.py $(STAGE)/agent/
	cp build/chunks.jsonl build/cards.jsonl build/glossary.json build/id_aliases.json $(STAGE)/build/
	rm -f $(DIST)/solution.zip
	cd $(STAGE) && zip -r ../solution.zip solution.py requirements.txt agent build
	@echo "wrote $(DIST)/solution.zip"
