PYTHON ?= /usr/bin/python3
# Fine-tuning needs torch etc.: a separate env, never the daemon's python.
TRAIN_PY ?= conda run -n aiproof-train python
EVAL_PRIVATE ?= $(HOME)/.local/share/aiproof/evals_private.json

.PHONY: test test-flow eval eval-private bench train-data train export \
	training-check deb install-user uninstall-user run-daemon clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -q

# Route-by-route flow tests, verbose with live logs; also writes the
# per-route log report to build/flow-report.md.
test-flow:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -m flow -v --log-level=DEBUG
	@echo "Route report: build/flow-report.md"

# Live prompt/pipeline evaluation against the configured model+endpoint.
# Args: make eval EVAL_ARGS="--model llama3:latest --show-output"
eval:
	$(PYTHON) evals/run_evals.py $(EVAL_ARGS)

# The private gold set (hand-reviewed real proofreads, never committed).
eval-private:
	$(PYTHON) evals/run_evals.py --cases $(EVAL_PRIVATE) $(EVAL_ARGS)

# Raw tokens/s per Ollama model: make bench BENCH_ARGS="qwen2.5:3b qwen2.5:1.5b"
bench:
	$(PYTHON) training/bench_ollama.py $(BENCH_ARGS)

# -- custom model (see training/README.md) --------------------------------------
train-data:
	PYTHONPATH=src $(PYTHON) training/mine_history.py $(MINE_ARGS)
	PYTHONPATH=src $(TRAIN_PY) training/build_dataset.py $(DATA_ARGS)

train:
	PYTHONPATH=src $(TRAIN_PY) training/train.py $(TRAIN_ARGS)

# make export RUN=training/runs/qwen1.5b-edit NAME=aiproof-edit-1.5b
export:
	PYTHONPATH=src TRAIN_PY="$(TRAIN_PY)" training/export.sh $(RUN) $(NAME)

training-check:
	@for d in training/data training/runs; do git check-ignore -q $$d/probe \
		|| { echo "ERROR: $$d is not git-ignored"; exit 1; }; done
	@! git ls-files --cached --others --exclude-standard | grep -E '\.(gguf|safetensors|jsonl)$$' \
		|| { echo "ERROR: model/data files are tracked or unignored"; exit 1; }
	@echo "training data, runs and model files are ignored"

deb:
	dpkg-buildpackage -us -uc -b
	@echo "Package built: $$(ls -t ../aiproof_*_all.deb | head -1)"

# Dev-mode install: no root, iterates straight from the source tree.
install-user:
	mkdir -p ~/.local/bin ~/.local/share/applications \
		~/.local/share/icons/hicolor/scalable/apps
	printf '#!/bin/sh\nexport PYTHONPATH="$(CURDIR)/src"\nexec $(PYTHON) -m aiproof.cli "$$@"\n' \
		> ~/.local/bin/aiproof
	chmod +x ~/.local/bin/aiproof
	sed 's|^Exec=aiproof|Exec=$(HOME)/.local/bin/aiproof|' \
		data/io.github.edwatson.aiproof.desktop \
		> ~/.local/share/applications/io.github.edwatson.aiproof.desktop
	cp data/icons/io.github.edwatson.aiproof.svg \
		~/.local/share/icons/hicolor/scalable/apps/
	gtk-update-icon-cache -q ~/.local/share/icons/hicolor 2>/dev/null || true
	update-desktop-database ~/.local/share/applications 2>/dev/null || true
	@echo "Installed. Start with: ~/.local/bin/aiproof daemon"

uninstall-user:
	rm -f ~/.local/bin/aiproof \
		~/.local/share/applications/io.github.edwatson.aiproof.desktop \
		~/.local/share/icons/hicolor/scalable/apps/io.github.edwatson.aiproof.svg \
		~/.config/autostart/io.github.edwatson.aiproof-daemon.desktop

run-daemon:
	PYTHONPATH=src $(PYTHON) -m aiproof.cli -v daemon

clean:
	rm -rf build dist src/aiproof.egg-info .pybuild debian/aiproof \
		debian/.debhelper debian/files debian/debhelper-build-stamp \
		debian/aiproof.substvars
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
