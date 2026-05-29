IMAGE   ?= king:latest
ADAPTER := ./engine-adapter.sh
CUTECHESS ?= cutechess-cli

.PHONY: unit perft build test-uci test-game clean help
help:
	@echo "make unit | perft | build | test-uci | test-game | clean"
unit:
	cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
perft:
	@build/engine.exe perft 5
build:
	docker build -t $(IMAGE) .
test-uci: build
	@printf "uci\nquit\n" | docker run --rm -i $(IMAGE) | grep -q "uciok" && echo "PASS: uciok" || (echo "FAIL"; exit 1)
test-game: build
	@chmod +x $(ADAPTER)
	ENGINE_IMAGE=$(IMAGE) ENGINE_MEMORY=2g ENGINE_NET=none $(CUTECHESS) \
	  -engine cmd=$(ADAPTER) name=KINg1 proto=uci \
	  -engine cmd=$(ADAPTER) name=KINg2 proto=uci \
	  -each tc=10+0.1 -games 2 -rounds 1 -repeat -recover -pgnout selfplay.pgn \
	  && echo "PASS: game(s) finished" || echo "FAIL: cutechess returned non-zero"
clean:
	docker rmi -f $(IMAGE) 2>/dev/null || true
