# =============================================================================
# KINg — local Docker test harness (competition framework).
#
# Targets:
#   make build        build the engine Docker image (tag = $(IMAGE))
#   make test-uci     smoke-test: container answers "uci" with "uciok"
#   make test-game    quick self-play (KINg vs KINg) via cutechess-cli
#   make test-tucano  match KINg vs Tucano (needs a $(TUCANO_IMAGE) image)
#   make unit         host build + ctest (CMake unit tests in build/)
#   make perft        run perft(5) on the host binary
#   make clean        remove the engine Docker image
#
# NOTE: do NOT run `cmake` in the repo root — it overwrites this Makefile with
# a generated one. Always build into a fresh build/ dir (the unit target does).
# =============================================================================

IMAGE        ?= king:latest
TUCANO_IMAGE ?= tucano:latest
ADAPTER      := ./engine-adapter.sh
CUTECHESS    ?= cutechess-cli

# Match parameters (override on the command line, e.g. `make test-tucano GAMES=100`)
TC     ?= 10+0.1
GAMES  ?= 20
ROUNDS ?= 10

.PHONY: help build test-uci test-game test-tucano unit perft clean

help:
	@echo "make build | test-uci | test-game | test-tucano | unit | perft | clean"

build:
	docker build -t $(IMAGE) .

test-uci: build
	@printf "uci\nquit\n" | docker run --rm -i $(IMAGE) | grep -q "uciok" \
	  && echo "PASS: uciok" || (echo "FAIL: no uciok"; exit 1)

test-game: build
	@chmod +x $(ADAPTER)
	ENGINE_IMAGE=$(IMAGE) ENGINE_MEMORY=2g ENGINE_NET=none $(CUTECHESS) \
	  -engine cmd=$(ADAPTER) name=KINg1 proto=uci \
	  -engine cmd=$(ADAPTER) name=KINg2 proto=uci \
	  -each tc=$(TC) -games 2 -rounds 1 -repeat -recover -pgnout selfplay.pgn \
	  && echo "PASS: game(s) finished" || echo "FAIL: cutechess returned non-zero"

# KINg vs Tucano. Build the Tucano image first, e.g.:
#   git clone https://github.com/kinoai/chess_engine_competition ../tucano-ref
#   docker build -t $(TUCANO_IMAGE) ../tucano-ref
test-tucano: build
	@chmod +x $(ADAPTER)
	$(CUTECHESS) \
	  -engine name=KINg   cmd="env ENGINE_IMAGE=$(IMAGE) $(ADAPTER)"        proto=uci \
	  -engine name=Tucano cmd="env ENGINE_IMAGE=$(TUCANO_IMAGE) $(ADAPTER)" proto=uci \
	  -each tc=$(TC) -games $(GAMES) -rounds $(ROUNDS) -repeat -recover \
	  -pgnout king_vs_tucano.pgn \
	  && echo "PASS: match finished -> king_vs_tucano.pgn" \
	  || echo "FAIL: cutechess returned non-zero"

unit:
	cmake -B build -DCMAKE_BUILD_TYPE=Release \
	  && cmake --build build -j \
	  && ctest --test-dir build --output-on-failure

perft:
	@build/engine perft 5

clean:
	docker rmi -f $(IMAGE) 2>/dev/null || true
