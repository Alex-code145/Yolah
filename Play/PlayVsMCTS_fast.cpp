#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int BLACK_PLAYER = 0;
constexpr int WHITE_PLAYER = 1;

constexpr uint64_t FILE_A = 0x0101010101010101ULL;
constexpr uint64_t FILE_H = FILE_A << 7;

inline uint64_t bit_not(uint64_t n) {
    return ~n;
}

enum class Direction : int {
    NORTH = 8,
    EAST = 1,
    SOUTH = -8,
    WEST = -1,
    NORTH_EAST = 9,
    SOUTH_EAST = -7,
    SOUTH_WEST = -9,
    NORTH_WEST = 7,
};

inline uint64_t shift(Direction dir, uint64_t b) {
    switch (dir) {
        case Direction::NORTH:
            return b << 8;
        case Direction::EAST:
            return (b & bit_not(FILE_H)) << 1;
        case Direction::SOUTH:
            return b >> 8;
        case Direction::WEST:
            return (b & bit_not(FILE_A)) >> 1;
        case Direction::NORTH_EAST:
            return (b & bit_not(FILE_H)) << 9;
        case Direction::SOUTH_EAST:
            return (b & bit_not(FILE_H)) >> 7;
        case Direction::SOUTH_WEST:
            return (b & bit_not(FILE_A)) >> 9;
        case Direction::NORTH_WEST:
            return (b & bit_not(FILE_A)) << 7;
    }
    return 0;
}

const std::array<Direction, 8> kAllDirections = {
    Direction::NORTH,
    Direction::EAST,
    Direction::SOUTH,
    Direction::WEST,
    Direction::NORTH_EAST,
    Direction::SOUTH_EAST,
    Direction::SOUTH_WEST,
    Direction::NORTH_WEST,
};

struct Move {
    int from = 0;
    int to = 0;

    static Move none() {
        return {0, 0};
    }

    bool is_none() const {
        return from == 0 && to == 0;
    }

    std::string to_str() const {
        auto sq_to_str = [](int sq) {
            char file = static_cast<char>('a' + (sq % 8));
            char rank = static_cast<char>('1' + (sq / 8));
            std::string s;
            s.push_back(file);
            s.push_back(rank);
            return s;
        };
        return sq_to_str(from) + ":" + sq_to_str(to);
    }

    static bool from_str(const std::string& s, Move& out) {
        if (s.size() != 5 || s[2] != ':') {
            return false;
        }
        auto parse_sq = [](char f, char r, int& sq) {
            if (f < 'a' || f > 'h' || r < '1' || r > '8') {
                return false;
            }
            sq = (r - '1') * 8 + (f - 'a');
            return true;
        };
        int from_sq = 0;
        int to_sq = 0;
        if (!parse_sq(s[0], s[1], from_sq) || !parse_sq(s[3], s[4], to_sq)) {
            return false;
        }
        out = {from_sq, to_sq};
        return true;
    }
};

struct Yolah {
    uint64_t black = 0;
    uint64_t white = 0;
    uint64_t empty = 0;
    int black_score = 0;
    int white_score = 0;
    int ply = 0;

    Yolah() {
        reset();
    }

    void reset() {
        black = 0b1000000000000000000000000000100000010000000000000000000000000001ULL;
        white = 0b0000000100000000000000000001000000001000000000000000000010000000ULL;
        empty = 0;
        black_score = 0;
        white_score = 0;
        ply = 0;
    }

    int current_player() const {
        return (ply & 1) ? WHITE_PLAYER : BLACK_PLAYER;
    }

    std::vector<Move> moves_for(int player) const {
        std::vector<Move> res;
        uint64_t free = bit_not(black | white | empty);
        uint64_t pieces = (player == BLACK_PLAYER) ? black : white;

        while (pieces) {
            uint64_t pos = pieces & -pieces;
            int from_sq = static_cast<int>(__builtin_ctzll(pos));
            for (Direction dir : kAllDirections) {
                uint64_t dst = shift(dir, pos);
                while (dst & free) {
                    int to_sq = static_cast<int>(__builtin_ctzll(dst));
                    res.push_back({from_sq, to_sq});
                    dst = shift(dir, dst);
                }
            }
            pieces &= ~pos;
        }

        if (res.empty()) {
            res.push_back(Move::none());
        }
        return res;
    }

    std::vector<Move> moves() const {
        return moves_for(current_player());
    }

    bool game_over() const {
        uint64_t possible = bit_not(black) & bit_not(white) & bit_not(empty);
        return (shift(Direction::NORTH, black) & possible) == 0 &&
               (shift(Direction::EAST, black) & possible) == 0 &&
               (shift(Direction::SOUTH, black) & possible) == 0 &&
               (shift(Direction::WEST, black) & possible) == 0 &&
               (shift(Direction::NORTH_EAST, black) & possible) == 0 &&
               (shift(Direction::SOUTH_EAST, black) & possible) == 0 &&
               (shift(Direction::SOUTH_WEST, black) & possible) == 0 &&
               (shift(Direction::NORTH_WEST, black) & possible) == 0 &&
               (shift(Direction::NORTH, white) & possible) == 0 &&
               (shift(Direction::EAST, white) & possible) == 0 &&
               (shift(Direction::SOUTH, white) & possible) == 0 &&
               (shift(Direction::WEST, white) & possible) == 0 &&
               (shift(Direction::NORTH_EAST, white) & possible) == 0 &&
               (shift(Direction::SOUTH_EAST, white) & possible) == 0 &&
               (shift(Direction::SOUTH_WEST, white) & possible) == 0 &&
               (shift(Direction::NORTH_WEST, white) & possible) == 0;
    }

    void play(const Move& m) {
        if (m.is_none()) {
            ply += 1;
            return;
        }

        uint64_t from_bb = 1ULL << m.from;
        uint64_t to_bb = 1ULL << m.to;

        if (current_player() == BLACK_PLAYER) {
            black = (black & ~from_bb) | to_bb;
            black_score += 1;
        } else {
            white = (white & ~from_bb) | to_bb;
            white_score += 1;
        }
        empty |= from_bb;
        ply += 1;
    }

    int result() const {
        if (!game_over()) {
            return 0;
        }
        if (black_score > white_score) {
            return 1;
        }
        if (white_score > black_score) {
            return -1;
        }
        return 0;
    }

    std::string board_string() const {
        std::ostringstream out;
        out << "  a  b  c  d  e  f  g  h\n";
        for (int i = 0; i < 8; ++i) {
            out << (8 - i);
            for (int j = 0; j < 8; ++j) {
                int sq = (7 - i) * 8 + j;
                uint64_t bit = 1ULL << sq;
                if (black & bit) {
                    out << " O ";
                } else if (white & bit) {
                    out << " X ";
                } else if (empty & bit) {
                    out << "   ";
                } else {
                    out << " . ";
                }
            }
            out << (8 - i) << "\n";
        }
        out << "  a  b  c  d  e  f  g  h\n";
        out << "score: " << black_score << "/" << white_score;
        return out.str();
    }
};

struct MCTSNode {
    Yolah state;
    int parent = -1;
    Move move;
    std::vector<int> children;
    int visit_count = 0;
    double value_sum = 0.0;

    double value() const {
        return (visit_count > 0) ? (value_sum / visit_count) : 0.0;
    }
};

struct MCTSStats {
    int iterations = 0;
    double elapsed_s = 0.0;
    double playouts_per_s = 0.0;
};

double uct(const MCTSNode& parent, const MCTSNode& child, double c = 1.4) {
    double exploration = c * std::sqrt(std::log(parent.visit_count + 1.0) / (child.visit_count + 1e-9));
    return -child.value() + exploration;
}

int select_leaf(std::vector<MCTSNode>& nodes, int node_idx) {
    while (!nodes[node_idx].children.empty()) {
        int best_child = nodes[node_idx].children[0];
        double best_score = -std::numeric_limits<double>::infinity();
        for (int child_idx : nodes[node_idx].children) {
            double score = uct(nodes[node_idx], nodes[child_idx]);
            if (score > best_score) {
                best_score = score;
                best_child = child_idx;
            }
        }
        node_idx = best_child;
    }
    return node_idx;
}

int expand(std::vector<MCTSNode>& nodes, int node_idx, std::mt19937_64& rng) {
    if (nodes[node_idx].state.game_over()) {
        return node_idx;
    }

    auto moves = nodes[node_idx].state.moves();
    for (const Move& m : moves) {
        Yolah s2 = nodes[node_idx].state;
        s2.play(m);
        MCTSNode child;
        child.state = s2;
        child.parent = node_idx;
        child.move = m;
        nodes.push_back(std::move(child));
        nodes[node_idx].children.push_back(static_cast<int>(nodes.size() - 1));
    }

    std::uniform_int_distribution<size_t> dist(0, nodes[node_idx].children.size() - 1);
    return nodes[node_idx].children[dist(rng)];
}

int rollout(Yolah state, std::mt19937_64& rng) {
    while (!state.game_over()) {
        auto moves = state.moves();
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        state.play(moves[dist(rng)]);
    }
    return state.result();
}

void backpropagate(std::vector<MCTSNode>& nodes, int node_idx, int reward) {
    while (node_idx >= 0) {
        nodes[node_idx].visit_count += 1;
        nodes[node_idx].value_sum += static_cast<double>(reward);
        reward = -reward;
        node_idx = nodes[node_idx].parent;
    }
}

Move mcts_best_move(const Yolah& root_state, int iterations, double time_limit_s, bool verbose = true, MCTSStats* out_stats = nullptr) {
    std::random_device rd;
    std::mt19937_64 rng(rd());

    std::vector<MCTSNode> nodes;
    nodes.reserve(120000);
    nodes.push_back(MCTSNode{root_state, -1, Move::none(), {}, 0, 0.0});

    const auto start = std::chrono::steady_clock::now();
    int it = 0;

    while (true) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (time_limit_s > 0.0 && elapsed >= time_limit_s) {
            break;
        }
        if (time_limit_s <= 0.0 && it >= iterations) {
            break;
        }

        int leaf_idx = select_leaf(nodes, 0);
        int expanded_idx = expand(nodes, leaf_idx, rng);
        int result = rollout(nodes[expanded_idx].state, rng);
        int reward = (nodes[expanded_idx].state.current_player() == BLACK_PLAYER) ? result : -result;
        backpropagate(nodes, expanded_idx, reward);
        it += 1;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double nps = elapsed > 0.0 ? (static_cast<double>(it) / elapsed) : 0.0;

    if (out_stats != nullptr) {
        out_stats->iterations = it;
        out_stats->elapsed_s = elapsed;
        out_stats->playouts_per_s = nps;
    }

    if (verbose) {
        std::cout << "Fast MCTS: iterations=" << it
                  << ", elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                  << ", playouts/s=" << std::setprecision(0) << nps << "\n";
    }

    if (nodes[0].children.empty()) {
        return Move::none();
    }

    std::vector<int> ranked = nodes[0].children;
    std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
        return nodes[a].visit_count > nodes[b].visit_count;
    });

    if (verbose) {
        std::cout << "Top moves: ";
        for (size_t i = 0; i < std::min<size_t>(3, ranked.size()); ++i) {
            const auto& ch = nodes[ranked[i]];
            double q_root = (ch.visit_count > 0) ? (-ch.value_sum / ch.visit_count) : 0.0;
            std::cout << ch.move.to_str() << "(N=" << ch.visit_count << ",Q=" << std::setprecision(3) << q_root << ")";
            if (i + 1 < std::min<size_t>(3, ranked.size())) {
                std::cout << " | ";
            }
        }
        std::cout << "\n";
    }

    return nodes[ranked[0]].move;
}

bool is_legal(const std::vector<Move>& moves, const Move& m) {
    for (const auto& candidate : moves) {
        if (candidate.from == m.from && candidate.to == m.to) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // Engine mode: compute a single move from a serialized game state.
    // Usage:
    //   ./PlayVsMCTS_fast --mcts <black> <white> <empty> <black_score> <white_score> <ply> <iterations> <time_limit_s>
    if (argc >= 2 && std::string(argv[1]) == "--mcts") {
        if (argc != 10) {
            std::cerr << "Usage: --mcts <black> <white> <empty> <black_score> <white_score> <ply> <iterations> <time_limit_s>\n";
            return 2;
        }

        Yolah state;
        state.black = static_cast<uint64_t>(std::stoull(argv[2]));
        state.white = static_cast<uint64_t>(std::stoull(argv[3]));
        state.empty = static_cast<uint64_t>(std::stoull(argv[4]));
        state.black_score = std::stoi(argv[5]);
        state.white_score = std::stoi(argv[6]);
        state.ply = std::stoi(argv[7]);

        int iterations = std::stoi(argv[8]);
        double time_limit_s = std::stod(argv[9]);

        MCTSStats stats;
        Move best = mcts_best_move(state, iterations, time_limit_s, false, &stats);
        std::cerr << "iterations=" << stats.iterations
                  << " elapsed=" << std::fixed << std::setprecision(3) << stats.elapsed_s << "s"
                  << " playouts_per_s=" << std::setprecision(0) << stats.playouts_per_s
                  << "\n";
        std::cout << best.to_str() << "\n";
        return 0;
    }

    Yolah state;
    std::cout << state.board_string() << "\n";

    while (!state.game_over()) {
        if (state.current_player() == BLACK_PLAYER) {
            auto moves = state.moves();
            std::cout << "Possible moves: ";
            for (const auto& m : moves) {
                std::cout << m.to_str() << " ";
            }
            std::cout << "\n";

            if (moves.size() == 1 && moves[0].is_none()) {
                std::cout << "No move for Black, pass.\n";
                state.play(Move::none());
            } else {
                std::string input;
                std::cout << "Your move (e.g. a1:a3): ";
                std::cin >> input;
                Move m;
                if (!Move::from_str(input, m) || !is_legal(moves, m)) {
                    std::cout << "Illegal move, try again.\n";
                    continue;
                }
                state.play(m);
            }
        } else {
            std::cout << "AI (fast C++) thinking...\n";
            Move ai = mcts_best_move(state, 120000, 3.0, true);
            std::cout << "AI plays: " << ai.to_str() << "\n";
            state.play(ai);
        }

        std::cout << state.board_string() << "\n";
    }

    int res = state.result();
    std::cout << "Game over. Result: " << res << " (1=Black, -1=White, 0=Draw)\n";
    return 0;
}
