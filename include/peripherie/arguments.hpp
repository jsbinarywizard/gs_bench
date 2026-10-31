// ============================================================================
// Common Benchmark Infrastructure
// ============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "options.hpp"


// ============================================================================
// Benchmark configuration
// ============================================================================

struct BenchmarkConfig {

    // ------------------------------------------------------------------------
    // Iteration control (time based)
    // ------------------------------------------------------------------------
    double warmup_time    = 1.0; // seconds
    double benchmark_time = 5.0; // seconds

    // ------------------------------------------------------------------------
    // Resource limits
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // Benchmark configuration
    // ------------------------------------------------------------------------
    std::vector<Scaling>   scaling   = {Scaling::STRONG};

    // Strong scaling: global problem sizes. Weak scaling: per-rank sizes.
    std::vector<int> strong_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> weak_sizes   = {32, 64, 128, 256, 512, 1024, 2048};

    // ------------------------------------------------------------------------
    // MPI Cartesian decomposition
    // ------------------------------------------------------------------------
    std::array<bool, 2> periodic = {false, false}; // {dim 0 (x), dim 1 (y)}

    // ------------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------------
    std::string output_file = "benchmark_results.csv";
};


// ============================================================================
// Benchmark argument parser
// ============================================================================

class BenchmarkParser {

private:

    std::string benchmark_name;

    enum class ArgResult { OK, HELP, NOT_STANDARD };

    // ------------------------------------------------------------------------
    // String helpers
    // ------------------------------------------------------------------------
    static std::string trim_copy(const std::string& input)
    {
        std::size_t start = 0;
        while (start < input.size() &&
               std::isspace(static_cast<unsigned char>(input[start]))) {
            ++start;
        }

        std::size_t end = input.size();
        while (end > start &&
               std::isspace(static_cast<unsigned char>(input[end - 1]))) {
            --end;
        }

        return input.substr(start, end - start);
    }

    static std::string lowercase_copy(const std::string& input)
    {
        std::string result = input;
        for (char& c : result) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    }

    // Trim + lowercase + treat '-' and '_' as equivalent.
    static std::string normalize_token(const std::string& input)
    {
        std::string result = lowercase_copy(trim_copy(input));
        std::replace(result.begin(), result.end(), '-', '_');
        return result;
    }

    // Split on ',' (empty tokens are preserved so callers can reject them).
    static std::vector<std::string> split_commas(const std::string& input)
    {
        std::vector<std::string> tokens;
        std::size_t begin = 0;

        while (true) {
            const std::size_t comma = input.find(',', begin);
            tokens.push_back(input.substr(
                begin,
                comma == std::string::npos ? std::string::npos : comma - begin));

            if (comma == std::string::npos) {
                break;
            }
            begin = comma + 1;
        }

        return tokens;
    }

    template <typename T>
    static void append_unique(std::vector<T>& list, const T& value)
    {
        if (std::find(list.begin(), list.end(), value) == list.end()) {
            list.push_back(value);
        }
    }

    // ------------------------------------------------------------------------
    // Number parsing (whole string must be consumed)
    // ------------------------------------------------------------------------
    static bool parse_int(const char* value, int& result)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        try {
            std::size_t consumed = 0;
            const std::string str(value);
            const int parsed = std::stoi(str, &consumed);

            if (consumed != str.size()) {
                return false;
            }

            result = parsed;
            return true;
        }
        catch (const std::exception&) {
            return false;
        }
    }

    static bool parse_double(const char* value, double& result)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        try {
            std::size_t consumed = 0;
            const std::string str(value);
            const double parsed = std::stod(str, &consumed);

            if (consumed != str.size() || !std::isfinite(parsed)) {
                return false;
            }

            result = parsed;
            return true;
        }
        catch (const std::exception&) {
            return false;
        }
    }


    // ------------------------------------------------------------------------
    // Parse periodic boundaries: comma-separated list of
    //   x | y | xy | both | all | none      (0 / 1 are aliases for x / y)
    // e.g. "x", "x,y", "both", "none". Unlisted dimensions are non-periodic.
    // ------------------------------------------------------------------------
    static bool parse_periodic(const char* value, std::array<bool, 2>& out)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        std::array<bool, 2> parsed = {false, false};

        for (const auto& raw : split_commas(value)) {
            const std::string token = normalize_token(raw);

            if (token == "x" || token == "0") {
                parsed[0] = true;
            }
            else if (token == "y" || token == "1") {
                parsed[1] = true;
            }
            else if (token == "xy" || token == "yx" ||
                     token == "both" || token == "all") {
                parsed = {true, true};
            }
            else if (token == "none") {
                // explicitly non-periodic; changes nothing
            }
            else {
                return false; // unknown or empty token
            }
        }

        out = parsed;
        return true;
    }

    // ------------------------------------------------------------------------
    // Parse "strong" | "weak" | "both".
    // ------------------------------------------------------------------------
    static bool parse_scalings(const char* value, std::vector<Scaling>& out)
    {
        if (value == nullptr) {
            return false;
        }

        const std::string token = normalize_token(value);

        if (token == "strong") {
            out = {Scaling::STRONG};
        }
        else if (token == "weak") {
            out = {Scaling::WEAK};
        }
        else if (token == "both" || token == "all") {
            out = {Scaling::STRONG, Scaling::WEAK};
        }
        else {
            return false;
        }

        return true;
    }

    // ------------------------------------------------------------------------
    // Parse a list of positive integers. Common separator mistakes
    // ('/', ';', '|') are silently treated as commas. The result is sorted
    // in ascending order and duplicates are removed.
    // ------------------------------------------------------------------------
    static bool parse_sizes(const char* value, std::vector<int>& sizes)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        std::string input(value);

        std::replace_if(
            input.begin(), input.end(),
            [](char c) { return c == '/' || c == ';' || c == '|'; },
            ',');

        std::vector<int> parsed;

        for (const auto& raw : split_commas(input)) {
            const std::string token = trim_copy(raw);

            int size = 0;
            if (!parse_int(token.c_str(), size) || size <= 0) {
                return false;
            }

            parsed.push_back(size);
        }

        std::sort(parsed.begin(), parsed.end());
        parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());

        sizes = std::move(parsed);
        return !sizes.empty();
    }

    // ------------------------------------------------------------------------
    // Fetch the value following an option, or throw.
    // ------------------------------------------------------------------------
    static const char* require_value(
        const char* arg, int argc, char* argv[], int& i)
    {
        if (i + 1 >= argc) {
            throw std::runtime_error(
                std::string("Error: ") + arg + " requires a value");
        }
        return argv[++i];
    }

    // ------------------------------------------------------------------------
    // Parse one argument. Throws std::runtime_error on invalid input.
    // ------------------------------------------------------------------------
    ArgResult parse_arg(
        const char* arg, int argc, char* argv[], int& i, BenchmarkConfig& config)
    {
        const auto is = [arg](const char* a, const char* b = nullptr) {
            return std::strcmp(arg, a) == 0 ||
                   (b != nullptr && std::strcmp(arg, b) == 0);
        };

        // Help
        if (is("--help", "-h")) {
            return ArgResult::HELP;
        }

        // Warmup time
        if (is("--warmup-time", "-w")) {
            double v = 0.0;
            if (!parse_double(require_value(arg, argc, argv, i), v) || v < 0.0) {
                throw std::runtime_error(
                    "Error: Warmup time must be a non-negative number of seconds");
            }
            config.warmup_time = v;
            return ArgResult::OK;
        }

        // Benchmark time
        if (is("--bench-time", "-b")) {
            double v = 0.0;
            if (!parse_double(require_value(arg, argc, argv, i), v) || v <= 0.0) {
                throw std::runtime_error(
                    "Error: Benchmark time must be a positive number of seconds");
            }
            config.benchmark_time = v;
            return ArgResult::OK;
        }

        // Output file
        if (is("--output", "-o")) {
            const std::string file = require_value(arg, argc, argv, i);
            if (file.empty()) {
                throw std::runtime_error(
                    "Error: Output filename must not be empty");
            }
            config.output_file = file;
            return ArgResult::OK;
        }

        // Periodic boundaries
        if (is("--periodic")) {
            if (!parse_periodic(require_value(arg, argc, argv, i),
                                config.periodic)) {
                throw std::runtime_error(
                    "Error: --periodic must be a comma-separated list of: "
                    "'x', 'y', 'both', 'none'");
            }
            return ArgResult::OK;
        }

        // Scaling
        if (is("--scaling")) {
            if (!parse_scalings(require_value(arg, argc, argv, i),
                                config.scaling)) {
                throw std::runtime_error(
                    "Error: Scaling must be 'strong', 'weak', or 'both'");
            }
            return ArgResult::OK;
        }

        // Grid sizes: --sizes sets both lists, the others set one of them.
        // Later options override earlier ones.
        if (is("--sizes") || is("--strong-sizes") || is("--weak-sizes")) {
            std::vector<int> sizes;
            if (!parse_sizes(require_value(arg, argc, argv, i), sizes)) {
                throw std::runtime_error(
                    std::string("Error: ") + arg +
                    " must contain positive integers separated by commas");
            }

            if (!is("--weak-sizes")) {
                config.strong_sizes = sizes;
            }
            if (!is("--strong-sizes")) {
                config.weak_sizes = std::move(sizes);
            }
            return ArgResult::OK;
        }

        return ArgResult::NOT_STANDARD;
    }

    // Join a list of ints for the usage text.
    static std::string join(const std::vector<int>& values)
    {
        std::ostringstream os;
        for (std::size_t k = 0; k < values.size(); ++k) {
            os << (k ? "," : "") << values[k];
        }
        return os.str();
    }

public:

    explicit BenchmarkParser(const std::string& name = "benchmark")
        : benchmark_name(name)
    {
    }

    // ------------------------------------------------------------------------
    // Print command-line usage. Defaults are taken from BenchmarkConfig so
    // the help text can never drift from the actual defaults.
    // ------------------------------------------------------------------------
    void print_usage(const char* program_name) const
    {
        const BenchmarkConfig d;

        std::cout
            << benchmark_name << "\n"
            << "Usage: " << program_name << " [options]\n\n"

            << "General options:\n"
            << "  -h, --help\n"
            << "        Show this help message and exit.\n\n"

            << "Iteration control:\n"
            << "  -w, --warmup-time SECONDS\n"
            << "        Warmup duration before timing.\n"
            << "        (default: " << d.warmup_time << ")\n\n"

            << "  -b, --bench-time SECONDS\n"
            << "        Duration of the timed benchmark.\n"
            << "        (default: " << d.benchmark_time << ")\n\n"

            << "  --periodic LIST\n"
            << "        Periodic boundaries, comma-separated:\n"
            << "            x      -> periodic in dimension 0\n"
            << "            y      -> periodic in dimension 1\n"
            << "            both   -> periodic in both dimensions\n"
            << "            none   -> no periodic boundaries\n"
            << "        Example: --periodic x,y\n"
            << "        (default: none)\n\n"

            << "  --scaling TYPE\n"
            << "        strong -> --sizes gives the fixed GLOBAL problem size,\n"
            << "                  split across ranks\n"
            << "        weak   -> --sizes gives the fixed LOCAL (per-rank) size\n"
            << "        both   -> run both\n"
            << "        (default: strong)\n\n"

            << "  --sizes LIST\n"
            << "        Comma-separated grid sizes, used for both scalings.\n"
            << "        Example: 32,64,128,256\n\n"

            << "  --strong-sizes LIST / --weak-sizes LIST\n"
            << "        Grid sizes for one scaling mode only. Later options\n"
            << "        override earlier ones.\n"
            << "        (default strong: " << join(d.strong_sizes) << ")\n"
            << "        (default weak:   " << join(d.weak_sizes) << ")\n\n"

            << "Output:\n"
            << "  -o, --output FILE\n"
            << "        CSV file used to store benchmark results.\n"
            << "        (default: " << d.output_file << ")\n";
    }

    // ------------------------------------------------------------------------
    // Parse all arguments.
    //
    // Returns:
    //   true  -> parsing succeeded, run the benchmark
    //   false -> help was requested (usage printed on rank 0)
    //
    // Throws std::runtime_error with a descriptive message on invalid input.
    // ------------------------------------------------------------------------
    bool parse(int argc, char* argv[], BenchmarkConfig& config, int rank = 0)
    {
        for (int i = 1; i < argc; ++i) {
            switch (parse_arg(argv[i], argc, argv, i, config)) {
                case ArgResult::OK:
                    break;

                case ArgResult::HELP:
                    if (rank == 0) {
                        print_usage(argv[0]);
                    }
                    return false;

                case ArgResult::NOT_STANDARD:
                    throw std::runtime_error(
                        "Error: Invalid argument: " + std::string(argv[i]));
            }
        }

        return true;
    }
};