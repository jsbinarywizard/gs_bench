#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "options.hpp"

template <typename Real>
const char* precision_name() {
    return std::is_same_v<Real, float> ? "single" : "double";
}

struct BenchmarkResult {
    std::string transport;
    std::string precision;
    Scaling scaling;
    int ranks = 0;
    int size = 0;
    int iterations = 0;
    double total_time = 0.0;
    double updates_per_second = 0.0;
};

class ResultsWriter {
public:
    ResultsWriter(const std::string& path, bool active) : active_(active) {
        if (!active_) return;

        csv_.open(path);
        if (!csv_) {
            throw std::runtime_error("Could not open output file '" + path + "'");
        }
        csv_ << "transport,precision,scaling,ranks,size,iterations,total_time,"
                "updates_per_second\n";
    }

    void write(const BenchmarkResult& result) {
        if (!active_) return;

        csv_ << result.transport << ','
             << result.precision << ','
             << scaling_name(result.scaling) << ','
             << result.ranks << ','
             << result.size << ','
             << result.iterations << ','
             << result.total_time << ','
             << result.updates_per_second << '\n';
        if (!csv_) {
            throw std::runtime_error("Failed writing benchmark results");
        }
    }

private:
    bool active_ = false;
    std::ofstream csv_;
};
