#pragma once

#include <Kokkos_Core.hpp>
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#include <Kokkos_Graph.hpp>
#endif
#if defined(KOKKOS_ENABLE_CUDA)
#include <Kokkos_Cuda.hpp>
#include <nccl.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <Kokkos_HIP.hpp>
#include <rccl/rccl.h>
#endif

#include <mpi.h>
#include "../compute/kernel.hpp"
#include "../peripherie/arguments.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace gs_bench {

constexpr int stream_count = 5;
constexpr int mpi_tag_top = 100;
constexpr int mpi_tag_bottom = 101;
using execution_space = Kokkos::DefaultExecutionSpace;

template <typename Real = double>
using view_type = gs_view<Real>;

template <typename Real = double>
using buffer_type = Kokkos::View<Real*>;

template <typename Real = double>
inline MPI_Datatype mpi_datatype();

template <>
inline MPI_Datatype mpi_datatype<float>() {
    return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_datatype<double>() {
    return MPI_DOUBLE;
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <typename Real = double>
inline ncclDataType_t nccl_datatype();

template <>
inline ncclDataType_t nccl_datatype<float>() {
    return ncclFloat;
}

template <>
inline ncclDataType_t nccl_datatype<double>() {
    return ncclDouble;
}
#endif

struct StreamSet {
    std::array<execution_space, stream_count> streams = {
        execution_space(), execution_space(), execution_space(),
        execution_space(), execution_space()};

    void fence() const {
        for (const auto& stream : streams) stream.fence();
    }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    auto native(int index) const {
#if defined(KOKKOS_ENABLE_CUDA)
        return streams[index].cuda_stream();
#else
        return streams[index].hip_stream();
#endif
    }
#endif
};

inline void check_mpi(int error, const char* expression, const char* file, int line) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        std::fprintf(stderr, "%s:%d: MPI call '%s' failed: %.*s\n", file, line, expression,
                     length, message);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define GS_MPI_CHECK(call) ::gs_bench::check_mpi((call), #call, __FILE__, __LINE__)

struct Arguments {
    BenchmarkConfig config;
};

inline Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    int rank = 0;
    GS_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    try {
        BenchmarkParser parser("gs_bench");
        if (!parser.parse(argc, argv, args.config, rank)) {
            MPI_Finalize();
            std::exit(EXIT_SUCCESS);
        }
    } catch (const std::exception& error) {
        if (rank == 0) std::fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return args;
}

struct BenchmarkCase {
    Scaling scaling;
    int size;
    int global_rows;
    int columns;
};

inline std::vector<BenchmarkCase> make_cases(const BenchmarkConfig& config, int rank_count) {
    std::vector<BenchmarkCase> cases;
    for (const Scaling scaling : config.scaling) {
        const auto& sizes = scaling == Scaling::STRONG ? config.strong_sizes : config.weak_sizes;
        for (const int size : sizes) {
            const int global_rows = scaling == Scaling::STRONG ? size : size * rank_count;
            cases.push_back({scaling, size, global_rows, size});
        }
    }
    return cases;
}

struct Partition {
    int rank = 0;
    int size = 0;
    int rows = 0;
    int columns = 0;

    explicit Partition(int global_rows, int global_columns) : columns(global_columns) {
        GS_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        GS_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        const int base = global_rows / size;
        const int remainder = global_rows % size;
        rows = base + (rank < remainder ? 1 : 0);
        if (rows < 3) {
            if (rank == 0) {
                std::fprintf(stderr, "Each MPI rank must own at least 3 rows\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int top_rank() const { return (rank + size - 1) % size; }
    int bottom_rank() const { return (rank + 1) % size; }
};

template <typename Real = double>
struct Fields {
    view_type<Real> u;
    view_type<Real> v;
    view_type<Real> next_u;
    view_type<Real> next_v;
    buffer_type<Real> send_top;
    buffer_type<Real> send_bottom;
    buffer_type<Real> recv_top;
    buffer_type<Real> recv_bottom;

    explicit Fields(const Partition& part)
        : u("u", part.rows + 2, part.columns + 2),
          v("v", part.rows + 2, part.columns + 2),
          next_u("next_u", part.rows + 2, part.columns + 2),
          next_v("next_v", part.rows + 2, part.columns + 2),
          send_top("send_top", 2 * part.columns),
          send_bottom("send_bottom", 2 * part.columns),
          recv_top("recv_top", 2 * part.columns),
          recv_bottom("recv_bottom", 2 * part.columns) {}

    void swap_fields() {
        std::swap(u, next_u);
        std::swap(v, next_v);
    }
};

template <typename Real>
inline void initialize_fields(Fields<Real>& fields, const StreamSet& streams) {
    const auto u = fields.u;
    const auto v = fields.v;
    Kokkos::parallel_for(
        "initialize fields", Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>(
                                  streams.streams[0],
                                  {0, 0}, {static_cast<int>(u.extent(0)),
                                           static_cast<int>(u.extent(1))}),
        KOKKOS_LAMBDA(const int row, const int column) {
            u(row, column) = Real(0.5) + Real((row + column) % 17) * Real(0.001);
            v(row, column) = Real(0.5) + Real((row + column + 3) % 17) * Real(0.001);
        });
    streams.fence();
}

template <typename Real>
inline void periodic_x(const view_type<Real>& field, const Partition& part,
                       const execution_space& stream) {
    const auto data = field;
    Kokkos::parallel_for(
        "periodic x boundaries", Kokkos::RangePolicy<execution_space>(stream, 0, part.rows + 2),
        KOKKOS_LAMBDA(const int row) {
            data(row, 0) = data(row, part.columns);
            data(row, part.columns + 1) = data(row, 1);
        });
}

template <typename Real>
inline void compute_bulk(const view_type<Real>& u, const view_type<Real>& v,
                         const view_type<Real>& next_u, const view_type<Real>& next_v,
                         const execution_space& stream) {
    coefficients<Real> c;
    gs_compute_interior<Real, true>(u, v, next_u, next_v, c, stream);
}

template <typename Real>
inline void compute_top(const view_type<Real>& u, const view_type<Real>& v,
                        const view_type<Real>& next_u, const view_type<Real>& next_v,
                        const execution_space& stream) {
    coefficients<Real> c;
    gs_compute_top<Real, true>(u, v, next_u, next_v, c, stream);
}

template <typename Real>
inline void compute_bottom(const view_type<Real>& u, const view_type<Real>& v,
                           const view_type<Real>& next_u, const view_type<Real>& next_v,
                           const execution_space& stream) {
    coefficients<Real> c;
    gs_compute_bottom<Real, true>(u, v, next_u, next_v, c, stream);
}

template <typename Real>
inline void pack_row(const view_type<Real>& u, const view_type<Real>& v,
                     const buffer_type<Real>& buffer, int row, const Partition& part,
                     const execution_space& stream) {
    const auto u_ = u;
    const auto v_ = v;
    const auto buffer_ = buffer;
    Kokkos::parallel_for(
        "pack halo row", Kokkos::RangePolicy<execution_space>(stream, 0, part.columns),
        KOKKOS_LAMBDA(const int column) {
            buffer_(column) = u_(row, column + 1);
            buffer_(part.columns + column) = v_(row, column + 1);
        });
}

template <typename Real>
inline void unpack_row(view_type<Real>& u, view_type<Real>& v, const buffer_type<Real>& buffer,
                       int row, const Partition& part, const execution_space& stream) {
    const auto u_ = u;
    const auto v_ = v;
    const auto buffer_ = buffer;
    Kokkos::parallel_for(
        "unpack halo row",
        Kokkos::RangePolicy<execution_space>(stream, 0, part.columns),
        KOKKOS_LAMBDA(const int column) {
            u_(row, column + 1) = buffer_(column);
            v_(row, column + 1) = buffer_(part.columns + column);
        });
}

inline int buffer_count(const Partition& part) { return 2 * part.columns; }

inline int estimate_benchmark_iterations(double warmup_elapsed, int warmup_iterations,
                                         double benchmark_time) {
    if (warmup_elapsed <= 0.0 || warmup_iterations <= 0 || benchmark_time <= 0.0) return 1;
    return std::max(
        1, static_cast<int>(std::ceil(benchmark_time * warmup_iterations / warmup_elapsed)));
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <typename Real = double>
class KernelGraphs {
    using graph_type = Kokkos::Experimental::Graph<execution_space>;

    graph_type before_communication_;
    graph_type after_communication_;

    static graph_type make_before_graph(const Fields<Real>& fields, const Partition& part,
                                        const execution_space& stream) {
        const auto u = fields.u;
        const auto v = fields.v;
        const auto next_u = fields.next_u;
        const auto next_v = fields.next_v;
        const auto send_top = fields.send_top;
        const auto send_bottom = fields.send_bottom;
        const int rows = part.rows;
        const int columns = part.columns;

        return Kokkos::Experimental::create_graph(
            stream, [&](auto root) {
                coefficients<Real> c;
                auto node = root.then_parallel_for(
                    "compute bulk",
                    Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>(
                        stream, {2, 1}, {rows, columns + 1}),
                    KOKKOS_LAMBDA(const int row, const int column) {
                        gs_kernel<Real, true>(row, column, u, v, next_u, next_v, c);
                    });
                auto pack_top = node.then_parallel_for(
                    "pack top", Kokkos::RangePolicy<execution_space>(stream, 0, columns),
                    KOKKOS_LAMBDA(const int column) {
                        send_top(column) = u(1, column + 1);
                        send_top(columns + column) = v(1, column + 1);
                    });
                pack_top.then_parallel_for(
                    "pack bottom",
                    Kokkos::RangePolicy<execution_space>(stream, 0, columns),
                    KOKKOS_LAMBDA(const int column) {
                        send_bottom(column) = u(rows, column + 1);
                        send_bottom(columns + column) = v(rows, column + 1);
                    });
            });
    }

    static graph_type make_after_graph(const Fields<Real>& fields, const Partition& part,
                                       const execution_space& stream) {
        const auto u = fields.u;
        const auto v = fields.v;
        const auto next_u = fields.next_u;
        const auto next_v = fields.next_v;
        const auto recv_top = fields.recv_top;
        const auto recv_bottom = fields.recv_bottom;
        const int columns = part.columns;
        const int bottom_row = part.rows + 1;

        return Kokkos::Experimental::create_graph(
            stream, [&](auto root) {
                auto unpack_top = root.then_parallel_for(
                    "unpack top", Kokkos::RangePolicy<execution_space>(stream, 0, columns),
                    KOKKOS_LAMBDA(const int column) {
                        u(0, column + 1) = recv_top(column);
                        v(0, column + 1) = recv_top(columns + column);
                    });
                coefficients<Real> c;
                auto compute_top = unpack_top.then_parallel_for(
                    "compute top",
                    Kokkos::RangePolicy<execution_space>(stream, 1, columns + 1),
                    KOKKOS_LAMBDA(const int column) {
                        gs_kernel<Real, true>(1, column, u, v, next_u, next_v, c);
                    });
                auto unpack_bottom = compute_top.then_parallel_for(
                    "unpack bottom",
                    Kokkos::RangePolicy<execution_space>(stream, 0, columns),
                    KOKKOS_LAMBDA(const int column) {
                        u(bottom_row, column + 1) = recv_bottom(column);
                        v(bottom_row, column + 1) = recv_bottom(columns + column);
                    });
                unpack_bottom.then_parallel_for(
                    "compute bottom",
                    Kokkos::RangePolicy<execution_space>(stream, 1, columns + 1),
                    KOKKOS_LAMBDA(const int column) {
                        gs_kernel<Real, true>(bottom_row - 1, column, u, v, next_u, next_v, c);
                    });
            });
    }

public:
    KernelGraphs(const Fields<Real>& fields, const Partition& part,
                 const execution_space& stream)
        : before_communication_(make_before_graph(fields, part, stream)),
          after_communication_(make_after_graph(fields, part, stream)) {}

    void submit_before() const { before_communication_.submit(); }
    void submit_after() const { after_communication_.submit(); }
};
#endif

}  // namespace gs_bench
