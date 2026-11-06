#include "../include/benchmarks/gs_gpu_common.hpp"
#include "../include/peripherie/output.hpp"

#include <memory>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define GS_NCCL_CHECK(call) do { \
    const ncclResult_t status = (call); \
    if (status != ncclSuccess) { \
        std::fprintf(stderr, "NCCL/RCCL call failed: %s\n", ncclGetErrorString(status)); \
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    } \
} while (false)
#endif

using namespace gs_bench;

// Variant 2: two communicators, two streams, one per direction. Send and recv of a
// direction are grouped together on that direction's communicator and stream.
//
//   "up" direction   (comm_up,   stream 1): send my top row    -> top neighbour
//                                           recv bottom ghost  <- bottom neighbour
//                    pack(top) -> group -> unpack(bottom ghost) -> compute_bottom
//
//   "down" direction (comm_down, stream 2): send my bottom row -> bottom neighbour
//                                           recv top ghost     <- top neighbour
//                    pack(bottom) -> group -> unpack(top ghost) -> compute_top
//
//   stream 0: compute_bulk
//
// Tip: export CUDA_DEVICE_MAX_CONNECTIONS=32 so that the concurrent streams can overlap.
int run(int argc, char** argv) {
    GS_MPI_CHECK(MPI_Init(&argc, &argv));
    const Arguments args = parse_arguments(argc, argv);
    int rank_count = 0;
    int rank = 0;
    GS_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &rank_count));
    GS_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    std::unique_ptr<ResultsWriter> results_writer;
    try {
        results_writer = std::make_unique<ResultsWriter>(args.config.output_file, rank == 0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (const BenchmarkCase& test : make_cases(args.config, rank_count)) {
    const Partition part(test.global_rows, test.columns);
    const StreamSet streams;

    // One unique id per communicator.
    ncclUniqueId id_up;
    ncclUniqueId id_down;
    if (part.rank == 0) {
        GS_NCCL_CHECK(ncclGetUniqueId(&id_up));
        GS_NCCL_CHECK(ncclGetUniqueId(&id_down));
    }
    GS_MPI_CHECK(MPI_Bcast(&id_up, sizeof(id_up), MPI_BYTE, 0, MPI_COMM_WORLD));
    GS_MPI_CHECK(MPI_Bcast(&id_down, sizeof(id_down), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t comm_up;
    ncclComm_t comm_down;
    GS_NCCL_CHECK(ncclCommInitRank(&comm_up, part.size, id_up, part.rank));
    GS_NCCL_CHECK(ncclCommInitRank(&comm_down, part.size, id_down, part.rank));

    Fields<> fields(part);
    initialize_fields(fields, streams);
    GS_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto execute_iteration = [&]() {
        if (args.config.periodic[0]) {
            periodic_x(fields.u, part, streams.streams[0]);
            periodic_x(fields.v, part, streams.streams[1]);
            streams.streams[0].fence();
            streams.streams[1].fence();
        }
        // Pack, communicate and unpack of each direction share one stream,
        // so stream order provides all dependencies.
        pack_row(fields.u, fields.v, fields.send_top, 1, part, streams.streams[1]);
        pack_row(fields.u, fields.v, fields.send_bottom, part.rows, part, streams.streams[2]);
        compute_bulk(fields.u, fields.v, fields.next_u, fields.next_v, streams.streams[0]);

        // "up": my top row goes to the top neighbour, I receive my bottom ghost row
        // from the bottom neighbour (who sends its top row).
        GS_NCCL_CHECK(ncclGroupStart());
        GS_NCCL_CHECK(ncclSend(fields.send_top.data(), buffer_count(part), nccl_datatype<>(),
                               part.top_rank(), comm_up, streams.native(1)));
        GS_NCCL_CHECK(ncclRecv(fields.recv_bottom.data(), buffer_count(part), nccl_datatype<>(),
                               part.bottom_rank(), comm_up, streams.native(1)));
        GS_NCCL_CHECK(ncclGroupEnd());

        // "down": my bottom row goes to the bottom neighbour, I receive my top ghost row
        // from the top neighbour (who sends its bottom row).
        GS_NCCL_CHECK(ncclGroupStart());
        GS_NCCL_CHECK(ncclSend(fields.send_bottom.data(), buffer_count(part), nccl_datatype<>(),
                               part.bottom_rank(), comm_down, streams.native(2)));
        GS_NCCL_CHECK(ncclRecv(fields.recv_top.data(), buffer_count(part), nccl_datatype<>(),
                               part.top_rank(), comm_down, streams.native(2)));
        GS_NCCL_CHECK(ncclGroupEnd());

        unpack_row(fields.u, fields.v, fields.recv_bottom, part.rows + 1, part, streams.streams[1]);
        compute_bottom(fields.u, fields.v, fields.next_u, fields.next_v, streams.streams[1]);
        unpack_row(fields.u, fields.v, fields.recv_top, 0, part, streams.streams[2]);
        compute_top(fields.u, fields.v, fields.next_u, fields.next_v, streams.streams[2]);
        streams.fence();
        fields.swap_fields();
    };
    const double warmup_start = MPI_Wtime();
    int warmup_iterations = 0;
    while (true) {
        int continue_warmup = MPI_Wtime() - warmup_start < args.config.warmup_time;
        GS_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &continue_warmup, 1, MPI_INT, MPI_MIN,
                                   MPI_COMM_WORLD));
        if (!continue_warmup) break;
        execute_iteration();
        ++warmup_iterations;
    }
    streams.fence();
    const double warmup_elapsed = MPI_Wtime() - warmup_start;
    double slowest_warmup_elapsed = 0.0;
    GS_MPI_CHECK(MPI_Allreduce(&warmup_elapsed, &slowest_warmup_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                               MPI_COMM_WORLD));
    int benchmark_iterations = 0;
    if (part.rank == 0) {
        benchmark_iterations = estimate_benchmark_iterations(
            slowest_warmup_elapsed, warmup_iterations, args.config.benchmark_time);
    }
    GS_MPI_CHECK(MPI_Bcast(&benchmark_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD));
    GS_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start_time = MPI_Wtime();
    for (int iterations = 0; iterations < benchmark_iterations; ++iterations) {
        execute_iteration();
    }
    streams.fence();
    const double elapsed = MPI_Wtime() - start_time;
    if (part.rank == 0) {
        results_writer->write({"nccl-rccl-2comm", precision_name<double>(), test.scaling, rank_count,
                               test.size, benchmark_iterations, elapsed,
                               (benchmark_iterations * test.global_rows * test.columns) / elapsed});
    }

    GS_NCCL_CHECK(ncclCommDestroy(comm_up));
    GS_NCCL_CHECK(ncclCommDestroy(comm_down));
    }
    GS_MPI_CHECK(MPI_Finalize());
    return 0;
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    const int result = run(argc, argv);
    Kokkos::finalize();
    return result;
}