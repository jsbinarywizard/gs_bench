#include "../include/benchmarks/gs_gpu_common.hpp"
#include "../include/peripherie/output.hpp"

#include <memory>

using namespace gs_bench;

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
    Fields<> fields(part);
    initialize_fields(fields, streams);
    MPI_Request requests[4];
    GS_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto execute_iteration = [&]() {
        if (args.config.periodic[0]) {
            periodic_x(fields.u, part, streams.streams[0]);
            periodic_x(fields.v, part, streams.streams[0]);
            streams.streams[0].fence();
        }
        compute_bulk(fields.u, fields.v, fields.next_u, fields.next_v, streams.streams[0]);
        streams.streams[0].fence();

        GS_MPI_CHECK(MPI_Irecv(fields.recv_top.data(), buffer_count(part), mpi_datatype<>(),
                               part.top_rank(), mpi_tag_bottom, MPI_COMM_WORLD, &requests[0]));
        GS_MPI_CHECK(MPI_Irecv(fields.recv_bottom.data(), buffer_count(part), mpi_datatype<>(),
                               part.bottom_rank(), mpi_tag_top, MPI_COMM_WORLD, &requests[1]));
        pack_row(fields.u, fields.v, fields.send_top, 1, part, streams.streams[3]);
        pack_row(fields.u, fields.v, fields.send_bottom, part.rows, part, streams.streams[4]);
        streams.streams[3].fence();
        streams.streams[4].fence();
        GS_MPI_CHECK(MPI_Isend(fields.send_top.data(), buffer_count(part), mpi_datatype<>(),
                               part.top_rank(), mpi_tag_top, MPI_COMM_WORLD, &requests[2]));
        GS_MPI_CHECK(MPI_Isend(fields.send_bottom.data(), buffer_count(part), mpi_datatype<>(),
                               part.bottom_rank(), mpi_tag_bottom, MPI_COMM_WORLD, &requests[3]));
        for (int receives_completed = 0; receives_completed < 2; ++receives_completed) {
            int completed_request = MPI_UNDEFINED;
            GS_MPI_CHECK(MPI_Waitany(2, requests, &completed_request, MPI_STATUS_IGNORE));
            if (completed_request == 0) {
                unpack_row(fields.u, fields.v, fields.recv_top, 0, part, streams.streams[1]);
                compute_top(fields.u, fields.v, fields.next_u, fields.next_v, streams.streams[1]);
            } else if (completed_request == 1) {
                unpack_row(fields.u, fields.v, fields.recv_bottom, part.rows + 1, part,
                           streams.streams[2]);
                compute_bottom(fields.u, fields.v, fields.next_u, fields.next_v,
                              streams.streams[2]);
            }
        }
        GS_MPI_CHECK(MPI_Waitall(2, &requests[2], MPI_STATUSES_IGNORE));
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
        results_writer->write({"mpi-isend-irecv", precision_name<double>(), test.scaling,
                               rank_count, test.size, benchmark_iterations, elapsed,
                               benchmark_iterations / elapsed});
    }
    if (part.rank == 0)
        std::printf("mpi-isend-irecv,%s,%d,%.6f,%.3f\n",
                    scaling_name(test.scaling), test.size, elapsed,
                    benchmark_iterations / elapsed);
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
