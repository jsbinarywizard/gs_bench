// Gray-Scott halo exchange benchmark: MPI vs RCCL (Kokkos HIP backend)
//
// Self-contained rewrite, structured like the working bench_mpi_rccl.cpp:
//   * MPI_Init first, then Kokkos::initialize, then own hipStreams, then communicators
//   * communicators are created ONCE (not per benchmark case)
//   * Kokkos execution-space instances wrap our own hipStreams, so RCCL and Kokkos
//     kernels share exactly the same streams
//   * stdout is line buffered so output survives a crash
//
// Decomposition: 1D over rows, periodic ring in y (rank ring) and in x (modulo in kernel).
// Local field: (rows+2) x cols, row 0 and row rows+1 are ghost rows.
//
// Variants:
//   mpi    : Irecv/Isend, device pointers (needs MPICH_GPU_SUPPORT_ENABLED=1)
//   rccl1  : one communicator, one comm stream, one group with all four P2P ops
//   rccl2  : two communicators + two streams (one per direction), send/recv grouped
//
// Build (example):
//   hipcc/mpicxx -O3 gs_halo_bench.cpp -o gs_halo_bench $(Kokkos flags) -lrccl -lmpi
// Run (one GPU per rank):
//   srun --gpus-per-task=1 --gpu-bind=closest ./gs_halo_bench --sizes 512,2048 --time 1.0
//   options: --sizes a,b,c   --time seconds   --strong   --variants mpi,rccl1,rccl2
//            --output file.csv   (default gs_halo_results.csv; results are APPENDED, the header
//                                 is only written to an empty file, so runs with different rank
//                                 counts can accumulate in one file; rank 0 writes it)

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <rccl/rccl.h>   // may be <rccl.h> depending on install
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using Exec   = Kokkos::HIP;
using Field  = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HIPSpace>;
using Buffer = Kokkos::View<double*, Kokkos::HIPSpace>;

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
  MPI_Abort(MPI_COMM_WORLD, 1);} } while (0)
#define NCCL_CHECK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
  std::fprintf(stderr, "RCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); \
  MPI_Abort(MPI_COMM_WORLD, 1);} } while (0)
#define MPI_CHECK(x) do { int e_ = (x); if (e_ != MPI_SUCCESS) { \
  std::fprintf(stderr, "MPI error %d at %s:%d\n", e_, __FILE__, __LINE__); \
  MPI_Abort(MPI_COMM_WORLD, 1);} } while (0)

enum class Variant { MPI = 0, RCCL1 = 1, RCCL2 = 2 };
static const char* variant_name(Variant v) {
  switch (v) {
    case Variant::MPI:   return "mpi";
    case Variant::RCCL1: return "rccl-1comm";
    default:             return "rccl-2comm";
  }
}

constexpr int TAG_TOP = 100;     // message that carries a rank's top row
constexpr int TAG_BOTTOM = 101;  // message that carries a rank's bottom row

struct Domain {
  int rank, size, rows, cols, row_offset, top, bottom;
};

struct State {
  Field u, v, nu, nv;
  Buffer send_top, send_bottom, recv_top, recv_bottom;
  State(int rows, int cols)
    : u("u", rows + 2, cols), v("v", rows + 2, cols),
      nu("nu", rows + 2, cols), nv("nv", rows + 2, cols),
      send_top("send_top", 2 * cols), send_bottom("send_bottom", 2 * cols),
      recv_top("recv_top", 2 * cols), recv_bottom("recv_bottom", 2 * cols) {}
  void swap_fields() { std::swap(u, nu); std::swap(v, nv); }
};

struct Ctx {
  Domain d;
  Exec e0, e1, e2;           // e0: bulk, e1/e2: halo paths
  hipStream_t h1, h2;        // native streams behind e1 / e2
  ncclComm_t comm1, commA, commB;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
static void init_state(State& s, const Ctx& x) {
  Field u = s.u, v = s.v;
  const int C = x.d.cols, rows = x.d.rows, off = x.d.row_offset;
  Kokkos::parallel_for("init",
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>(x.e0, {1, 0}, {rows + 1, C}),
    KOKKOS_LAMBDA(const int r, const int c) {
      const int g = off + r - 1;
      u(r, c) = 0.5 + double((g + c) % 17) * 0.001;
      v(r, c) = 0.5 + double((g + c + 3) % 17) * 0.001;
    });
  x.e0.fence();
}

// Gray-Scott update for rows [r0, r1) of the local field (rows are 1..rows).
static void compute_rows(const State& s, Exec exec, int r0, int r1) {
  if (r1 <= r0) return;
  Field u = s.u, v = s.v, nu = s.nu, nv = s.nv;
  const int C = int(s.u.extent(1));
  const double Du = 0.2, Dv = 0.1, F = 0.04, k = 0.06, dt = 1.0;
  Kokkos::parallel_for("gs_update",
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>(exec, {r0, 0}, {r1, C}),
    KOKKOS_LAMBDA(const int r, const int c) {
      const int cl = (c == 0) ? C - 1 : c - 1;
      const int cr = (c == C - 1) ? 0 : c + 1;
      const double uc = u(r, c), vc = v(r, c);
      const double lu = u(r - 1, c) + u(r + 1, c) + u(r, cl) + u(r, cr) - 4.0 * uc;
      const double lv = v(r - 1, c) + v(r + 1, c) + v(r, cl) + v(r, cr) - 4.0 * vc;
      const double uvv = uc * vc * vc;
      nu(r, c) = uc + dt * (Du * lu - uvv + F * (1.0 - uc));
      nv(r, c) = vc + dt * (Dv * lv + uvv - (F + k) * vc);
    });
}

static void pack(const State& s, Exec exec, Buffer buf, int row) {
  Field u = s.u, v = s.v;
  const int C = int(s.u.extent(1));
  Kokkos::parallel_for("pack", Kokkos::RangePolicy<Exec>(exec, 0, C),
    KOKKOS_LAMBDA(const int c) {
      buf(c) = u(row, c);
      buf(C + c) = v(row, c);
    });
}

static void unpack(const State& s, Exec exec, Buffer buf, int row) {
  Field u = s.u, v = s.v;
  const int C = int(s.u.extent(1));
  Kokkos::parallel_for("unpack", Kokkos::RangePolicy<Exec>(exec, 0, C),
    KOKKOS_LAMBDA(const int c) {
      u(row, c) = buf(c);
      v(row, c) = buf(C + c);
    });
}

static void fence_all(const Ctx& x) { x.e0.fence(); x.e1.fence(); x.e2.fence(); }

// ---------------------------------------------------------------------------
// One time step per variant
// ---------------------------------------------------------------------------
static void step_mpi(State& s, const Ctx& x) {
  const Domain& d = x.d;
  const int n = 2 * d.cols;
  MPI_Request rq[4];
  MPI_CHECK(MPI_Irecv(s.recv_top.data(), n, MPI_DOUBLE, d.top, TAG_BOTTOM, MPI_COMM_WORLD, &rq[0]));
  MPI_CHECK(MPI_Irecv(s.recv_bottom.data(), n, MPI_DOUBLE, d.bottom, TAG_TOP, MPI_COMM_WORLD, &rq[1]));
  pack(s, x.e1, s.send_top, 1);
  pack(s, x.e2, s.send_bottom, d.rows);
  compute_rows(s, x.e0, 2, d.rows);                       // bulk overlaps communication
  x.e1.fence();
  MPI_CHECK(MPI_Isend(s.send_top.data(), n, MPI_DOUBLE, d.top, TAG_TOP, MPI_COMM_WORLD, &rq[2]));
  x.e2.fence();
  MPI_CHECK(MPI_Isend(s.send_bottom.data(), n, MPI_DOUBLE, d.bottom, TAG_BOTTOM, MPI_COMM_WORLD, &rq[3]));
  for (int k = 0; k < 2; ++k) {
    int idx = MPI_UNDEFINED;
    MPI_CHECK(MPI_Waitany(2, rq, &idx, MPI_STATUS_IGNORE));
    if (idx == 0) {
      unpack(s, x.e1, s.recv_top, 0);
      compute_rows(s, x.e1, 1, 2);
    } else {
      unpack(s, x.e2, s.recv_bottom, d.rows + 1);
      compute_rows(s, x.e2, d.rows, d.rows + 1);
    }
  }
  MPI_CHECK(MPI_Waitall(2, &rq[2], MPI_STATUSES_IGNORE));
  fence_all(x);
  s.swap_fields();
}

// One communicator, one comm stream (h1/e1), one group with all four ops.
// NCCL has no tags: ops to the same peer match in issue order. If top == bottom
// (1 or 2 ranks) my send_top must land in the peer's recv_bottom, so the receives
// are issued in the order (bottom, top) while the sends are issued (top, bottom).
static void step_rccl1(State& s, const Ctx& x) {
  const Domain& d = x.d;
  const int n = 2 * d.cols;
  pack(s, x.e1, s.send_top, 1);
  pack(s, x.e1, s.send_bottom, d.rows);
  compute_rows(s, x.e0, 2, d.rows);
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclSend(s.send_top.data(), n, ncclDouble, d.top, x.comm1, x.h1));
  NCCL_CHECK(ncclSend(s.send_bottom.data(), n, ncclDouble, d.bottom, x.comm1, x.h1));
  NCCL_CHECK(ncclRecv(s.recv_bottom.data(), n, ncclDouble, d.bottom, x.comm1, x.h1));
  NCCL_CHECK(ncclRecv(s.recv_top.data(), n, ncclDouble, d.top, x.comm1, x.h1));
  NCCL_CHECK(ncclGroupEnd());
  unpack(s, x.e1, s.recv_top, 0);
  unpack(s, x.e1, s.recv_bottom, d.rows + 1);
  compute_rows(s, x.e1, 1, 2);
  compute_rows(s, x.e1, d.rows, d.rows + 1);
  fence_all(x);
  s.swap_fields();
}

// Two communicators, two streams, one per direction.
//   "up"   (commA, h1/e1): send my top row -> top,     recv bottom ghost <- bottom
//   "down" (commB, h2/e2): send my bottom row -> bottom, recv top ghost  <- top
static void step_rccl2(State& s, const Ctx& x) {
  const Domain& d = x.d;
  const int n = 2 * d.cols;
  pack(s, x.e1, s.send_top, 1);
  pack(s, x.e2, s.send_bottom, d.rows);
  compute_rows(s, x.e0, 2, d.rows);

  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclSend(s.send_top.data(), n, ncclDouble, d.top, x.commA, x.h1));
  NCCL_CHECK(ncclRecv(s.recv_bottom.data(), n, ncclDouble, d.bottom, x.commA, x.h1));
  NCCL_CHECK(ncclGroupEnd());

  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclSend(s.send_bottom.data(), n, ncclDouble, d.bottom, x.commB, x.h2));
  NCCL_CHECK(ncclRecv(s.recv_top.data(), n, ncclDouble, d.top, x.commB, x.h2));
  NCCL_CHECK(ncclGroupEnd());

  unpack(s, x.e1, s.recv_bottom, d.rows + 1);
  compute_rows(s, x.e1, d.rows, d.rows + 1);
  unpack(s, x.e2, s.recv_top, 0);
  compute_rows(s, x.e2, 1, 2);
  fence_all(x);
  s.swap_fields();
}

static void step(Variant v, State& s, const Ctx& x) {
  switch (v) {
    case Variant::MPI:   step_mpi(s, x);   break;
    case Variant::RCCL1: step_rccl1(s, x); break;
    default:             step_rccl2(s, x); break;
  }
}

// ---------------------------------------------------------------------------
// Checksum / correctness / timing
// ---------------------------------------------------------------------------
static double checksum(const State& s, const Ctx& x) {
  Field u = s.u;
  const int C = x.d.cols, rows = x.d.rows;
  double local = 0.0;
  Kokkos::parallel_reduce("checksum",
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>(x.e0, {1, 0}, {rows + 1, C}),
    KOKKOS_LAMBDA(const int r, const int c, double& acc) { acc += u(r, c); }, local);
  x.e0.fence();
  double global = 0.0;
  MPI_CHECK(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  return global;
}

static double run_checksum(Variant v, State& s, const Ctx& x) {
  init_state(s, x);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  for (int i = 0; i < 5; ++i) step(v, s, x);
  return checksum(s, x);
}

// Returns seconds per step (max over ranks); iters_out gets the iteration count.
static double run_timing(Variant v, State& s, const Ctx& x, double target, int& iters_out) {
  init_state(s, x);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  for (int i = 0; i < 10; ++i) step(v, s, x);              // warmup
  fence_all(x);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  double t0 = MPI_Wtime();
  for (int i = 0; i < 10; ++i) step(v, s, x);              // calibration
  fence_all(x);
  double per = (MPI_Wtime() - t0) / 10.0, per_max = 0.0;
  MPI_CHECK(MPI_Allreduce(&per, &per_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  const int iters = std::max(20, int(std::ceil(target / per_max)));   // same on all ranks
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  t0 = MPI_Wtime();
  for (int i = 0; i < iters; ++i) step(v, s, x);
  fence_all(x);
  double elapsed = MPI_Wtime() - t0, elapsed_max = 0.0;
  MPI_CHECK(MPI_Allreduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  iters_out = iters;
  return elapsed_max / iters;
}

static ncclComm_t make_comm(int rank, int size) {
  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
  return comm;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  std::setvbuf(stdout, nullptr, _IOLBF, 0);   // survive crashes / aborts

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // ---- options ----
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
  double target = 1.0;
  bool strong = false;
  std::string out_path = "gs_halo_results.csv";
  bool want[3] = {true, true, true};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) {
      sizes.clear();
      std::string list = argv[++i];
      char* tok = std::strtok(&list[0], ",");
      while (tok) { sizes.push_back(std::atoi(tok)); tok = std::strtok(nullptr, ","); }
    } else if (!std::strcmp(argv[i], "--time") && i + 1 < argc) {
      target = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) {
      out_path = argv[++i];
    } else if (!std::strcmp(argv[i], "--strong")) {
      strong = true;
    } else if (!std::strcmp(argv[i], "--variants") && i + 1 < argc) {
      const std::string list = argv[++i];
      want[0] = list.find("mpi") != std::string::npos;
      want[1] = list.find("rccl1") != std::string::npos;
      want[2] = list.find("rccl2") != std::string::npos;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    // Streams shared by Kokkos and RCCL; they must outlive Exec instances and comms.
    hipStream_t h0, h1, h2;
    HIP_CHECK(hipStreamCreate(&h0));
    HIP_CHECK(hipStreamCreate(&h1));
    HIP_CHECK(hipStreamCreate(&h2));

    ncclComm_t comm1 = nullptr, commA = nullptr, commB = nullptr;
    if (want[1]) comm1 = make_comm(rank, size);
    if (want[2]) { commA = make_comm(rank, size); commB = make_comm(rank, size); }

    static const char* csv_header =
        "variant,scaling,ranks,size,iterations,sec_per_step,updates_per_s,check\n";
    std::FILE* out = nullptr;   // only rank 0 writes the file
    if (rank == 0) {
      std::printf("# ranks=%d scaling=%s target_time=%.2fs output=%s\n", size,
                  strong ? "strong" : "weak", target, out_path.c_str());
      std::printf("%s", csv_header);
      out = std::fopen(out_path.c_str(), "a");
      if (!out) {
        std::fprintf(stderr, "Cannot open output file '%s'\n", out_path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::fseek(out, 0, SEEK_END);
      if (std::ftell(out) == 0) std::fputs(csv_header, out);   // header only for a new file
      std::fflush(out);
    }

    {   // everything holding Exec instances or Views lives in this scope
      Exec e0(h0), e1(h1), e2(h2);

      for (const int n : sizes) {
        int rows = n, cols = n;
        if (strong) {
          if (n % size != 0) {
            if (rank == 0) std::printf("# skip size %d: not divisible by %d ranks\n", n, size);
            continue;
          }
          rows = n / size;
        }
        if (rows < 3) {
          if (rank == 0) std::printf("# skip size %d: fewer than 3 rows per rank\n", n);
          continue;
        }

        Domain d;
        d.rank = rank; d.size = size; d.rows = rows; d.cols = cols;
        d.row_offset = rank * rows;
        d.top = (rank + size - 1) % size;
        d.bottom = (rank + 1) % size;
        Ctx ctx{d, e0, e1, e2, h1, h2, comm1, commA, commB};

        {   // Views are destroyed before the next size is allocated
          State s(rows, cols);

          // Correctness: every variant must reproduce the MPI checksum.
          const double ref = run_checksum(Variant::MPI, s, ctx);

          for (int vi = 0; vi < 3; ++vi) {
            if (!want[vi]) continue;
            const Variant v = static_cast<Variant>(vi);
            const double chk = run_checksum(v, s, ctx);
            const bool ok = std::abs(chk - ref) <= 1e-10 * std::max(1.0, std::abs(ref));

            int iters = 0;
            const double per_step = run_timing(v, s, ctx, target, iters);
            if (rank == 0) {
              const double updates = double(rows) * cols * size / per_step;
              char line[256];
              std::snprintf(line, sizeof(line), "%s,%s,%d,%d,%d,%.6e,%.6e,%s\n",
                            variant_name(v), strong ? "strong" : "weak", size, n, iters,
                            per_step, updates, ok ? "PASS" : "FAIL");
              std::fputs(line, stdout);
              std::fputs(line, out);
              std::fflush(out);   // keep results even if a later case crashes
            }
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      fence_all(Ctx{Domain{}, e0, e1, e2, h1, h2, nullptr, nullptr, nullptr});
    }   // Exec instances destroyed while the streams are still valid

    if (out) std::fclose(out);
    if (comm1) NCCL_CHECK(ncclCommDestroy(comm1));
    if (commA) NCCL_CHECK(ncclCommDestroy(commA));
    if (commB) NCCL_CHECK(ncclCommDestroy(commB));
    HIP_CHECK(hipStreamDestroy(h0));
    HIP_CHECK(hipStreamDestroy(h1));
    HIP_CHECK(hipStreamDestroy(h2));
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
