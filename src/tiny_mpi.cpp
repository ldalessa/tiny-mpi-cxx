#include "tiny_mpi/tiny_mpi.hpp"
#include <cstdio>

bool
tiny_mpi::mpi_initialized(sloc_t sloc)
{
  int mpi_initialized;
  if (int e = MPI_Initialized(&mpi_initialized)) {
    fprintf(stderr, "%s:%s MPI_Initialized failed with (%d)\n",
            sloc.function_name(), sloc.line(), e);
    exit(EXIT_FAILURE);
  }
  return mpi_initialized != 0;
}

bool
tiny_mpi::mpi_finalized(sloc_t sloc)
{
  int mpi_finalized;
  if (int e = MPI_Finalized(&mpi_finalized)) {
    fprintf(stderr, "%s:%s MPI_Finalized failed with (%d)\n",
            sloc.function_name(), sloc.line(), e);
    exit(EXIT_FAILURE);
  }
  return mpi_finalized;
}

void
tiny_mpi::mpi_init(sloc_t sloc)
{
  if (mpi_initialized(sloc)) {
    return;
  }

  if (int e = MPI_Init(nullptr, nullptr)) {
    fprintf(stderr, "%s:%s MPI_Init failed with (%d)\n",
            sloc.function_name(), sloc.line(), e);
    exit(EXIT_FAILURE);
  }
}

void
tiny_mpi::mpi_fini(sloc_t sloc)
{
  if (!mpi_initialized(sloc)) {
    return;
  }

  if (int e = MPI_Finalize()) {
    fprintf(stderr, "%s:%s MPI_Finalize failed with (%d)\n",
            sloc.function_name(), sloc.line(), e);
    exit(EXIT_FAILURE);
  }
}

void
tiny_mpi::mpi_abort(int e, sloc_t sloc)
{
  if (!mpi_initialized(sloc)) {
    fprintf(stderr, "%s:%s MPI_Abort called with error code (%d)\n",
            sloc.function_name(), sloc.line(), e);
    exit(EXIT_FAILURE);
  }

  int error = MPI_Abort(MPI_COMM_WORLD, e);
  fprintf(stderr, "%s:%s MPI_Abort failed with error code (%d)\n",
          sloc.function_name(), sloc.line(), error);
  exit(EXIT_FAILURE);
}

void
tiny_mpi::mpi_print_error(const char* f, int e, sloc_t sloc)
{
  char str[MPI_MAX_ERROR_STRING];
  int _;
  if (MPI_Error_string(e, str, &_)) {
    str[0] = '\0';
  }
  fprintf(stderr, "%s:%s %s returned error %s (%d)\n",
          sloc.function_name(), sloc.line(), f, str, e);
}

int
tiny_mpi::mpi_probe(int source, int tag, MPI_Datatype type, sloc_t sloc)
{
  MPI_Status status;
  mpi_check(sloc, MPI_Probe, source, tag, MPI_COMM_WORLD, &status);

  int n;
  mpi_check(sloc, MPI_Get_count, &status, type, &n);
  return n;
}

void
tiny_mpi::mpi_wait(std::span<MPI_Request> reqs, sloc_t sloc)
{
  mpi_check(sloc, MPI_Waitall, ssize(reqs), reqs.data(), MPI_STATUSES_IGNORE);
}
