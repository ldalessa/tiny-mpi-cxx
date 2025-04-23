#include "tiny_mpi/tiny_mpi.hpp"
#include <cstdio>

bool
tiny_mpi::initialized(std::source_location sloc)
{
    int mpi_initialized;
    if (int e = MPI_Initialized(&mpi_initialized)) {
        fprintf(stderr, "%s:%d MPI_Initialized failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
    return mpi_initialized != 0;
}

bool
tiny_mpi::finalized(std::source_location sloc)
{
    int mpi_finalized;
    if (int e = MPI_Finalized(&mpi_finalized)) {
        fprintf(stderr, "%s:%d MPI_Finalized failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
    return mpi_finalized;
}

auto
tiny_mpi::init(tiny_mpi::thread_support_t support,
               std::source_location sloc)
    -> tiny_mpi::thread_support_t
{
    int out;
    if (initialized(sloc)) {
        check(std::source_location::current(), tiny_mpi_check_op(MPI_Query_thread), &out);
        return static_cast<thread_support_t>(out);
    }

    if (int e = MPI_Init_thread(nullptr, nullptr, support, &out)) {
        fprintf(stderr, "%s:%d MPI_Init failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
    return static_cast<thread_support_t>(out);
}

void
tiny_mpi::fini(std::source_location sloc)
{
    if (!initialized(sloc)) {
        return;
    }

    if (int e = MPI_Finalize()) {
        fprintf(stderr, "%s:%d MPI_Finalize failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
}

void
tiny_mpi::abort(int e, std::source_location sloc)
{
    if (!initialized(sloc)) {
        fprintf(stderr, "%s:%d MPI_Abort called with error code (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }

    int error = MPI_Abort(MPI_COMM_WORLD, e);
    fprintf(stderr, "%s:%d MPI_Abort failed with error code (%d)\n",
            sloc.function_name(), sloc.line(), error);
    exit(EXIT_FAILURE);
}

void
tiny_mpi::print_error(const char* f, int e, std::source_location sloc)
{
    char str[MPI_MAX_ERROR_STRING];
    int _;
    if (MPI_Error_string(e, str, &_)) {
        str[0] = '\0';
    }
    fprintf(stderr, "%s:%d %s returned error %s (%d)\n",
            sloc.function_name(), sloc.line(), f, str, e);
}

int
tiny_mpi::probe(rank_t source, MPI_Datatype type, tag_t tag, std::source_location sloc)
{
    MPI_Status status;
    check(sloc, tiny_mpi_check_op(MPI_Probe), source, tag, MPI_COMM_WORLD, &status);

    int n;
    check(sloc, tiny_mpi_check_op(MPI_Get_count), &status, type, &n);
    return n;
}

void
tiny_mpi::wait(std::span<request_t> reqs, std::source_location sloc)
{
    check(sloc, tiny_mpi_check_op(MPI_Waitall), ssize(reqs), data(reqs), MPI_STATUSES_IGNORE);
}
