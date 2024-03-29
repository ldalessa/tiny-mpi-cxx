#include "tiny_mpi/tiny_mpi.hpp"
#include <cstdio>

bool
tiny_mpi::initialized(sloc_t sloc)
    noexcept
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
tiny_mpi::finalized(sloc_t sloc)
    noexcept
{
    int mpi_finalized;
    if (int e = MPI_Finalized(&mpi_finalized)) {
        fprintf(stderr, "%s:%s MPI_Finalized failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
    return mpi_finalized;
}

auto
tiny_mpi::init(tiny_mpi::thread_support_t support,
               sloc_t sloc) noexcept
    -> tiny_mpi::thread_support_t
{
    int out;
    if (initialized(sloc)) {
        check(sloc_t::current(), tiny_mpi_check_op(MPI_Query_thread), &out);
        return static_cast<thread_support_t>(out);
    }

    if (int e = MPI_Init_thread(nullptr, nullptr, support, &out)) {
        fprintf(stderr, "%s:%s MPI_Init failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
    return static_cast<thread_support_t>(out);
}

void
tiny_mpi::fini(sloc_t sloc)
    noexcept
{
    if (!initialized(sloc)) {
        return;
    }

    if (int e = MPI_Finalize()) {
        fprintf(stderr, "%s:%s MPI_Finalize failed with (%d)\n",
                sloc.function_name(), sloc.line(), e);
        exit(EXIT_FAILURE);
    }
}

void
tiny_mpi::abort(int e, sloc_t sloc)
    noexcept
{
    if (!initialized(sloc)) {
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
tiny_mpi::print_error(const char* f, int e, sloc_t sloc)
    noexcept
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
tiny_mpi::probe(rank_t source, MPI_Datatype type, tag_t tag, sloc_t sloc)
    noexcept
{
    MPI_Status status;
    check(sloc, tiny_mpi_check_op(MPI_Probe), source, tag, MPI_COMM_WORLD, &status);

    int n;
    check(sloc, tiny_mpi_check_op(MPI_Get_count), &status, type, &n);
    return n;
}

void
tiny_mpi::wait(std::span<request_t> reqs, sloc_t sloc)
    noexcept
{
    check(sloc, tiny_mpi_check_op(MPI_Waitall), ssize(reqs), data(reqs), MPI_STATUSES_IGNORE);
}
