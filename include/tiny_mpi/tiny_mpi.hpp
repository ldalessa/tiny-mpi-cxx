#ifndef TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP
#define TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP

#ifdef __cpp_lib_source_location
#include <source_location>
#else
#include <experimental/source_location>
#endif

#include <span>
#include <vector>
#include <mpi.h>

namespace tiny_mpi
{

#ifdef __cpp_lib_source_location
using sloc_t = std::source_location;
#else
using sloc_t = std::experimental::source_location;
#endif

/// Simple variadic template to map arithmatic types to their MPI equivalents.
template <class T> constexpr MPI_Datatype mpi_type = not defined(mpi_type<T>);
template <> constexpr MPI_Datatype mpi_type<std::byte>          = MPI_BYTE;
template <> constexpr MPI_Datatype mpi_type<char>               = MPI_CHAR;
template <> constexpr MPI_Datatype mpi_type<signed char>        = MPI_CHAR;
template <> constexpr MPI_Datatype mpi_type<short>              = MPI_SHORT;
template <> constexpr MPI_Datatype mpi_type<int>                = MPI_INT;
template <> constexpr MPI_Datatype mpi_type<long>               = MPI_LONG;
template <> constexpr MPI_Datatype mpi_type<long long>          = MPI_LONG_LONG_INT;
template <> constexpr MPI_Datatype mpi_type<unsigned char>      = MPI_UNSIGNED_CHAR;
template <> constexpr MPI_Datatype mpi_type<unsigned short>     = MPI_UNSIGNED_SHORT;
template <> constexpr MPI_Datatype mpi_type<unsigned int>       = MPI_UNSIGNED;
template <> constexpr MPI_Datatype mpi_type<unsigned long>      = MPI_UNSIGNED_LONG;
template <> constexpr MPI_Datatype mpi_type<unsigned long long> = MPI_UNSIGNED_LONG;
template <> constexpr MPI_Datatype mpi_type<float>              = MPI_FLOAT;
template <> constexpr MPI_Datatype mpi_type<double>             = MPI_DOUBLE;
template <> constexpr MPI_Datatype mpi_type<long double>        = MPI_LONG_DOUBLE;

/// Simple wrappers to check initialized and finalized.
bool mpi_initialized(
    sloc_t = sloc_t::current());                //!< debugging location

bool mpi_finalized(
    sloc_t = sloc_t::current());                //!< debugging location

/// Simple wrappers, will check initialized and finalized.
void mpi_init(
    sloc_t = sloc_t::current());                //!< debugging location

void mpi_fini(
    sloc_t = sloc_t::current());                //!< debugging location

/// Will `exit(EXIT_FAILURE)` if `!mpi_initialized()`.
[[noreturn]]
void mpi_abort(
    int e = -1,                                 //!< error code
    sloc_t = sloc_t::current());                //!< debugging location

/// Will translate `e` to a string and print an error.
void mpi_print_error(
    const char* symbol,                         //!< MPI_* as string
    int e,                                      //!< error code
    sloc_t = sloc_t::current());                //!< debugging location

/// Returns the count for a matching message.
int mpi_probe(
    int source,                                 //!< source rank
    int tag,                                    //!< user defined tag
    MPI_Datatype type,                          //!< type for count
    sloc_t = sloc_t::current());                //!< debugging location

/// Blocks until all of the requests are complete, ignores status.
void mpi_wait(
    std::span<MPI_Request> reqs,                //!< requests
    sloc_t = sloc_t::current());                //!< debugging location

/// If op(ts...)->error, prints and error and aborts.
template <class Op, class... Ts>
void mpi_check_op(sloc_t sloc, const char* symbol, Op&& op, Ts&&... ts) {
  if (int e = std::forward<Op>(op)(std::forward<Ts>(ts)...)) {
    mpi_print_error(symbol, e, sloc);
    mpi_abort(e, sloc);
  }
}

/// Helper macro for mpi_check_op.
#define mpi_check(sloc, op, ...) mpi_check_op(sloc, #op, (op), __VA_ARGS__)

static inline int mpi_rank(sloc_t sloc = sloc_t::current()) {
  int rank;
  mpi_check(sloc, MPI_Comm_rank, MPI_COMM_WORLD, &rank);
  return rank;
}

static inline int mpi_n_ranks(sloc_t sloc = sloc_t::current()) {
  int n_ranks;
  mpi_check(sloc, MPI_Comm_size, MPI_COMM_WORLD, &n_ranks);
  return n_ranks;
}

template <class T>
int mpi_probe(int source, int tag = 0, sloc_t sloc = sloc_t::current()) {
  return mpi_probe(source, tag, mpi_type<T>, sloc);
}

template <class T>
MPI_Request mpi_send(
    const T* from,
    int n,
    int to_rank,
    int tag = 0,
    sloc_t sloc = sloc_t::current())
{
  MPI_Request r;
  mpi_check(sloc, MPI_Isend, from, n, mpi_type<T>, to_rank, tag, MPI_COMM_WORLD, &r);
  return r;
}

template <class T>
MPI_Request mpi_recv(T* to, int n, int from_rank, int tag = 0, sloc_t sloc = sloc_t::current()) {
  MPI_Request r;
  mpi_check(sloc, MPI_Irecv, to, n, mpi_type<T>, from_rank, tag, MPI_COMM_WORLD, &r);
  return r;
}

template <class T>
MPI_Request mpi_allreduce(T* buffer, int n, MPI_Op op = MPI_SUM, sloc_t sloc = sloc_t::current()) {
  MPI_Request r;
  mpi_check(sloc, MPI_Iallreduce, MPI_IN_PLACE, buffer, n, mpi_type<T>, op, MPI_COMM_WORLD, &r);
  return r;
}

template <class T>
MPI_Request mpi_allreduce(std::vector<T>& v, MPI_Op op = MPI_SUM, sloc_t sloc = sloc_t::current()) {
  using std::ssize;
  return mpi_allreduce(data(v), ssize(v), op, sloc);
}
}

#endif // TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP
