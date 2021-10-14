#ifndef TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP
#define TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP

#include <version>

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
template <class T> constexpr MPI_Datatype type = not defined(type<T>);
template <> constexpr MPI_Datatype type<std::byte>          = MPI_BYTE;
template <> constexpr MPI_Datatype type<char>               = MPI_CHAR;
template <> constexpr MPI_Datatype type<signed char>        = MPI_CHAR;
template <> constexpr MPI_Datatype type<short>              = MPI_SHORT;
template <> constexpr MPI_Datatype type<int>                = MPI_INT;
template <> constexpr MPI_Datatype type<long>               = MPI_LONG;
template <> constexpr MPI_Datatype type<long long>          = MPI_LONG_LONG_INT;
template <> constexpr MPI_Datatype type<unsigned char>      = MPI_UNSIGNED_CHAR;
template <> constexpr MPI_Datatype type<unsigned short>     = MPI_UNSIGNED_SHORT;
template <> constexpr MPI_Datatype type<unsigned int>       = MPI_UNSIGNED;
template <> constexpr MPI_Datatype type<unsigned long>      = MPI_UNSIGNED_LONG;
template <> constexpr MPI_Datatype type<unsigned long long> = MPI_UNSIGNED_LONG;
template <> constexpr MPI_Datatype type<float>              = MPI_FLOAT;
template <> constexpr MPI_Datatype type<double>             = MPI_DOUBLE;
template <> constexpr MPI_Datatype type<long double>        = MPI_LONG_DOUBLE;

/// Simple wrappers to check initialized and finalized.
bool initialized(
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

bool finalized(
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

/// Simple wrappers, will check initialized and finalized.
void init(
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

void fini(
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

/// Will `exit(EXIT_FAILURE)` if `!initialized()`.
[[noreturn]]
void abort(
    int e = -1,                                 //!< error code
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

/// Will translate `e` to a string and print an error.
void print_error(
    const char* symbol,                         //!< MPI_* as string
    int e,                                      //!< error code
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

/// Returns the count for a matching message.
int probe(
    int source,                                 //!< source rank
    int tag,                                    //!< user defined tag
    MPI_Datatype type,                          //!< type for count
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

/// Blocks until all of the requests are complete, ignores status.
void wait(
    std::span<MPI_Request> reqs,                //!< requests
    sloc_t = sloc_t::current()) noexcept;       //!< debugging location

/// If op(ts...)->error, prints and error and aborts.
template <class Op, class... Ts>
void check_op(sloc_t sloc, const char* symbol, Op&& op, Ts&&... ts) {
  if (int e = std::forward<Op>(op)(std::forward<Ts>(ts)...)) {
    print_error(symbol, e, sloc);
    abort(e, sloc);
  }
}

/// Helper macro for check_op.
#define check(sloc, op, ...) check_op(sloc, #op, (op), __VA_ARGS__)

static inline int rank(sloc_t sloc = sloc_t::current()) {
  int rank;
  check(sloc, MPI_Comm_rank, MPI_COMM_WORLD, &rank);
  return rank;
}

static inline int n_ranks(sloc_t sloc = sloc_t::current()) {
  int n_ranks;
  check(sloc, MPI_Comm_size, MPI_COMM_WORLD, &n_ranks);
  return n_ranks;
}

void wait(std::same_as<MPI_Request> auto... requests) {
  MPI_Request rs[] = { requests... };
  wait(rs);
}

template <class T>
int probe(int source, int tag = 0, sloc_t sloc = sloc_t::current()) {
  return probe(source, tag, type<T>, sloc);
}

template <class T>
MPI_Request send(
    const T* from,
    int n,
    int to_rank,
    int tag = 0,
    sloc_t sloc = sloc_t::current())
{
  MPI_Request r;
  check(sloc, MPI_Isend, from, n, type<T>, to_rank, tag, MPI_COMM_WORLD, &r);
  return r;
}

template <class T>
MPI_Request recv(T* to, int n, int from_rank, int tag = 0, sloc_t sloc = sloc_t::current()) {
  MPI_Request r;
  check(sloc, MPI_Irecv, to, n, type<T>, from_rank, tag, MPI_COMM_WORLD, &r);
  return r;
}

template <class T>
MPI_Request allreduce(T* buffer, int n, MPI_Op op = MPI_SUM, sloc_t sloc = sloc_t::current()) {
  MPI_Request r;
  check(sloc, MPI_Iallreduce, MPI_IN_PLACE, buffer, n, type<T>, op, MPI_COMM_WORLD, &r);
  return r;
}

template <class T>
MPI_Request allreduce(std::vector<T>& v, MPI_Op op = MPI_SUM, sloc_t sloc = sloc_t::current()) {
  using std::ssize;
  return allreduce(data(v), ssize(v), op, sloc);
}

template <std::size_t N>
struct async {
  MPI_Request rs[N];

  constexpr async(std::same_as<MPI_Request> auto... rs) noexcept : rs { rs... } {
  }

  constexpr ~async() {
    wait(rs);
  }
};

async(std::same_as<MPI_Request> auto... rs) -> async<sizeof...(rs)>;
} // namespace tiny_mpi

#endif // TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP
