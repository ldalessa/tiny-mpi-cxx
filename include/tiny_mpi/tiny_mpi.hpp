#ifndef TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP
#define TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP

#include <version>

#ifdef __cpp_lib_source_location
#include <source_location>
#else
#include <experimental/source_location>
#endif

#include <ranges>
#include <span>
#include <vector>
#include <mpi.h>

#define TINY_MPI_FWD(x) static_cast<decltype(x)&&>(x)

namespace tiny_mpi
{

#ifdef __cpp_lib_source_location
    using sloc_t = std::source_location;
#else
    using sloc_t = std::experimental::source_location;
#endif

    using Request = MPI_Request;

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
    [[nodiscard]]
    auto probe(
        int source,                                  //!< source rank
        int tag,                                     //!< user defined tag
        MPI_Datatype type,                           //!< type for count
        sloc_t = sloc_t::current()) noexcept -> int; //!< debugging location

    /// Blocks until all of the requests are complete, ignores status.
    void wait(
        std::span<Request> reqs,                    //!< requests
        sloc_t = sloc_t::current()) noexcept;       //!< debugging location

    /// If f(ts...)->error, prints and error and aborts.
    inline constexpr struct
    {
        constexpr void operator()(
            sloc_t sloc,
            const char* symbol,
            auto&& f,
            auto&&... ts) const
        {
            if (int e = TINY_MPI_FWD(f)(TINY_MPI_FWD(ts)...)) {
                print_error(symbol, e, sloc);
                abort(e, sloc);
            }
        }
    } check;

    /// Helper macro for check
#define tiny_mpi_op(op) #op, (op)

    [[nodiscard]]
    static inline auto rank(sloc_t sloc = sloc_t::current())
        -> int
    {
        int rank;
        check(sloc, tiny_mpi_op(MPI_Comm_rank), MPI_COMM_WORLD, &rank);
        return rank;
    }

    [[nodiscard]]
    static inline auto n_ranks(sloc_t sloc = sloc_t::current())
        -> int
    {
        int n_ranks;
        check(sloc, tiny_mpi_op(MPI_Comm_size), MPI_COMM_WORLD, &n_ranks);
        return n_ranks;
    }

    [[nodiscard]]
    static inline auto barrier(sloc_t sloc = sloc_t::current())
        -> Request
    {
        Request r;
        check(sloc, tiny_mpi_op(MPI_Ibarrier), MPI_COMM_WORLD, &r);
        return r;
    }

    void wait(std::same_as<Request> auto... requests) {
        Request rs[] = { requests... };
        wait(rs);
    }

    template <class T>
    [[nodiscard]]
    auto probe(
        int source,
        int tag = 0,
        sloc_t sloc = sloc_t::current()) -> int
    {
        return probe(source, tag, type<T>, sloc);
    }

    template <class T>
    [[nodiscard]]
    auto send(
        const T* from,
        int n,
        int to_rank,
        int tag = 0,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        Request r;
        check(sloc, tiny_mpi_op(MPI_Isend), from, n, type<T>, to_rank, tag, MPI_COMM_WORLD, &r);
        return r;
    }

    [[nodiscard]]
    auto send(
        std::ranges::contiguous_range auto const& from,
        int to_rank,
        int tag = 0,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        return send(
            std::ranges::begin(from),
            std::ranges::size(from),
            to_rank,
            tag,
            std::move(sloc));
    }

    template <class T>
    [[nodiscard]]
    auto recv(
        T* to,
        int n,
        int from_rank,
        int tag = 0,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        Request r;
        check(sloc, tiny_mpi_op(MPI_Irecv), to, n, type<T>, from_rank, tag, MPI_COMM_WORLD, &r);
        return r;
    }

    [[nodiscard]]
    auto recv(
        std::ranges::contiguous_range auto& to,
        int from_rank,
        int tag = 0,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        return recv(
            std::ranges::begin(to),
            std::ranges::size(to),
            from_rank,
            tag,
            std::move(sloc));
    }

    template <class T>
    [[nodiscard]]
    auto allreduce(
        T* buffer,
        int n,
        MPI_Op op = MPI_SUM,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        Request r;
        check(sloc, tiny_mpi_op(MPI_Iallreduce), MPI_IN_PLACE, buffer, n, type<T>, op, MPI_COMM_WORLD, &r);
        return r;
    }

    template <class T>
    [[nodiscard]]
    auto allreduce(
        std::vector<T>& v,
        MPI_Op op = MPI_SUM,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        return allreduce(data(v), ssize(v), op, sloc);
    }

    template <class T>
    [[nodiscard]]
    auto allgather(
        T* values,
        std::span<int const> counts,
        std::span<int const> offsets,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        Request r;
        check(
            sloc,
            tiny_mpi_op(MPI_Iallgatherv),
            MPI_IN_PLACE,
            0,
            MPI_DATATYPE_NULL,
            values,
            data(counts),
            data(offsets),
            type<T>,
            MPI_COMM_WORLD,
            &r);
        return r;
    }

    template <std::ranges::contiguous_range Range>
    [[nodiscard]]
    auto allgather(
        Range& values,
        std::span<int const> counts,
        std::span<int const> offsets,
        sloc_t sloc = sloc_t::current()) -> Request
    {
        return allgather(std::ranges::data(values), counts, offsets, sloc);
    }

    template <std::size_t N>
    struct async {
        Request rs[N];

        constexpr async(std::same_as<Request> auto... rs) noexcept : rs { rs... } {
        }

        constexpr ~async() {
            wait(rs);
        }
    };

    async(std::same_as<Request> auto... rs) -> async<sizeof...(rs)>;
} // namespace tiny_mpi

#endif // TINY_MPI_CXX_INCLUDE_TINY_MPI_TINY_MPI_HPP
