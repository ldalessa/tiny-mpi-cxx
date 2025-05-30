#pragma once

#include <mpi.h>
#include <concepts>
#include <functional>
#include <ranges>
#include <source_location>
#include <span>
#include <vector>

#define TINY_MPI_FWD(x) static_cast<decltype(x)&&>(x)

namespace tiny_mpi
{
	using request_t = MPI_Request;
	using rank_t = int;
	using tag_t = int;

	enum thread_support_t : int {
		THREAD_SINGLE = MPI_THREAD_FUNNELED,
		THREAD_FUNNELED = MPI_THREAD_FUNNELED,
		THREAD_SERIALIZED = MPI_THREAD_SERIALIZED,
		THREAD_MULTIPLE = MPI_THREAD_MULTIPLE
	};

	template <class T>
	concept user_defined_type = requires {
		{ std::remove_cvref_t<T>::mpi_type() } -> std::same_as<MPI_Datatype>;
	};

	template <user_defined_type T>
	struct DeferredType {
		operator MPI_Datatype() const {
			return std::remove_cvref_t<T>::mpi_type();
		}
	};

	/// Simple variable template to map arithmatic types to their MPI
	/// equivalents.
	template <class T> inline constexpr auto type = std::false_type{};

	/// OpenMPI uses non-contsexpr void* for MPI_Datatypes so we can't constexpr
	/// them in that case. MPICH uses constants.
#ifdef OMPI_MPI_H
	template <> inline MPI_Datatype const type<std::byte>          = MPI_BYTE;
	template <> inline MPI_Datatype const type<char>               = MPI_CHAR;
	template <> inline MPI_Datatype const type<signed char>        = MPI_CHAR;
	template <> inline MPI_Datatype const type<short>              = MPI_SHORT;
	template <> inline MPI_Datatype const type<int>                = MPI_INT;
	template <> inline MPI_Datatype const type<long>               = MPI_LONG;
	template <> inline MPI_Datatype const type<long long>          = MPI_LONG_LONG_INT;
	template <> inline MPI_Datatype const type<unsigned char>      = MPI_UNSIGNED_CHAR;
	template <> inline MPI_Datatype const type<unsigned short>     = MPI_UNSIGNED_SHORT;
	template <> inline MPI_Datatype const type<unsigned int>       = MPI_UNSIGNED;
	template <> inline MPI_Datatype const type<unsigned long>      = MPI_UNSIGNED_LONG;
	template <> inline MPI_Datatype const type<unsigned long long> = MPI_UNSIGNED_LONG;
	template <> inline MPI_Datatype const type<float>              = MPI_FLOAT;
	template <> inline MPI_Datatype const type<double>             = MPI_DOUBLE;
	template <> inline MPI_Datatype const type<long double>        = MPI_LONG_DOUBLE;
#else
	template <> inline constexpr MPI_Datatype type<std::byte>          = MPI_BYTE;
	template <> inline constexpr MPI_Datatype type<char>               = MPI_CHAR;
	template <> inline constexpr MPI_Datatype type<signed char>        = MPI_CHAR;
	template <> inline constexpr MPI_Datatype type<short>              = MPI_SHORT;
	template <> inline constexpr MPI_Datatype type<int>                = MPI_INT;
	template <> inline constexpr MPI_Datatype type<long>               = MPI_LONG;
	template <> inline constexpr MPI_Datatype type<long long>          = MPI_LONG_LONG_INT;
	template <> inline constexpr MPI_Datatype type<unsigned char>      = MPI_UNSIGNED_CHAR;
	template <> inline constexpr MPI_Datatype type<unsigned short>     = MPI_UNSIGNED_SHORT;
	template <> inline constexpr MPI_Datatype type<unsigned int>       = MPI_UNSIGNED;
	template <> inline constexpr MPI_Datatype type<unsigned long>      = MPI_UNSIGNED_LONG;
	template <> inline constexpr MPI_Datatype type<unsigned long long> = MPI_UNSIGNED_LONG;
	template <> inline constexpr MPI_Datatype type<float>              = MPI_FLOAT;
	template <> inline constexpr MPI_Datatype type<double>             = MPI_DOUBLE;
	template <> inline constexpr MPI_Datatype type<long double>        = MPI_LONG_DOUBLE;
#endif
	
	template <class T> requires std::is_enum_v<T>
	inline MPI_Datatype const type<T> = type<std::underlying_type_t<T>>;

	template <user_defined_type T> DeferredType<T> const type<T>{};

	template <class T>
	concept trivially_copyable = std::is_trivially_copyable_v<T>;

	template <class T>
	concept mpi_typed = trivially_copyable<T>
		            and std::convertible_to<decltype(type<T>), MPI_Datatype>;

	struct min {
		constexpr auto operator()(auto a, auto b) {
			return std::min(a, b);
		}
	};

	struct max {
		constexpr auto operator()(auto a, auto b) {
			return std::min(a, b);
		}
	};

	/// Simple variable template to map the common reduction operators.
	template <class T> inline constexpr std::false_type op = {};

	/// Again, OpenMPI uses types that can't be constexpr here.
#ifdef OMPI_MPI_H
	template <> inline MPI_Op const op<min>                        = MPI_MIN;
	template <> inline MPI_Op const op<max>                        = MPI_MAX;
	template <class T> inline MPI_Op const op<std::plus<T>>        = MPI_SUM;
	template <class T> inline MPI_Op const op<std::multiplies<T>>  = MPI_PROD;
	template <class T> inline MPI_Op const op<std::bit_or<T>>      = MPI_BOR;
	template <class T> inline MPI_Op const op<std::bit_and<T>>     = MPI_BAND;
	template <class T> inline MPI_Op const op<std::bit_xor<T>>     = MPI_BXOR;
	template <class T> inline MPI_Op const op<std::logical_or<T>>  = MPI_LOR;
	template <class T> inline MPI_Op const op<std::logical_and<T>> = MPI_LAND;
#else
	template <> inline constexpr MPI_Op op<min>                        = MPI_MIN;
	template <> inline constexpr MPI_Op op<max>                        = MPI_MAX;
	template <class T> inline constexpr MPI_Op op<std::plus<T>>        = MPI_SUM;
	template <class T> inline constexpr MPI_Op op<std::multiplies<T>>  = MPI_PROD;
	template <class T> inline constexpr MPI_Op op<std::bit_or<T>>      = MPI_BOR;
	template <class T> inline constexpr MPI_Op op<std::bit_and<T>>     = MPI_BAND;
	template <class T> inline constexpr MPI_Op op<std::bit_xor<T>>     = MPI_BXOR;
	template <class T> inline constexpr MPI_Op op<std::logical_or<T>>  = MPI_LOR;
	template <class T> inline constexpr MPI_Op op<std::logical_and<T>> = MPI_LAND;
#endif
	
	template <class T>
	concept reduction_op = std::same_as<decltype(op<T>), const MPI_Op>;

	/// Simple wrappers to check initialized and finalized.
	bool initialized(
		std::source_location = std::source_location::current()); //!< debugging location

	bool finalized(
		std::source_location = std::source_location::current()); //!< debugging location

	/// Simple wrappers, will check initialized and finalized.
	auto init(
		thread_support_t = THREAD_SERIALIZED,
		std::source_location = std::source_location::current()) //!< debugging location
		-> thread_support_t;

	void fini(
		std::source_location = std::source_location::current()); //!< debugging location

	/// Will `exit(EXIT_FAILURE)` if `!initialized()`.
	[[noreturn]]
	void abort(
		int e = -1,								//!< error code
		std::source_location = std::source_location::current()); //!< debugging location

	/// Will translate `e` to a string and print an error.
	void print_error(
		const char* symbol,						//!< MPI_* as string
		int e,									//!< error code
		std::source_location = std::source_location::current()); //!< debugging location

	/// Returns the count for a matching message.
	[[nodiscard]]
	auto probe(
		rank_t source,								 //!< source rank
		MPI_Datatype type,							 //!< type for count
		tag_t tag = 0,								 //!< user defined tag
		std::source_location = std::source_location::current()) -> int; //!< debugging location

	/// Blocks until all of the requests are complete, ignores status.
	void wait(
		std::span<request_t> reqs,				//!< requests
		std::source_location = std::source_location::current()); //!< debugging location

	/// If f(ts...)->error, prints and error and aborts.
	inline constexpr struct
	{
		constexpr void operator()(
			std::source_location sloc,
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
#define tiny_mpi_check_op(op) #op, (op)

	[[nodiscard]]
	static inline auto rank(std::source_location sloc = std::source_location::current())
		-> rank_t
	{
		rank_t rank;
		check(sloc, tiny_mpi_check_op(MPI_Comm_rank), MPI_COMM_WORLD, &rank);
		return rank;
	}

	[[nodiscard]]
	static inline auto n_ranks(std::source_location sloc = std::source_location::current())
		-> rank_t
	{
		rank_t n_ranks;
		check(sloc, tiny_mpi_check_op(MPI_Comm_size), MPI_COMM_WORLD, &n_ranks);
		return n_ranks;
	}

	[[nodiscard]]
	static inline auto ranks(std::source_location sloc = std::source_location::current())
	{
		return std::ranges::iota_view{0, n_ranks(std::move(sloc))};
	}

	[[nodiscard]]
	static inline auto barrier(std::source_location sloc = std::source_location::current())
		-> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Ibarrier), MPI_COMM_WORLD, &r);
		return r;
	}

	void wait(request_t first, std::same_as<request_t> auto... rest) {
		request_t rs[] = { first, rest... };
		wait(rs);
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto probe(
		rank_t source,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> int
	{
		return probe(source, type<T>, tag, sloc);
	}

	template <trivially_copyable T>
	[[nodiscard]]
	auto probe(
		rank_t source,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> int
	{
		return probe(source, type<char>, tag, sloc) / sizeof(T);
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto send(
		const T* from,
		int n,
		rank_t to_rank,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Isend), from, n, type<T>, to_rank, tag, MPI_COMM_WORLD, &r);
		return r;
	}

	template <trivially_copyable T>
	[[nodiscard]]
	auto send(
		const T* from,
		int n,
		rank_t to_rank,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Isend), from, sizeof(T) * n, type<char>, to_rank, tag, MPI_COMM_WORLD, &r);
		return r;
	}

	[[nodiscard]]
	auto send(
		std::ranges::contiguous_range auto const& from,
		rank_t to_rank,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		return send(
			std::ranges::data(from),
			std::ranges::size(from),
			to_rank,
			tag,
			std::move(sloc));
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto recv(
		T* to,
		int n,
		rank_t from_rank,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Irecv), to, n, type<T>, from_rank, tag, MPI_COMM_WORLD, &r);
		return r;
	}

	template <trivially_copyable T>
	[[nodiscard]]
	auto recv(
		T* to,
		int n,
		rank_t from_rank,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Irecv), to, sizeof(T) * n, type<char>, from_rank, tag, MPI_COMM_WORLD, &r);
		return r;
	}

	[[nodiscard]]
	auto recv(
		std::ranges::contiguous_range auto& to,
		rank_t from_rank,
		tag_t tag = 0,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		return recv(
			std::ranges::data(to),
			std::ranges::size(to),
			from_rank,
			tag,
			std::move(sloc));
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto allreduce(
		T* buffer,
		int n,
		MPI_Op op = MPI_SUM,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Iallreduce), MPI_IN_PLACE, buffer, n, type<T>, op, MPI_COMM_WORLD, &r);
		return r;
	}


	template <mpi_typed T, reduction_op Op>
	[[nodiscard]]
	auto allreduce(
		T* v,
		int n,
		Op,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		return allreduce(v, n, op<Op>, sloc);
	}

	template <std::ranges::contiguous_range Range>
	[[nodiscard]]
	auto allreduce(
		Range& v,
		MPI_Op op = MPI_SUM,
		std::source_location sloc = std::source_location::current()) -> request_t
		requires mpi_typed<std::ranges::range_value_t<Range>>
	{
		return allreduce(
			std::ranges::data(v),
			std::ranges::size(v),
			op,
			sloc);
	}

	template <std::ranges::contiguous_range Range, reduction_op Op>
	[[nodiscard]]
	auto allreduce(
		Range& v,
		Op,
		std::source_location sloc = std::source_location::current()) -> request_t
		requires mpi_typed<std::ranges::range_value_t<Range>>
	{
		return allreduce(v, op<Op>, sloc);
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto allreduce(
		T& value,
		MPI_Op op = MPI_SUM,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(sloc, tiny_mpi_check_op(MPI_Iallreduce), MPI_IN_PLACE, std::addressof(value), 1, type<T>, op, MPI_COMM_WORLD, &r);
		return r;
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto allgather(
		T* values,
		int count,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(
			sloc,
			tiny_mpi_check_op(MPI_Iallgather),
			MPI_IN_PLACE,
			0,
			MPI_DATATYPE_NULL,
			values,
			count,
			type<T>,
			MPI_COMM_WORLD,
			&r);
		return r;
	}

	template <std::ranges::contiguous_range Range>
	[[nodiscard]]
	auto allgather(
		Range& values,
		std::source_location sloc = std::source_location::current()) -> request_t
		requires mpi_typed<std::ranges::range_value_t<Range>>
	{
		return allgather(std::ranges::data(values), 1, sloc);
	}

	template <mpi_typed T>
	[[nodiscard]]
	auto allgather(
		T* values,
		std::span<int const> counts,
		std::span<int const> offsets,
		std::source_location sloc = std::source_location::current()) -> request_t
	{
		request_t r;
		check(
			sloc,
			tiny_mpi_check_op(MPI_Iallgatherv),
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
		std::source_location sloc = std::source_location::current()) -> request_t
		requires mpi_typed<std::ranges::range_value_t<Range>>
	{
		return allgather(std::ranges::data(values), counts, offsets, sloc);
	}

	template <std::size_t N>
	struct async {
		request_t rs[N];

		constexpr async(std::same_as<request_t> auto... rs) : rs { rs... } {
		}

		constexpr ~async() {
			wait(rs);
		}
	};

	async(std::same_as<request_t> auto... rs) -> async<sizeof...(rs)>;


	/// An raii-scoped init.
	inline constexpr struct scoped_init_fn {
		struct _raii {
			bool _wait;

			~_raii() {
				if (_wait) {
					wait(barrier());
				}
				fini();
			}
		};

		[[nodiscard]]
		auto operator()(
			bool wait = true,
			thread_support_t threading = THREAD_SERIALIZED,
			std::source_location sloc = std::source_location::current()) const
			-> _raii
		{
			init(threading, sloc);
			return _raii { ._wait = wait };
		}
	} scoped_init{};
} // namespace tiny_mpi
