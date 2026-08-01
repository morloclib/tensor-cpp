#ifndef __MORLOC__TENSOR_HPP__
#define __MORLOC__TENSOR_HPP__

// C++ implementations of the morloc tensor stdlib (ranks 2-5).
// Defines the mlc::Tensor owning type (vector + mdspan) shared by all
// ranks; .view() (mdspan operator()) provides multi-dim access. Rank-1
// (Vector) lives in vector-cpp/vector.hpp.

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

// std::mdspan is a C++23 header. Until libstdc++ 15 is widely deployed we
// use the kokkos/mdspan reference implementation (vendored as mdspan.hpp),
// which configures itself to inject mdspan / dextents / extents / layout_*
// into namespace std. Once <mdspan> ships natively, drop the include and
// the user-facing names below remain unchanged.
#if __cpp_lib_mdspan >= 202207L
#  include <mdspan>
#else
#  include "mdspan.hpp"
#endif

// ---------------------------------------------------------------------------
// mlc::Tensor -- thin owning tensor built on std::vector + std::mdspan.
// Ownership: std::vector<S> (contiguous row-major storage).
// View: std::mdspan via .view() for multidimensional access.
// ---------------------------------------------------------------------------

namespace mlc {

// Storage type trait: maps bool to uint8_t so tensor memory layout
// matches the voidstar format (1 byte) regardless of sizeof(bool).
template<typename T> struct tensor_storage { using type = T; };
template<> struct tensor_storage<bool> { using type = uint8_t; };
template<typename T> using tensor_storage_t = typename tensor_storage<T>::type;

template<typename T, int NDim>
class Tensor {
    using S = tensor_storage_t<T>;
public:
    // Construct with given shape
    Tensor(const std::array<int64_t, NDim>& dims) : shape_(dims) {
        size_t n = 1;
        for (int i = 0; i < NDim; i++) n *= (size_t)shape_[i];
        data_.resize(n);
    }

    // Construct from flat data + shape
    Tensor(std::vector<S> data, const std::array<int64_t, NDim>& dims)
        : data_(std::move(data)), shape_(dims) {}

    // Construct from flat data + shape pointer (for deserialization)
    Tensor(const S* src, const int64_t* shape, size_t total)
        : data_(src, src + total) {
        for (int i = 0; i < NDim; i++) shape_[i] = shape[i];
    }

    Tensor() = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // mdspan view (row-major / C order)
    auto view() {
        return std::mdspan<S, std::dextents<int64_t, NDim>>(data_.data(), shape_);
    }
    auto view() const {
        return std::mdspan<const S, std::dextents<int64_t, NDim>>(data_.data(), shape_);
    }

    // Raw access
    S* data() { return data_.data(); }
    const S* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    const std::array<int64_t, NDim>& shape() const { return shape_; }
    int64_t shape(int d) const { return shape_[d]; }
    constexpr int ndim() const { return NDim; }

    // Linear access
    S& operator[](size_t i) { return data_[i]; }
    const S& operator[](size_t i) const { return data_[i]; }

    // Move out the backing vector (for unpack / serialization)
    std::vector<S>& storage() { return data_; }
    const std::vector<S>& storage() const { return data_; }

private:
    std::vector<S> data_;
    std::array<int64_t, NDim> shape_{};
};

// Convenience aliases
template<typename T> using Tensor1 = Tensor<T, 1>;
template<typename T> using Tensor2 = Tensor<T, 2>;
template<typename T> using Tensor3 = Tensor<T, 3>;
template<typename T> using Tensor4 = Tensor<T, 4>;
template<typename T> using Tensor5 = Tensor<T, 5>;

// Type traits
template<typename T> struct is_mlc_tensor : std::false_type {};
template<typename T, int N> struct is_mlc_tensor<Tensor<T, N>> : std::true_type {};
template<typename T> inline constexpr bool is_mlc_tensor_v = is_mlc_tensor<T>::value;

template<typename T> struct tensor_element;
template<typename T, int N> struct tensor_element<Tensor<T, N>> { using type = T; };
template<typename T> using tensor_element_t = typename tensor_element<T>::type;

template<typename T> struct tensor_ndim;
template<typename T, int N> struct tensor_ndim<Tensor<T, N>>
    { static constexpr int value = N; };
template<typename T> inline constexpr int tensor_ndim_v = tensor_ndim<T>::value;

} // namespace mlc


// ---------------------------------------------------------------------------
// zeros / ones / identity: T is determined by the *result* type, not by any
// argument, so plain function templates would fail deduction. Each function
// returns a small proxy that holds the construction parameters and exposes
// templated conversion operators; T is resolved at the call site from the
// LHS type. The compiler emits e.g. `mlc::Tensor2<float> m = morloc_identity(4)`,
// and the matching conversion operator instantiates the concrete value with
// the correct element type.
//
// fill takes the value as its first argument, so T is deducible there and no
// proxy is needed.
// ---------------------------------------------------------------------------

namespace mlc_ctor {

template <std::size_t N>
struct Zeros {
    std::array<int64_t, N> dims;

    template <class T>
    operator mlc::Tensor<T, N>() const {
        return mlc::Tensor<T, N>(dims);
    }
};

template <std::size_t N>
struct Ones {
    std::array<int64_t, N> dims;

    template <class T>
    operator mlc::Tensor<T, N>() const {
        auto t = mlc::Tensor<T, N>(dims);
        std::fill(t.data(), t.data() + t.size(), T(1));
        return t;
    }
};

struct Identity {
    int64_t n;

    template <class T>
    operator mlc::Tensor<T, 2>() const {
        auto t = mlc::Tensor<T, 2>(std::array<int64_t, 2>{n, n});
        auto v = t.view();
        for (int64_t i = 0; i < n; i++) v(i, i) = T(1);
        return t;
    }
};

}  // namespace mlc_ctor

inline mlc_ctor::Zeros<2> morloc_zeros2(int64_t d1, int64_t d2) {
    return mlc_ctor::Zeros<2>{{d1, d2}};
}
inline mlc_ctor::Zeros<3> morloc_zeros3(int64_t d1, int64_t d2, int64_t d3) {
    return mlc_ctor::Zeros<3>{{d1, d2, d3}};
}
inline mlc_ctor::Zeros<4> morloc_zeros4(int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return mlc_ctor::Zeros<4>{{d1, d2, d3, d4}};
}
inline mlc_ctor::Zeros<5> morloc_zeros5(int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    return mlc_ctor::Zeros<5>{{d1, d2, d3, d4, d5}};
}

inline mlc_ctor::Ones<2> morloc_ones2(int64_t d1, int64_t d2) {
    return mlc_ctor::Ones<2>{{d1, d2}};
}
inline mlc_ctor::Ones<3> morloc_ones3(int64_t d1, int64_t d2, int64_t d3) {
    return mlc_ctor::Ones<3>{{d1, d2, d3}};
}
inline mlc_ctor::Ones<4> morloc_ones4(int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return mlc_ctor::Ones<4>{{d1, d2, d3, d4}};
}
inline mlc_ctor::Ones<5> morloc_ones5(int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    return mlc_ctor::Ones<5>{{d1, d2, d3, d4, d5}};
}

inline mlc_ctor::Identity morloc_identity(int64_t n) {
    return mlc_ctor::Identity{n};
}

// ---------------------------------------------------------------------------
// fill: T is deducible from the value argument, so a plain template suffices.
// Returns mlc::TensorN<T> for N in [2,5].
// ---------------------------------------------------------------------------

template <class T>
mlc::Tensor2<T> morloc_fill2(const T& v, int64_t d1, int64_t d2) {
    auto t = mlc::Tensor2<T>(std::array<int64_t, 2>{d1, d2});
    std::fill(t.data(), t.data() + t.size(), v);
    return t;
}

template <class T>
mlc::Tensor3<T> morloc_fill3(const T& v, int64_t d1, int64_t d2, int64_t d3) {
    auto t = mlc::Tensor3<T>(std::array<int64_t, 3>{d1, d2, d3});
    std::fill(t.data(), t.data() + t.size(), v);
    return t;
}

template <class T>
mlc::Tensor4<T> morloc_fill4(const T& v, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    auto t = mlc::Tensor4<T>(std::array<int64_t, 4>{d1, d2, d3, d4});
    std::fill(t.data(), t.data() + t.size(), v);
    return t;
}

template <class T>
mlc::Tensor5<T> morloc_fill5(const T& v, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    auto t = mlc::Tensor5<T>(std::array<int64_t, 5>{d1, d2, d3, d4, d5});
    std::fill(t.data(), t.data() + t.size(), v);
    return t;
}

// ---------------------------------------------------------------------------
// matmul: naive triple loop. K-loop in middle for cache-friendly access on the
// inner element of B. BLAS dispatch is a future optimisation.
// ---------------------------------------------------------------------------

template <class T>
mlc::Tensor2<T> morloc_matmul(const mlc::Tensor2<T>& A, const mlc::Tensor2<T>& B) {
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    mlc::Tensor2<T> C(std::array<int64_t, 2>{M, N});
    auto av = A.view();
    auto bv = B.view();
    auto cv = C.view();
    for (int64_t i = 0; i < M; i++) {
        for (int64_t k = 0; k < K; k++) {
            T aik = av(i, k);
            for (int64_t j = 0; j < N; j++) {
                cv(i, j) += aik * bv(k, j);
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// Eq: structural equality on shape and storage. mlc::Tensor itself defines
// no operator==, so the generic morloc_eq template in root-cpp would fail
// to compile on tensors. Each rank routes through a specialized comparator.
// ---------------------------------------------------------------------------

template <class T, int N>
bool morloc_tensor_eq(const mlc::Tensor<T, N>& a, const mlc::Tensor<T, N>& b) {
    return a.shape() == b.shape() && a.storage() == b.storage();
}

// ---------------------------------------------------------------------------
// Packable: tuple-of-(dims, flat-vector) <-> mlc::Tensor.
//
// pack: take a (dims-tuple, flat std::vector) and construct an mlc::Tensor
//   that takes ownership of the vector. No data copy: the existing storage
//   is moved into the tensor.
// unpack: take an mlc::Tensor by const reference and yield
//   (dims-tuple, flat std::vector). Storage is copied -- the morloc pool
//   binds unpack arguments by const reference, so moving out is unavailable.
//
// The morloc compiler routes serialization through these functions for any
// cross-language tensor passing.
// ---------------------------------------------------------------------------

// Wire dim type: matches morloc's `Int` -> C++ `int` mapping in root-cpp.
// Internally we cast to int64_t for mlc::Tensor's shape array.
//
// Storage type: mlc::Tensor uses tensor_storage_t<T> as its element type
// (e.g., uint8_t for bool to avoid std::vector<bool>'s proxy semantics). The
// wire `std::vector<T>` may differ; pack/unpack convert when so.
//
// Vector itself has no pack/unpack: morloc Vector maps directly to
// std::vector in C++, so the wire form is the runtime form and no
// conversion is required.
namespace mlc_pack_detail {
    template <class T, class S>
    std::vector<S> to_storage(std::vector<T>&& data) {
        if constexpr (std::is_same_v<T, S>) {
            return std::move(data);
        } else {
            std::vector<S> out(data.size());
            for (size_t i = 0; i < data.size(); i++) out[i] = static_cast<S>(data[i]);
            return out;
        }
    }
    template <class T, class S>
    std::vector<T> from_storage(const std::vector<S>& data) {
        if constexpr (std::is_same_v<T, S>) {
            return data;
        } else {
            std::vector<T> out(data.size());
            for (size_t i = 0; i < data.size(); i++) out[i] = static_cast<T>(data[i]);
            return out;
        }
    }
}

template <class T>
mlc::Tensor2<T> morloc_packMatrix(
    std::tuple<std::tuple<int, int>, std::vector<T>> packed)
{
    using S = mlc::tensor_storage_t<T>;
    auto& dims = std::get<0>(packed);
    auto& data = std::get<1>(packed);
    int64_t d1 = std::get<0>(dims);
    int64_t d2 = std::get<1>(dims);
    if (data.size() != static_cast<size_t>(d1) * static_cast<size_t>(d2)) {
        throw std::runtime_error("packMatrix: data length does not match dims");
    }
    auto storage = mlc_pack_detail::to_storage<T, S>(std::move(data));
    return mlc::Tensor2<T>(std::move(storage), std::array<int64_t, 2>{d1, d2});
}

// unpack borrows the tensor's storage when the wire element type matches the
// storage element type (the common case): the unpacked wire tuple is consumed
// within the same full expression that calls unpack (`_put_value(unpack(t),
// schema)`), so a reference into `t`'s storage stays valid for its only use and
// no copy is made. When the types differ (e.g. bool vs uint8_t) a conversion is
// unavoidable, so an owning vector is returned instead.
template <class T>
auto morloc_unpackMatrix(const mlc::Tensor2<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    if constexpr (std::is_same_v<T, S>) {
        return std::tuple<std::tuple<int, int>, const std::vector<S>&>(
            std::make_tuple((int)sh[0], (int)sh[1]), t.storage());
    } else {
        return std::make_tuple(
            std::make_tuple((int)sh[0], (int)sh[1]),
            mlc_pack_detail::from_storage<T, S>(t.storage()));
    }
}

template <class T>
mlc::Tensor3<T> morloc_packTensor3(
    std::tuple<std::tuple<int, int, int>, std::vector<T>> packed)
{
    using S = mlc::tensor_storage_t<T>;
    auto& dims = std::get<0>(packed);
    auto& data = std::get<1>(packed);
    int64_t d1 = std::get<0>(dims);
    int64_t d2 = std::get<1>(dims);
    int64_t d3 = std::get<2>(dims);
    size_t n = static_cast<size_t>(d1) * static_cast<size_t>(d2) * static_cast<size_t>(d3);
    if (data.size() != n) {
        throw std::runtime_error("packTensor3: data length does not match dims");
    }
    auto storage = mlc_pack_detail::to_storage<T, S>(std::move(data));
    return mlc::Tensor3<T>(std::move(storage), std::array<int64_t, 3>{d1, d2, d3});
}

template <class T>
auto morloc_unpackTensor3(const mlc::Tensor3<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    if constexpr (std::is_same_v<T, S>) {
        return std::tuple<std::tuple<int, int, int>, const std::vector<S>&>(
            std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2]), t.storage());
    } else {
        return std::make_tuple(
            std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2]),
            mlc_pack_detail::from_storage<T, S>(t.storage()));
    }
}

template <class T>
mlc::Tensor4<T> morloc_packTensor4(
    std::tuple<std::tuple<int, int, int, int>, std::vector<T>> packed)
{
    using S = mlc::tensor_storage_t<T>;
    auto& dims = std::get<0>(packed);
    auto& data = std::get<1>(packed);
    int64_t d1 = std::get<0>(dims);
    int64_t d2 = std::get<1>(dims);
    int64_t d3 = std::get<2>(dims);
    int64_t d4 = std::get<3>(dims);
    size_t n = static_cast<size_t>(d1) * static_cast<size_t>(d2)
             * static_cast<size_t>(d3) * static_cast<size_t>(d4);
    if (data.size() != n) {
        throw std::runtime_error("packTensor4: data length does not match dims");
    }
    auto storage = mlc_pack_detail::to_storage<T, S>(std::move(data));
    return mlc::Tensor4<T>(std::move(storage), std::array<int64_t, 4>{d1, d2, d3, d4});
}

template <class T>
auto morloc_unpackTensor4(const mlc::Tensor4<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    if constexpr (std::is_same_v<T, S>) {
        return std::tuple<std::tuple<int, int, int, int>, const std::vector<S>&>(
            std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3]), t.storage());
    } else {
        return std::make_tuple(
            std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3]),
            mlc_pack_detail::from_storage<T, S>(t.storage()));
    }
}

template <class T>
mlc::Tensor5<T> morloc_packTensor5(
    std::tuple<std::tuple<int, int, int, int, int>, std::vector<T>> packed)
{
    using S = mlc::tensor_storage_t<T>;
    auto& dims = std::get<0>(packed);
    auto& data = std::get<1>(packed);
    int64_t d1 = std::get<0>(dims);
    int64_t d2 = std::get<1>(dims);
    int64_t d3 = std::get<2>(dims);
    int64_t d4 = std::get<3>(dims);
    int64_t d5 = std::get<4>(dims);
    size_t n = static_cast<size_t>(d1) * static_cast<size_t>(d2)
             * static_cast<size_t>(d3) * static_cast<size_t>(d4) * static_cast<size_t>(d5);
    if (data.size() != n) {
        throw std::runtime_error("packTensor5: data length does not match dims");
    }
    auto storage = mlc_pack_detail::to_storage<T, S>(std::move(data));
    return mlc::Tensor5<T>(std::move(storage), std::array<int64_t, 5>{d1, d2, d3, d4, d5});
}

template <class T>
auto morloc_unpackTensor5(const mlc::Tensor5<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    if constexpr (std::is_same_v<T, S>) {
        return std::tuple<std::tuple<int, int, int, int, int>, const std::vector<S>&>(
            std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3], (int)sh[4]), t.storage());
    } else {
        return std::make_tuple(
            std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3], (int)sh[4]),
            mlc_pack_detail::from_storage<T, S>(t.storage()));
    }
}

#endif
