#ifndef __MORLOC__TENSOR_HPP__
#define __MORLOC__TENSOR_HPP__

// C++ implementations of the morloc tensor stdlib.
// Built on mlc::Tensor (vector + mdspan). Uses .view() (mdspan operator())
// for multi-dim access.

#include "mlc_tensor.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

// ---------------------------------------------------------------------------
// zeros / ones / identity: T is determined by the *result* type, not by any
// argument, so plain function templates would fail deduction. Each function
// returns a small proxy that holds the construction parameters and exposes
// templated conversion operators; T is resolved at the call site from the
// LHS type. The compiler emits e.g. `std::vector<double> v = morloc_zeros1(3)`
// or `mlc::Tensor2<float> m = morloc_identity(4)`, and the matching conversion
// operator instantiates the concrete value with the correct element type.
//
// fill takes the value as its first argument, so T is deducible there and no
// proxy is needed.
// ---------------------------------------------------------------------------

namespace mlc_ctor {

template <std::size_t N>
struct Zeros {
    std::array<int64_t, N> dims;

    template <class T>
    operator std::vector<T>() const {
        static_assert(N == 1, "rank > 1 zeros does not convert to std::vector");
        return std::vector<T>(static_cast<size_t>(dims[0]), T(0));
    }

    template <class T>
    operator mlc::Tensor<T, N>() const {
        return mlc::Tensor<T, N>(dims);
    }
};

template <std::size_t N>
struct Ones {
    std::array<int64_t, N> dims;

    template <class T>
    operator std::vector<T>() const {
        static_assert(N == 1, "rank > 1 ones does not convert to std::vector");
        return std::vector<T>(static_cast<size_t>(dims[0]), T(1));
    }

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

inline mlc_ctor::Zeros<1> morloc_zeros1(int64_t d1) {
    return mlc_ctor::Zeros<1>{{d1}};
}
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

inline mlc_ctor::Ones<1> morloc_ones1(int64_t d1) {
    return mlc_ctor::Ones<1>{{d1}};
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
// fill1 returns std::vector<T> (Vector maps to std::vector); higher ranks
// return mlc::TensorN<T>.
// ---------------------------------------------------------------------------

template <class T>
std::vector<T> morloc_fill1(const T& v, int64_t d1) {
    return std::vector<T>(static_cast<size_t>(d1), v);
}

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
// Packable: tuple-of-(dims, flat-vector) <-> mlc::Tensor.
//
// pack: take a (dims-tuple, flat std::vector) and construct an mlc::Tensor
//   that takes ownership of the vector. No data copy: the existing storage
//   is moved into the tensor.
// unpack: take an mlc::Tensor and yield (dims-tuple, flat std::vector). The
//   tensor's storage is moved out, also no data copy.
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
    std::vector<T> from_storage(std::vector<S>&& data) {
        if constexpr (std::is_same_v<T, S>) {
            return std::move(data);
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

template <class T>
std::tuple<std::tuple<int, int>, std::vector<T>>
morloc_unpackMatrix(mlc::Tensor2<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    return std::make_tuple(
        std::make_tuple((int)sh[0], (int)sh[1]),
        mlc_pack_detail::from_storage<T, S>(std::move(t.storage())));
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
std::tuple<std::tuple<int, int, int>, std::vector<T>>
morloc_unpackTensor3(mlc::Tensor3<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    return std::make_tuple(
        std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2]),
        mlc_pack_detail::from_storage<T, S>(std::move(t.storage())));
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
std::tuple<std::tuple<int, int, int, int>, std::vector<T>>
morloc_unpackTensor4(mlc::Tensor4<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    return std::make_tuple(
        std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3]),
        mlc_pack_detail::from_storage<T, S>(std::move(t.storage())));
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
std::tuple<std::tuple<int, int, int, int, int>, std::vector<T>>
morloc_unpackTensor5(mlc::Tensor5<T>& t)
{
    using S = mlc::tensor_storage_t<T>;
    auto sh = t.shape();
    return std::make_tuple(
        std::make_tuple((int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3], (int)sh[4]),
        mlc_pack_detail::from_storage<T, S>(std::move(t.storage())));
}

#endif
