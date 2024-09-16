#pragma once

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <functional>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <format>
#include <optional>
#include <cmath>

#include <Eigen/Dense>


namespace deepl {
	typedef std::ptrdiff_t index;

	auto dyn = Eigen::Dynamic;

	template<typename T>
	concept Int_t = std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value;

	template<typename T>
	concept EFloat = std::is_same<T, float>::value || std::is_same<T, double>::value;

	template<typename T>
	concept Real = EFloat<T> || Int_t<T>;

	template<Int_t T> 
	using IntMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

	template<Real T> 
	using RealMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

	template<Real T>
	class Tensor {
	public:
		Tensor() = delete;
		Tensor(int dim1, int dim2, const T init);
		Tensor(RealMatrix<T>&& values) noexcept;
		void Print() const;
		std::vector<index> Shape() const;
		RealMatrix<T>& Values();
		std::optional<std::function<Tensor<T>()>>& GradL();
		std::optional<std::function<Tensor<T>()>>& GradR();

		static Tensor<T> Add(Tensor<T>& a, Tensor<T>& b);
		static Tensor<T> Mul(Tensor<T>& a, Tensor<T>& b);
		static Tensor<T> Dot(Tensor<T>& a, Tensor<T>& b);

		static Tensor<T> ReLU(Tensor<T>& a);

		static Tensor<double> Cross_Entropy(Tensor<T>& a, const size_t& target);
		//static Tensor<double> Cross_Entropy(Tensor<T>& a, const vector<size_t>& targets);

		//IntMatrix<T> Sub(const IntMatrix<T>& a, const IntMatrix<T>& b);
		//IntMatrix<T> Div(const IntMatrix<T>& a, const IntMatrix<T>& b);
		//IntMatrix<T> Pow(const IntMatrix<T>& a, const IntMatrix<T>& b);
	private:
		RealMatrix<T> mValues;
		RealMatrix<T> mGrad1;
		RealMatrix<T> mGrad2;
		std::optional<std::function<Tensor<T>()>> mGradR;
		std::optional<std::function<Tensor<T>()>> mGradL;
	};
}

#include "Tensor.cpp"

#endif
