#include "Tensor.h"

namespace deepl {

	template<Real T>
	Tensor<T>::Tensor(int dim1, int dim2, const T init):
		mValues(Eigen::Matrix<T, -1, -1>::Constant(dim1, dim2, init)) {}
	
	template<Real T>
	Tensor<T>::Tensor(RealMatrix<T>&& values) noexcept:
		mValues(std::move(values)) {}

	template<Real T>
	void Tensor<T>::Print() const 
	{
		std::cout << mValues << std::endl;
	}

	template<Real T>
	std::vector<index> Tensor<T>::Shape() const
	{
		return std::vector<index>{mValues.rows(), mValues.cols()};
	}

	template<Real T>
	RealMatrix<T>& Tensor<T>::Values()
	{
		return mValues;
	}

	template<Real T>
	std::optional<std::function<Tensor<T>()>>& Tensor<T>::GradL() 
	{
		return mGradL;
	}

	template<Real T>
	std::optional<std::function<Tensor<T>()>>& Tensor<T>::GradR()
	{
		return mGradR;
	}

	template<Real T>
	Tensor<T> Tensor<T>::Add(Tensor<T>& a, Tensor<T>& b)
	{
		auto shapeA = a.Shape();
		auto shapeB = b.Shape();
		if (shapeA != shapeB) {
			std::string errMsg = std::format("ERR: in Add(a, b) a and b don't have the same shape. ({}, {}) != ({}, {})\n",
				shapeA[0], shapeA[1], shapeB[0], shapeB[1]);
			std::cout << errMsg;
		}
		const RealMatrix<T>& aValues = a.Values();
		const RealMatrix<T>& bValues = b.Values();
		RealMatrix<T> cValues = aValues + bValues;
		//std::cout << "a:" << std::endl << aValues << std::endl << "b:\n" << bValues << std::endl << "a+b:\n" << cValues << "\n" << std::endl;
		auto agt0 = (aValues.array() > 0); 
		auto bgt0 = (bValues.array() > 0);
		auto cgt0 = (cValues.array() > 0);
		if (!(((agt0 && bgt0 == cgt0) || (agt0 ^ bgt0)).all())) {
			std::string errMsg = "WARNING: Integer Overflow or Underflow occured while Add(a, b)\n";
			std::cout << errMsg;
		}
		Tensor<T> c(std::move(cValues));
		c.GradL() = [shapeA = std::move(shapeA)]() { return Tensor<T>(static_cast<int>(shapeA[0]), static_cast<int>(shapeA[1]), T(1)); }; // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Ones(shapeA[0], shapeA[1]);};
		c.GradR() = c.GradL();
		return c;
	}

	template<Real T>
	Tensor<T> Tensor<T>::Mul(Tensor<T>& a, Tensor<T>& b) {
		auto shapeA = a.Shape();
		auto shapeB = b.Shape();
		if (shapeA != shapeB) {
			std::string errMsg = std::format("ERR: in Mul(a, b) a and b don't have the same shape. ({}, {}) != ({}, {})\n",
				shapeA[0], shapeA[1], shapeB[0], shapeB[1]);
			std::cout << errMsg;
		}
		const RealMatrix<T>& aValues = a.Values();
		const RealMatrix<T>& bValues = b.Values();
		RealMatrix<T> cValues = aValues.cwiseProduct(bValues);
		auto agt0 = (aValues.array() > 0); 
		auto bgt0 = (bValues.array() > 0);
		auto cgt0 = (cValues.array() == 0);
		Tensor<T> c(std::move(cValues));
		c.GradL() = [&b]() { return b; };
		c.GradR() = [&a]() { return a; };
		return c;
	}

	// put flag for checking overflow
	// error for overflow and underflow is not correct
	// test std::move and move semantics
	// how to avoid extra copy in a = b + c

	template<Real T>
	Tensor<T> Tensor<T>::Dot(Tensor<T>& a, Tensor<T>& b)
	{
		auto shapeA = a.Shape();
		auto shapeB = b.Shape();
		if (shapeA != shapeB) {
			std::string errMsg = std::format("ERR: in Dot(a, b) a and b don't have the same shape. ({}, {}) != ({}, {})\n",
				shapeA[0], shapeA[1], shapeB[0], shapeB[1]);
			std::cout << errMsg;
		}
		const RealMatrix<T>& aValues = a.Values();
		const RealMatrix<T>& bValues = b.Values();
		RealMatrix<T> cValues = aValues * bValues;
		auto agt0 = (aValues.array() > 0); 
		auto bgt0 = (bValues.array() > 0);
		auto cgt0 = (cValues.array() == 0);
		Tensor<T> c(std::move(cValues));
		c.GradL() = [&b, shapeA]() { return Tensor<T>(Eigen::Matrix<T, -1, -1>::Constant(shapeA[0], shapeA[1], 1) * b.Values().transpose()); };
		c.GradR() = [&a, shapeA = std::move(shapeA)]() { return Tensor<T>(Eigen::Matrix<T, -1, -1>::Constant(shapeA[0], shapeA[1], 1) * a.Values().transpose()); };
		return c;
	}

	template<Real T>
	Tensor<T> Tensor<T>::ReLU(Tensor<T>& a)
	{
		auto shapeA = a.Shape();
		const RealMatrix<T>& aValues = a.Values();
		RealMatrix<T> cValues = aValues.unaryExpr([](T x) { return x > 0 ? x : 0; });
		Tensor<T> c(std::move(cValues));
		c.GradL() = [&a, shapeA = std::move(shapeA)]() { return Tensor<T>(a.Values().unaryExpr([](T x) { return x > 0 ? 1 : 0; })); };
		c.GradR() = std::nullopt;
		return c;
	}

	// must support batch 
	template<Real T>
	Tensor<double> Tensor<T>::Cross_Entropy(Tensor<T>& a, const size_t &target)
	{
		auto shapeA = a.Shape();
		const RealMatrix<T>& aValues = a.Values();
		Eigen::MatrixXd exp_a = aValues.cast<double>().unaryExpr([](T x) { return std::exp(static_cast<double>(x)); });
		std::cout << "!!!!!!!!!!!!!!!!!!!!!\n";
		std::cout << exp_a << std::endl;
		Eigen::MatrixXd logits = (exp_a.array() / (exp_a * Eigen::MatrixXd::Ones(shapeA[1], shapeA[1])).array());
		Tensor<double> c(1, 1, -std::log10(logits(0, target)));
		c.GradL() = [logits = std::move(logits), target]() { 
			Eigen::MatrixXd grad = (logits.array() * (-1)) * logits(0, target);
			grad(0, target) += logits(0, target);
			grad.array() /= logits(0, target);
			return Tensor<double>(std::move(grad)); 
		};
		c.GradR() = std::nullopt;

		return c;
	}

	//template<Real T>
	//Tensor<double> Tensor<T>::Cross_Entropy(Tensor<T>& a, const vector<size_t>& targets)
	//{
	//	auto shapeA = a.Shape();
	//	const RealMatrix<T>& aValues = a.Values();
	//	Eigen::MatrixXd exp_a = aValues.cast<double>().unaryExpr([](T x) { return std::exp(static_cast<double>(x)); });
	//	std::cout << exp_a << std::endl;
	//	Eigen::MatrixXd logits = (exp_a.array() / (exp_a * Eigen::MatrixXd::Ones(shapeA[1], shapeA[1])).array());
	//	double loss = 0.0;
	//	for (size_t i = 0; i < shapeA[0]; i++)
	//		loss += -std::log10(logits(i, targets[i]));
	//	Tensor<double> c(1, 1, loss/shapeA[0]);
	//	c.GradL() = [logits = std::move(logits), target]() { 
	//		Eigen::MatrixXd grad = (logits.array() * (-1)) * logits(0, target);
	//		grad(0, target) += logits(0, target);
	//		grad.array() /= logits(0, target);
	//		return Tensor<double>(std::move(grad)); 
	//	};
	//	c.GradR() = std::nullopt;
	//	return c;
	//}

	//template<Real T>
	//IntMatrix<T> Sub(const IntMatrix<T>& a, const IntMatrix<T>& b);
	//template<Real T>
	//IntMatrix<T> Div(const IntMatrix<T>& a, const IntMatrix<T>& b);
	//template<Real T>
	//IntMatrix<T> Pow(const IntMatrix<T>& a, const IntMatrix<T>& b);

	//std::vector<uint32_t> Dim();
}
