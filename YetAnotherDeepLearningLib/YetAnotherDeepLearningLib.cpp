// YetAnotherDeepLearningLib.cpp : Defines the entry point for the application.
//

#include <Eigen/Dense>
#include <limits>
#include <cmath>

#include "YetAnotherDeepLearningLib.h"
#include "Test/Test.h"
#include "Tensor/Tensor.h"

using namespace std;
using namespace deepl;


//int main() {
//	//Test<float> test{3.0f};
//	Test<float> test;
//	test.mI = 10.0f;
//	test.Print();
//	return 0;
//}

typedef Tensor<int32_t> Ti32;
typedef Tensor<double> Td;

void testTensor() {
	Ti32 t1(3, 3, 4);
	Ti32 t2(3, 3, 9);
	Ti32 t3 = Ti32::Add(t1, t2);
	Ti32 t4 = Ti32::Mul(t1, t2);
	Ti32 t5 = Ti32::Dot(t1, t2);
	std::cout << "========GradL============" << std::endl;
	t3.GradL().value()().Print();
	std::cout << "========GradR============" << std::endl;
	t3.GradR().value()().Print();
	std::cout << "========t1+t2============" << std::endl;
	t3.Print();
	std::cout << "=========================" << std::endl;
	std::cout << "========GradL============" << std::endl;
	t4.GradL().value()().Print();
	std::cout << "========GradR============" << std::endl;
	t4.GradR().value()().Print();
	std::cout << "========t1*t2============" << std::endl;
	t4.Print();
	std::cout << "=========================" << std::endl;
	std::cout << "========GradL============" << std::endl;
	t5.GradL().value()().Print();
	std::cout << "========GradR============" << std::endl;
	t5.GradR().value()().Print();
	std::cout << "========t1@t2============" << std::endl;
	t5.Print();
	std::cout << "=========================" << std::endl;
	std::flush(std::cout);
}

// Check the available options 
int main2() {
	const int32_t c = 10;
	Tensor<int32_t> a (5, 5, c);
	a.Print();
	std::cout << "\n\n";
	Eigen::Matrix2i g = -Eigen::MatrixXi::Constant(2, 2, std::numeric_limits<int>::max());
	Eigen::Matrix2i gg;
	gg << 1, 2, std::numeric_limits<int>::min(), 1;
	auto z = gg + g;
	std::cout << z << "\n\n";
	std::cout << z + g << "\n\n";
	std::cout << z + gg << "\n\n";
	std::cout << (g.array() > 0) << "\n\n";
	std::cout << (gg.array() > 0) << "\n\n";
	std::cout << (z.array() > 0) << "\n\n";
	std::cout << (((g.array() > 0 && gg.array() > 0) == z.array() > 0) || (g.array() > 0 ^ gg.array() > 0)) << "\n\n";
	Eigen::Vector3i q(1, 2, 3);
	std::cout << q.rows() << q.cols() << std::endl;
	return 1;
}

void main4() {
	Eigen::RowVector4i  a;
	a << 1, 1, 1, 1;
	Tensor<int> t(a);
	std::cout << "test Cross Entropy" << std::endl;
	t.Print();
	Td l = Ti32::Cross_Entropy(t, 3);
	std::cout << "Loss: " << std::endl;
	l.Print();
	std::cout << "Grad: " << std::endl;
	Tensor<double> grad = l.GradL().value()();
	grad.Print();
}



int main() {
	main4();
	Eigen::Matrix3i a;
	a << 1, 3, 4, 2, 6, 2, 1, 1, 1;

	Eigen::Matrix3i d;
	d << 1, 3, 4, 2, 6, 2, 1, 1, 1;

	Eigen::MatrixXf mat(2, 4);
	mat <<	1, 2, 6, 9,
			3, 1, 7, 2;
	Eigen::MatrixXf::Index maxIndex[2];
	Eigen::VectorXf maxVal(2);
	for (int i = 0; i < mat.rows(); i++) {
		maxVal(i) = mat.row(i).maxCoeff(&maxIndex[i]);
	}
	for (auto a : maxIndex)
		std::cout << a << std::endl;
	std::cout << mat(0, maxIndex[0]);
	std::cout << "==========================\n";

	std::cout << "Maxima at positions " << endl;
	std::cout << maxIndex << std::endl;
	std::cout << "maxVal " << maxVal << endl;

	std::cout << a.array() / d.array();
	Eigen::Matrix3d exp_a = a.unaryExpr([](int x) { return std::exp(static_cast<double>(x)); }).cast<double>();
	Eigen::Matrix3d b = (exp_a.array() / (exp_a * Eigen::MatrixXd::Ones(3, 3)).array());
	//// a.unaryExpr([](int x) { return x > 0 ? x : -0.1 * x; });
	std::cout << std::endl << b;
	//testTensor();
}

