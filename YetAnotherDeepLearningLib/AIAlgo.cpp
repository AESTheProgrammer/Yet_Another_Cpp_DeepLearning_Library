// AIAlgo.cpp : Defines the entry point for the application.
//

#include "AIAlgo.h"
//#include "GradientDescent/GradientDescent.h"
//#include "Tensor/Tensor.h"
#include "Test/Test.h"
#include <Eigen/Dense>

using namespace std;
using namespace aialgo;

//int main() {
	//Test test(3);
	//test.Print();
	//return 0;
	//return 0;
//}

//int main() {
//	const int32_t c = 10;
//	Tensor<int32_t> a (5, 5, c);
//	a.Print();
//	return 1;
//}

//int main2()
//{
//	auto func = [](vector<double>* point) { 
//		double x = point->at(0);
//		return x * x;
//	};
//	auto gd = GradientDescent();
//	gd.SetStartPoint(vector<double>{5.0});
//	gd.SetFunc(func);
//	std::pair<vector<double>, double> result = gd.Optimize();
//	//for (size_t i = 0; i < result.first.size(); i++) {
//	//	cout << result.first[i] << ", ";
//	//}
//	for (auto elem : result.first) {
//		cout << elem << ", ";
//	}
//	cout << endl;
//	cout << "point: " << "\n" << "mFunc(point): " << result.second;
//	return 0;
//}

int main() {
	Eigen::MatrixXd md(2, 2);
	md(0, 0) = 3;
	md(1, 0) = 2.5;
	md(0, 1) = -1;
	md(1, 1) = md(1, 0) + md(0, 1);
	std::cout << "Here is the matrix md:\n" << md << std::endl;
	Eigen::VectorXd v(2);
	v(0) = 4;
	v(1) = v(0) - 1;
	std::cout << "Here is the vector v:\n" << v << std::endl;
	Eigen::MatrixXd mi(3, 3);
	mi.resize(2, 2);
	mi << 1, 2, 3, 4;
	std::cout << "md + mi:\n" << md + mi << std::endl;
	Eigen::Matrix3f mg(3, 3);
	mg << 1, 2, 3, 1, 2, 3, 1, 2, 3;
	std::cout << mg * mg << std::endl;
	Eigen::MatrixXcf a = Eigen::MatrixXcf::Random(2, 2);
	cout << "Here is the matrix a\n" << a << endl;
	cout << "Here is the matrix a^T\n" << a.transpose() << endl;
	cout << "Here is the conjugate of a\n" << a.conjugate() << endl;
	cout << "Here is the matrix a^*\n" << a.adjoint() << endl;
	Eigen::Matrix3f mg2(3, 3);
	mg2 = mg;
	mg2(0, 1) = 24;
	std::cout << mg2 - mg << std::endl;
	std::cout << "=====================================\n";
	vector<Eigen::Matrix3i> mgs;
	for (int i = 0; i < 2; i++) {
		mgs[i] << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	}
	for (int i = 0; i < 2; i++) {
		std::cout << mgs[i] << std::endl;
	}
	return 1;
}
// Question how to create 3d matrixes
