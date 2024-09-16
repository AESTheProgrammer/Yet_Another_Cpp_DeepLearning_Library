#include "Test.h"

namespace deepl {
	template<typename T>
	Test<T>::Test() {};

	template<typename T>
	Test<T>::Test(T i) : mI(i) {};

	template<typename T>
	void Test<T>::Print() const {
		std::cout << "print Test:" << mI << '\n';
	}
}