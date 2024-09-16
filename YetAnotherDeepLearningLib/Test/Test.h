
#ifndef TEST_H
#define TEST_H

#include <iostream>

namespace deepl {
	template<typename T>
	class Test {
	public:
		Test();
		Test(T i);
		void Print() const;
		int mI;
	};
}

#include "Test.cpp"

#endif
