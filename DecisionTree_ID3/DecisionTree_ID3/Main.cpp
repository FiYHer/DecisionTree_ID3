
#include "CDecisionTree.h"

void TestDecisionTree()
{
	//创建决策数类
	CDecisionTree MyTree;

	//从文件读取数据
	MyTree.ReadBufferFromFile("TestData.txt");

	//创建决策树
	MyTree.BuildDecisionTree(MyTree.GetDecisionTreePoint(), MyTree.GetData(), MyTree.GetAttribute());

	//输出决策树
	MyTree.ShowDecisionTree(MyTree.GetDecisionTreePoint(), 0);

	//释放决策数
	MyTree.ReleaseDecisionTree(MyTree.GetDecisionTreePoint());
}



int main(int argc, char* argv[])
{
	TestDecisionTree();
	getchar();
	return 1;
}
