
#include "CDecisionTree.h"

void TestDecisionTree()
{
	//������������
	CDecisionTree MyTree;

	//���ļ���ȡ����
	MyTree.ReadBufferFromFile("TestData.txt");

	//����������
	MyTree.BuildDecisionTree(MyTree.GetDecisionTreePoint(), MyTree.GetData(), MyTree.GetAttribute());

	//���������
	MyTree.ShowDecisionTree(MyTree.GetDecisionTreePoint(), 0);

	//�ͷž�����
	MyTree.ReleaseDecisionTree(MyTree.GetDecisionTreePoint());
}



int main(int argc, char* argv[])
{
	TestDecisionTree();
	getchar();
	return 1;
}
