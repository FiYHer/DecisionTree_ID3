#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

//�������ڵ�
typedef struct _TreeNode
{
	string strAttribute;				//����
	string strAttributeValue;			//����ֵ
	vector<_TreeNode*> cChildNode;		//���ӽڵ�
}TreeNode,*PTreeNode;

class CDecisionTree
{
private:
	vector<string> m_cStrAttribute;					//�ַ�������
	vector<vector<string>> m_cStrData;				//�ַ�������
	map<string, vector<string>> m_cAttributeType;	//�ַ�������������Ӧ

	vector<string> m_cStrLabelType;					//�ַ�����ǩ�����

	PTreeNode m_pDecisionTree;						//������

	int m_nTreeDepth;								//�����������

	//ͳ�����
	bool AnaliseType();

	//��������Ƿ�һ
	bool CheckAllTabel(vector<vector<string>> cData, string strLabel);

	//�����ǩ����Ϣ��
	double ComputeLabelEntropy(vector<vector<string>> cData);

	//�������Ե���Ϣ��
	double ComputeEntropy(vector<vector<string>> cData, string strAttribute, string strAttriobuteValue);

	//�������Ե���Ϣ����
	double ComputeGain(vector<vector<string>> cData, string strAttribute);

	//��ȡ����������ı�ǩ
	int GetMostLabelFromData(vector<vector<string>> cData);

public:
	CDecisionTree();
	~CDecisionTree();

	inline PTreeNode GetDecisionTreePoint()const { return m_pDecisionTree; }
	inline vector<vector<string>> GetData()const { return m_cStrData; }
	inline vector<string> GetAttribute()const { return m_cStrAttribute; }

	//���ļ������ȡ�ַ�������
	bool ReadBufferFromFile(string&& strPath);

	//����������
	PTreeNode BuildDecisionTree(PTreeNode pTreeNode, vector<vector<string>> cData, vector<string> cAttribute);

	//�ͷž�����
	bool ReleaseDecisionTree(PTreeNode pTreeNode);

	//��ʾ�������ṹ
	void ShowDecisionTree(PTreeNode pNode,int nTreeDepth);

};

