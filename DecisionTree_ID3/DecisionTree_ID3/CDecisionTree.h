#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

//决策树节点
typedef struct _TreeNode
{
	string strAttribute;				//属性
	string strAttributeValue;			//属性值
	vector<_TreeNode*> cChildNode;		//孩子节点
}TreeNode,*PTreeNode;

class CDecisionTree
{
private:
	vector<string> m_cStrAttribute;					//字符串属性
	vector<vector<string>> m_cStrData;				//字符串数据
	map<string, vector<string>> m_cAttributeType;	//字符串属性与类别对应

	vector<string> m_cStrLabelType;					//字符串标签的类别

	PTreeNode m_pDecisionTree;						//决策树

	int m_nTreeDepth;								//决策树的深度

	//统计类别
	bool AnaliseType();

	//检查样例是否单一
	bool CheckAllTabel(vector<vector<string>> cData, string strLabel);

	//计算标签的信息熵
	double ComputeLabelEntropy(vector<vector<string>> cData);

	//计算属性的信息熵
	double ComputeEntropy(vector<vector<string>> cData, string strAttribute, string strAttriobuteValue);

	//计算属性的信息增益
	double ComputeGain(vector<vector<string>> cData, string strAttribute);

	//获取样例中做多的标签
	int GetMostLabelFromData(vector<vector<string>> cData);

public:
	CDecisionTree();
	~CDecisionTree();

	inline PTreeNode GetDecisionTreePoint()const { return m_pDecisionTree; }
	inline vector<vector<string>> GetData()const { return m_cStrData; }
	inline vector<string> GetAttribute()const { return m_cStrAttribute; }

	//从文件里面读取字符串数据
	bool ReadBufferFromFile(string&& strPath);

	//创建决策树
	PTreeNode BuildDecisionTree(PTreeNode pTreeNode, vector<vector<string>> cData, vector<string> cAttribute);

	//释放决策树
	bool ReleaseDecisionTree(PTreeNode pTreeNode);

	//显示决策树结构
	void ShowDecisionTree(PTreeNode pNode,int nTreeDepth);

};

