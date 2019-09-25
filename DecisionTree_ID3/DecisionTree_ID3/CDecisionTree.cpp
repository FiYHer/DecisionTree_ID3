#include "CDecisionTree.h"

bool CDecisionTree::AnaliseType()
{
	//是否存储的标记
	bool bPush = false;

	//对每一个属性进行操作
	for (unsigned int i = 0; i < m_cStrAttribute.size(); i++)
	{
		//临时缓存数据，采访属性的类别
		vector<string> cTempType;

		//我们对每一排数据进行操作
		for (unsigned int j = 0; j < m_cStrData.size(); j++)
		{
			bPush = false;//初始化

			//我们只保存属性的不同类别
			for (unsigned int k = 0; k < cTempType.size(); k++)
			{
				//相同的类别
				if (cTempType[k] == m_cStrData[j][i]) { bPush = true; break; }
			}
			if (!bPush)cTempType.emplace_back(m_cStrData[j][i]);
		}
		m_cAttributeType[m_cStrAttribute[i]] = cTempType;
	}

	//对标签
	int nLabelnIndex = m_cStrData[0].size();
	for (unsigned int i = 0; i < m_cStrData.size(); i++)
	{
		bPush = false;
		for (unsigned int j = 0; j < m_cStrLabelType.size(); j++)
		{
			if (m_cStrLabelType[j] == m_cStrData[i][nLabelnIndex - 1]) { bPush = true; break; }
		}
		if (!bPush) m_cStrLabelType.emplace_back(m_cStrData[i][nLabelnIndex - 1]);
	}

	return true;
}

bool CDecisionTree::CheckAllTabel(
	vector<vector<string>> cData,
	string strLabel)
{
	//标识的数量统计
	int nLabelCount = 0;

	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < cData.size(); i++)
		if (cData[i][nLabelnIndex - 1] == strLabel) nLabelCount++;

	return nLabelCount == (cData.size());
}

double CDecisionTree::ComputeLabelEntropy(vector<vector<string>> cData)
{
	//统计每一个类别的数量
	vector<unsigned int> cCount(m_cStrLabelType.size(), 0);

	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < cData.size(); i++)
	{
		for (unsigned int j = 0; j < m_cStrLabelType.size(); j++)
		{
			if (cData[i][nLabelnIndex - 1] == m_cStrLabelType[j])cCount[j]++;
		}
	}

	//如果样例唯一性
	for (unsigned int i = 0; i < cCount.size(); i++)
		if (!cCount[i]) return 0.0;

	//获取样例总数
	double dSum = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dSum += cCount[i];

	//计算信息熵
	double dEntropy = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dEntropy -= cCount[i] / dSum * log(cCount[i] / dSum) / log(2.0);

	//返回信息熵
	return dEntropy;
}

double CDecisionTree::ComputeEntropy(
	vector<vector<string>> cData,
	string strAttribute,
	string strAttriobuteValue)
{
	//获取该属性有多少类型
	vector<int> cCount(m_cStrLabelType.size(), 0);

	//属性遍历
	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < m_cStrAttribute.size(); i++)
	{
		if (m_cStrAttribute[i] == strAttribute)
		{
			for (unsigned int j = 0; j < cData.size(); j++)
			{
				if (cData[j][i] == strAttriobuteValue)
				{
					for (unsigned int k = 0; k < m_cStrLabelType.size(); k++)
					{
						if (cData[j][nLabelnIndex - 1] == m_cStrLabelType[k])cCount[k]++;
					}
				}
			}
			break;
		}
	}

	//如果样例唯一性
	for (unsigned int i = 0; i < cCount.size(); i++)
		if (!cCount[i]) return 0.0;

	//获取样例总数
	double dSum = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dSum += cCount[i];

	//计算信息熵
	double dEntropy = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dEntropy -= cCount[i] / dSum * log(cCount[i] / dSum) / log(2.0);

	//返回信息熵
	return dEntropy;
}

double CDecisionTree::ComputeGain(
	vector<vector<string>> cData,
	string strAttribute)
{
	//获取该属性的所有类别
	vector<string> cAttributeAllType = m_cAttributeType[strAttribute];

	//该属性的类别的的比率
	vector<double> cRatio;

	//该属性的每一个类别的数量
	vector<int> cCount;

	//对该属性的所有类别进行
	for (unsigned int i = 0; i < cAttributeAllType.size(); i++)
	{
		int nTemp = 0;
		//属性查找
		for (unsigned int j = 0; j < m_cStrAttribute.size(); j++)
		{
			if (m_cStrAttribute[j] == strAttribute)
			{
				for (unsigned int k = 0; k < cData.size(); k++)
					if (cData[k][j] == cAttributeAllType[i])nTemp++;
				break;
			}
		}
		cCount.emplace_back(nTemp);
	}

	//统计他们的比率
	for (unsigned int i = 0; i < cAttributeAllType.size(); i++)
		cRatio.emplace_back((double)cCount[i] / (double)cData.size());

	//获取该属性的信息熵
	double dEntropy = 0.0;
	for (unsigned int i = 0; i < cAttributeAllType.size(); i++)
	{
		double dTemp = ComputeEntropy(cData, strAttribute, cAttributeAllType[i]);
		dEntropy += cRatio[i] * dTemp;
	}

	//返回信息熵
	return dEntropy;
}

int CDecisionTree::GetMostLabelFromData(vector<vector<string>> cData)
{
	//标签的类别数量
	vector<int> cCount(m_cStrLabelType.size(), 0);

	//对数据遍历
	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < cData.size(); i++)
	{
		for (unsigned int j = 0; j < m_cStrLabelType.size(); j++)
		{
			if (cData[i][nLabelnIndex-1] == m_cStrLabelType[j])cCount[j]++;
		}
	}

	//查找最大索引
	int nMaxnIndex = 0;
	for (unsigned int i = 0; i < cCount.size(); i++)
	{
		if (cCount[i] > cCount[nMaxnIndex])nMaxnIndex = i;
	}

	return nMaxnIndex;
}

CDecisionTree::CDecisionTree()
{
	m_pDecisionTree = nullptr;
}


CDecisionTree::~CDecisionTree()
{
}

bool CDecisionTree::ReadBufferFromFile(string&& strPath)
{
	//打开文件
	fstream cFile(strPath);
	if (!cFile.is_open())
	{
		cout << "打开文件失败..." << endl;
		return false;
	}

	//行 列 属性 的数量
	int nRow = 0, nRank = 0, nAttributeCount = 0;

	//临时数据缓存
	string strData;

	//初始化属性排
	bool bInitAttribute = false;

	//循环读取数据
	while (1)
	{
		//读取一段数据
		cFile >> strData;
		if (strData.empty())
		{
			cout << "数据读取发生错误" << endl;
			return false;
		}

		//读取到#开头的都是注释语句
		if(strData.find('#') == 0)continue;

		if (strData.find("Row") != string::npos)//读取一共有多少排数据
		{
			cFile >> strData;
			nRow = atoi(strData.c_str());
			continue;
		}
		else if (strData.find("Rank") != string::npos)//读取一共有多少列数据
		{
			cFile >> strData;
			nRank = atoi(strData.c_str());
			nAttributeCount = nRank - 1;
			continue;
		}

		if (bInitAttribute == false)//如果属性排没有初始化，就先初始化属性排
		{
			bInitAttribute = true;//初始化属性
			for (int i = 0; i < nAttributeCount; i++)
			{
				cFile >> strData;//读取一个属性
				m_cStrAttribute.emplace_back(strData);//保存属性
			}
			continue;
		}

		//读取数据和标签数据
		for (int i = 0; i < nRow; i++)
		{
			vector<string> cTempData;
			for (int j = 0; j < nRank; j++)
			{
				cFile >> strData;	
				cTempData.emplace_back(strData);
			}
			m_cStrData.emplace_back(cTempData);
		}
		break;
	}

	//输出属性
	for (unsigned int i = 0; i < m_cStrAttribute.size(); i++)
	{
		cout << m_cStrAttribute[i] << "\t";
	}
	cout << endl << endl;

	//输出数据和标签
	for (unsigned int i = 0; i < m_cStrData.size(); i++)
	{
		for (unsigned int j = 0; j < m_cStrData[0].size(); j++)
		{
			if (j == m_cStrData[0].size() - 1) { cout << "  ->  " << m_cStrData[i][j] << endl; break;}
			cout << m_cStrData[i][j] << "\t";
		}
	}
	cout << endl << endl;

	//关闭文件
	cFile.close();

	//顺便分类
	return AnaliseType();
}

PTreeNode CDecisionTree::BuildDecisionTree(
	PTreeNode pTreeNode, 
	vector<vector<string>> cData , 
	vector<string> cAttribute)
{
	//创建决策数节点
	if (pTreeNode == nullptr) pTreeNode = new TreeNode;
	if (m_pDecisionTree == nullptr) m_pDecisionTree = pTreeNode;

	//样例单一性判断
	for (unsigned int i = 0; i < m_cStrLabelType.size(); i++)
	{
		if (CheckAllTabel(cData, m_cStrLabelType[i]))
		{
			pTreeNode->strAttribute = m_cStrLabelType[i];
			return pTreeNode;
		}
	}

	//获取样例的信息熵
	double dLabelEntropy = ComputeLabelEntropy(cData);

	//最大的信息增益
	double dMaxGain = 0.0;

	//最大的信息增益的迭代器指针
	vector<string>::iterator it_MaxGain;

	cout << "标签信息熵:"<<dLabelEntropy << endl;

	//属性的信息增益计算
	for (vector<string>::iterator it = cAttribute.begin(); it != cAttribute.end(); it++)
	{
		//获取属性的信息增益
		double dTempGain = ComputeGain(cData, (*it));
		dTempGain = dLabelEntropy - dTempGain;

		cout << "属性:" << *it << "\t信息熵:" << dTempGain << endl;

		//获取最大增益和属性
		if (dTempGain > dMaxGain)
		{
			dMaxGain = dTempGain;
			it_MaxGain = it;
		}
	}

	cout << "信息熵对比结束" << endl << endl;

	//新的属性集
	vector<string> cNewAttribute;

	//新的数据集
	vector<vector<string>> cNewData;

	//剔除最大信息熵属性作为节点
	for (vector<string>::iterator it = cAttribute.begin(); it != cAttribute.end(); it++)
	{
		if ((*it_MaxGain) != (*it))cNewAttribute.emplace_back(*it);
	}

	//保存节点属性
	pTreeNode->strAttribute = *it_MaxGain;

	//获取该属性的所有类别
	vector<string> cAttributeType = m_cAttributeType[*it_MaxGain];

	//获取该属性的索引
	int nAttributenIndex = 0;
	for (vector<string>::iterator it = m_cStrAttribute.begin(); it != m_cStrAttribute.end(); it++)
	{
		if (*it == *it_MaxGain) break;
		nAttributenIndex++;
	}

	//决策数层数
	m_nTreeDepth++;

	//属性排查
	for (vector<string>::iterator it = cAttributeType.begin(); it != cAttributeType.end(); it++)
	{
		for (unsigned int i = 0; i < cData.size(); i++)
		{
			if (cData[i][nAttributenIndex] == *it)cNewData.emplace_back(cData[i]);
		}

		//创建一个新的节点
		PTreeNode pNewNode = new TreeNode();

		//设置新的节点的属性值
		pNewNode->strAttributeValue = *it;

		//如果新数据为0
		if (!cNewData.size())
		{
			int nMaxnIndex = GetMostLabelFromData(cData);
			pNewNode->strAttributeValue = m_cStrLabelType[nMaxnIndex];
		}
		else BuildDecisionTree(pNewNode, cNewData, cNewAttribute);

		//加入孩子节点
		pTreeNode->cChildNode.emplace_back(pNewNode);

		//清除
		cNewData.clear();
	}

	return pTreeNode;
}

bool CDecisionTree::ReleaseDecisionTree(PTreeNode pTreeNode)
{
	if (pTreeNode == nullptr)return false;

	for (vector<_TreeNode*>::iterator it = pTreeNode->cChildNode.begin(); it != pTreeNode->cChildNode.end(); it++)
		ReleaseDecisionTree(*it);

	delete pTreeNode;
	pTreeNode = nullptr;
	return true;
}

void CDecisionTree::ShowDecisionTree(PTreeNode pNode,int nTreeDepth)
{
	for (int i = 0; i < nTreeDepth; i++)cout << "\t";
	if (!pNode->strAttributeValue.empty())
	{
		cout << "(" << pNode->strAttributeValue << ")" << endl;
		for (int i = 0; i < nTreeDepth + 1; i++)cout << "\t";
	}
	cout << "[" << pNode->strAttribute << "]" << endl;
	for (vector<PTreeNode>::iterator it = pNode->cChildNode.begin(); it != pNode->cChildNode.end(); it++)
	{
		ShowDecisionTree(*it, nTreeDepth + 1);
	}
}
