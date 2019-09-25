#include "CDecisionTree.h"

bool CDecisionTree::AnaliseType()
{
	//�Ƿ�洢�ı��
	bool bPush = false;

	//��ÿһ�����Խ��в���
	for (unsigned int i = 0; i < m_cStrAttribute.size(); i++)
	{
		//��ʱ�������ݣ��ɷ����Ե����
		vector<string> cTempType;

		//���Ƕ�ÿһ�����ݽ��в���
		for (unsigned int j = 0; j < m_cStrData.size(); j++)
		{
			bPush = false;//��ʼ��

			//����ֻ�������ԵĲ�ͬ���
			for (unsigned int k = 0; k < cTempType.size(); k++)
			{
				//��ͬ�����
				if (cTempType[k] == m_cStrData[j][i]) { bPush = true; break; }
			}
			if (!bPush)cTempType.emplace_back(m_cStrData[j][i]);
		}
		m_cAttributeType[m_cStrAttribute[i]] = cTempType;
	}

	//�Ա�ǩ
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
	//��ʶ������ͳ��
	int nLabelCount = 0;

	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < cData.size(); i++)
		if (cData[i][nLabelnIndex - 1] == strLabel) nLabelCount++;

	return nLabelCount == (cData.size());
}

double CDecisionTree::ComputeLabelEntropy(vector<vector<string>> cData)
{
	//ͳ��ÿһ����������
	vector<unsigned int> cCount(m_cStrLabelType.size(), 0);

	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < cData.size(); i++)
	{
		for (unsigned int j = 0; j < m_cStrLabelType.size(); j++)
		{
			if (cData[i][nLabelnIndex - 1] == m_cStrLabelType[j])cCount[j]++;
		}
	}

	//�������Ψһ��
	for (unsigned int i = 0; i < cCount.size(); i++)
		if (!cCount[i]) return 0.0;

	//��ȡ��������
	double dSum = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dSum += cCount[i];

	//������Ϣ��
	double dEntropy = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dEntropy -= cCount[i] / dSum * log(cCount[i] / dSum) / log(2.0);

	//������Ϣ��
	return dEntropy;
}

double CDecisionTree::ComputeEntropy(
	vector<vector<string>> cData,
	string strAttribute,
	string strAttriobuteValue)
{
	//��ȡ�������ж�������
	vector<int> cCount(m_cStrLabelType.size(), 0);

	//���Ա���
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

	//�������Ψһ��
	for (unsigned int i = 0; i < cCount.size(); i++)
		if (!cCount[i]) return 0.0;

	//��ȡ��������
	double dSum = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dSum += cCount[i];

	//������Ϣ��
	double dEntropy = 0.0;
	for (unsigned int i = 0; i < cCount.size(); i++)
		dEntropy -= cCount[i] / dSum * log(cCount[i] / dSum) / log(2.0);

	//������Ϣ��
	return dEntropy;
}

double CDecisionTree::ComputeGain(
	vector<vector<string>> cData,
	string strAttribute)
{
	//��ȡ�����Ե��������
	vector<string> cAttributeAllType = m_cAttributeType[strAttribute];

	//�����Ե����ĵı���
	vector<double> cRatio;

	//�����Ե�ÿһ����������
	vector<int> cCount;

	//�Ը����Ե�����������
	for (unsigned int i = 0; i < cAttributeAllType.size(); i++)
	{
		int nTemp = 0;
		//���Բ���
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

	//ͳ�����ǵı���
	for (unsigned int i = 0; i < cAttributeAllType.size(); i++)
		cRatio.emplace_back((double)cCount[i] / (double)cData.size());

	//��ȡ�����Ե���Ϣ��
	double dEntropy = 0.0;
	for (unsigned int i = 0; i < cAttributeAllType.size(); i++)
	{
		double dTemp = ComputeEntropy(cData, strAttribute, cAttributeAllType[i]);
		dEntropy += cRatio[i] * dTemp;
	}

	//������Ϣ��
	return dEntropy;
}

int CDecisionTree::GetMostLabelFromData(vector<vector<string>> cData)
{
	//��ǩ���������
	vector<int> cCount(m_cStrLabelType.size(), 0);

	//�����ݱ���
	int nLabelnIndex = cData[0].size();
	for (unsigned int i = 0; i < cData.size(); i++)
	{
		for (unsigned int j = 0; j < m_cStrLabelType.size(); j++)
		{
			if (cData[i][nLabelnIndex-1] == m_cStrLabelType[j])cCount[j]++;
		}
	}

	//�����������
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
	//���ļ�
	fstream cFile(strPath);
	if (!cFile.is_open())
	{
		cout << "���ļ�ʧ��..." << endl;
		return false;
	}

	//�� �� ���� ������
	int nRow = 0, nRank = 0, nAttributeCount = 0;

	//��ʱ���ݻ���
	string strData;

	//��ʼ��������
	bool bInitAttribute = false;

	//ѭ����ȡ����
	while (1)
	{
		//��ȡһ������
		cFile >> strData;
		if (strData.empty())
		{
			cout << "���ݶ�ȡ��������" << endl;
			return false;
		}

		//��ȡ��#��ͷ�Ķ���ע�����
		if(strData.find('#') == 0)continue;

		if (strData.find("Row") != string::npos)//��ȡһ���ж���������
		{
			cFile >> strData;
			nRow = atoi(strData.c_str());
			continue;
		}
		else if (strData.find("Rank") != string::npos)//��ȡһ���ж���������
		{
			cFile >> strData;
			nRank = atoi(strData.c_str());
			nAttributeCount = nRank - 1;
			continue;
		}

		if (bInitAttribute == false)//���������û�г�ʼ�������ȳ�ʼ��������
		{
			bInitAttribute = true;//��ʼ������
			for (int i = 0; i < nAttributeCount; i++)
			{
				cFile >> strData;//��ȡһ������
				m_cStrAttribute.emplace_back(strData);//��������
			}
			continue;
		}

		//��ȡ���ݺͱ�ǩ����
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

	//�������
	for (unsigned int i = 0; i < m_cStrAttribute.size(); i++)
	{
		cout << m_cStrAttribute[i] << "\t";
	}
	cout << endl << endl;

	//������ݺͱ�ǩ
	for (unsigned int i = 0; i < m_cStrData.size(); i++)
	{
		for (unsigned int j = 0; j < m_cStrData[0].size(); j++)
		{
			if (j == m_cStrData[0].size() - 1) { cout << "  ->  " << m_cStrData[i][j] << endl; break;}
			cout << m_cStrData[i][j] << "\t";
		}
	}
	cout << endl << endl;

	//�ر��ļ�
	cFile.close();

	//˳�����
	return AnaliseType();
}

PTreeNode CDecisionTree::BuildDecisionTree(
	PTreeNode pTreeNode, 
	vector<vector<string>> cData , 
	vector<string> cAttribute)
{
	//�����������ڵ�
	if (pTreeNode == nullptr) pTreeNode = new TreeNode;
	if (m_pDecisionTree == nullptr) m_pDecisionTree = pTreeNode;

	//������һ���ж�
	for (unsigned int i = 0; i < m_cStrLabelType.size(); i++)
	{
		if (CheckAllTabel(cData, m_cStrLabelType[i]))
		{
			pTreeNode->strAttribute = m_cStrLabelType[i];
			return pTreeNode;
		}
	}

	//��ȡ��������Ϣ��
	double dLabelEntropy = ComputeLabelEntropy(cData);

	//������Ϣ����
	double dMaxGain = 0.0;

	//������Ϣ����ĵ�����ָ��
	vector<string>::iterator it_MaxGain;

	cout << "��ǩ��Ϣ��:"<<dLabelEntropy << endl;

	//���Ե���Ϣ�������
	for (vector<string>::iterator it = cAttribute.begin(); it != cAttribute.end(); it++)
	{
		//��ȡ���Ե���Ϣ����
		double dTempGain = ComputeGain(cData, (*it));
		dTempGain = dLabelEntropy - dTempGain;

		cout << "����:" << *it << "\t��Ϣ��:" << dTempGain << endl;

		//��ȡ������������
		if (dTempGain > dMaxGain)
		{
			dMaxGain = dTempGain;
			it_MaxGain = it;
		}
	}

	cout << "��Ϣ�ضԱȽ���" << endl << endl;

	//�µ����Լ�
	vector<string> cNewAttribute;

	//�µ����ݼ�
	vector<vector<string>> cNewData;

	//�޳������Ϣ��������Ϊ�ڵ�
	for (vector<string>::iterator it = cAttribute.begin(); it != cAttribute.end(); it++)
	{
		if ((*it_MaxGain) != (*it))cNewAttribute.emplace_back(*it);
	}

	//����ڵ�����
	pTreeNode->strAttribute = *it_MaxGain;

	//��ȡ�����Ե��������
	vector<string> cAttributeType = m_cAttributeType[*it_MaxGain];

	//��ȡ�����Ե�����
	int nAttributenIndex = 0;
	for (vector<string>::iterator it = m_cStrAttribute.begin(); it != m_cStrAttribute.end(); it++)
	{
		if (*it == *it_MaxGain) break;
		nAttributenIndex++;
	}

	//����������
	m_nTreeDepth++;

	//�����Ų�
	for (vector<string>::iterator it = cAttributeType.begin(); it != cAttributeType.end(); it++)
	{
		for (unsigned int i = 0; i < cData.size(); i++)
		{
			if (cData[i][nAttributenIndex] == *it)cNewData.emplace_back(cData[i]);
		}

		//����һ���µĽڵ�
		PTreeNode pNewNode = new TreeNode();

		//�����µĽڵ������ֵ
		pNewNode->strAttributeValue = *it;

		//���������Ϊ0
		if (!cNewData.size())
		{
			int nMaxnIndex = GetMostLabelFromData(cData);
			pNewNode->strAttributeValue = m_cStrLabelType[nMaxnIndex];
		}
		else BuildDecisionTree(pNewNode, cNewData, cNewAttribute);

		//���뺢�ӽڵ�
		pTreeNode->cChildNode.emplace_back(pNewNode);

		//���
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
