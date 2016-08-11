package deepDriver.dl.aml.cart;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import deepDriver.dl.aml.utils.AccuracyCaculator;

public class Cart implements Serializable {
	
	int maxDeepth = 3;
	double errorAccuracy = 0.00001;
	DecisionTree tree = new DecisionTree();
	
	public double[] predict(DataSet dataSet) {
		if (bestTree == null) {
			bestTree = tree;
		}
		return predict(bestTree.getRoot() ,dataSet);
	}
	
	public double[] predict(DecisionNode node, DataSet dataSet) {
		double [] ys = new double[dataSet.getDependentVars().length];
		double [][] vars = dataSet.getDependentVars();
		for (int i = 0; i < ys.length; i++) {
			ys[i] = predict(node ,vars[i]) ;
		}
		return ys;
	}
	
	public double predict(DecisionNode node ,double [] var) {
		double y = node.getCurrentY();
		if (var[node.indexOfVars] < node.decisionCondition ) {
//			y = node.leftY;
			if (node.leftNode != null) {
				y = y + predict(node.leftNode ,var);
			}
		} else {
//			y = node.rightY;
			if (node.rightNode != null) {
				y = y + predict(node.rightNode ,var);
			}
		}
		return y;
	}
	
	public DecisionTree trainTree(DataSet dataSet) {		
		DecisionNode currentNode = createNode(0, null, dataSet, 1);
		tree.setRoot(currentNode);
		pruning();
		return tree;
	}
	
	public DecisionNode createNode(double currentY, DecisionNode parent, DataSet dataSet, int deepth) {
		DecisionNode node = new DecisionNode();
		node.setParent(parent);
		node.setCurrentY(currentY);
		node.setTrainingDataSet(dataSet);
		double [][] vars = dataSet.getDependentVars();
		double [] ys = dataSet.getIndependentVars();
		double minErr = 0;
		double minS = 0;
		int minJ = 0;
		int minIndex = 0;
		double leftY = 0;
		double rightY = 0;
		MapData [] minData = null;
		boolean init = false;
		for (int i = 0; i < vars[0].length; i++) {
			MapData [] data = getSortedMapData(i, dataSet);
			for (int g = 0; g <= data.length; g++) {
				double err = 0;
				double s = 0;
				double leftSum = 0;
				double rightSum = 0;
				int leftCnt = 0;
				if (g == 0) {
					s = data[0].value - 1; 
				} else if (g == data.length) {
					s = data[data.length - 1].value + 1; 
				} else {
					s = (data[g-1].value + data[g].value) /2.0 ; 
				}
				for (int j2 = 0; j2 < data.length; j2++) {
					if (data[j2].value < s) {
						leftSum = leftSum +ys[data[j2].pos];
						leftCnt ++;
					} else {
						rightSum = rightSum +ys[data[j2].pos];
					}
				}
				double leftAvg = 0;
				double rightAvg = 0;
				if (leftCnt == 0) { 
					rightAvg =  rightSum/(double)(data.length);
				} else if (leftCnt == data.length) {			
					leftAvg =  leftSum/(double)(data.length);
				} else {
					leftAvg =  leftSum/(double)(leftCnt);
					rightAvg =  rightSum/(double)(data.length - leftCnt);
				}
				for (int j2 = 0; j2 < data.length; j2++) {
					if (data[j2].value < s) {
						err = err + Math.pow((ys[data[j2].pos] - leftAvg) , 2.0);
					} else {
						err = err + Math.pow((ys[data[j2].pos] - rightAvg) , 2.0);
					}
				}
				if (minErr > err || !init) {
					init = true;
					minErr = err;
					minS = s;
					minJ = leftCnt;
					minIndex = i;
					minData = data;
					leftY = leftAvg;
					rightY = rightAvg;
				}
			}
		}
		node.decisionCondition = minS;
		node.indexOfVars = minIndex;
//		node.leftY = leftY;
//		node.rightY = rightY;
		
		//split data now..., so many pointers will not be released :-(
		DataSet leftDataSet = new DataSet();		
		DataSet rightDataSet = new DataSet();		
		leftDataSet.dependentVars = new double[minJ][dataSet.dependentVars[0].length];
		leftDataSet.independentVars = new double[minJ];
		leftDataSet.labels = new String[minJ];
		rightDataSet.dependentVars = new double[minData.length - minJ][dataSet.dependentVars[0].length];
		rightDataSet.independentVars = new double[minData.length - minJ];
		rightDataSet.labels = new String[minData.length - minJ];
		for (int k = 0; k < minData.length; k++) {
			if (k < minJ) {
				leftDataSet.dependentVars[k] = dataSet.dependentVars[minData[k].pos];
				leftDataSet.independentVars[k] = 
						dataSet.independentVars[minData[k].pos] - leftY;
				leftDataSet.labels[k] = dataSet.labels[minData[k].pos];
			} else {
				int l = k - minJ;
				rightDataSet.dependentVars[l] = dataSet.dependentVars[minData[k].pos];
				rightDataSet.independentVars[l] = 
						dataSet.independentVars[minData[k].pos] - rightY;
				rightDataSet.labels[l] = dataSet.labels[minData[k].pos];
			}
		}
		//deepth < maxDeepth ||
//		System.out.println("Deepth is: "+deepth);
		if (deepth < maxDeepth && minErr > errorAccuracy) {//minErr need to take into consideration also
			if (minJ == 0) {
//				leftDataSet = null;
				node.setLeftNode(createNode(leftY, node));
				node.setRightNode(createNode(rightY, node, rightDataSet, deepth + 1));
			} else if (minJ == minData.length) {
//				rightDataSet = null;
				node.setRightNode(createNode(rightY, node));
				node.setLeftNode(createNode(leftY, node, leftDataSet, deepth + 1));
			} else {
				node.setLeftNode(createNode(leftY, node, leftDataSet, deepth + 1));
				node.setRightNode(createNode(rightY, node, rightDataSet, deepth + 1));
			}
		} else {			
			node.setLeftNode(createNode(leftY, node));
			node.setRightNode(createNode(rightY, node));
		}
		return node;
	}
	
	public DecisionNode createNode(double currentY, DecisionNode parent) {
		DecisionNode nd = new DecisionNode();
		nd.setParent(parent);
		nd.setCurrentY(currentY);
		return nd;
	}
	
	public MapData [] getSortedMapData(int index, DataSet dataSet) {
		MapData [] data = new MapData[dataSet.getDependentVars().length];
		double [][] vars = dataSet.getDependentVars();
		for (int i = 0; i < data.length; i++) {
			data[i] = new MapData();
			data[i].pos = i;
			data[i].value = vars[i][index];
		}
		MapData tmpData = null;
		for (int i = 0; i < data.length; i++) {
			for (int j = i+1; j < data.length; j++) {
				if (data[i].value > data[j].value) {
					tmpData = data[i];
					data[i] = data[j];
					data[j] = tmpData;
				}
			}
		}
		return data;
	}
	
	class MapData implements Serializable {
		int pos;
		double value;
	}
	
	DecisionTree bestTree = null;
	
	public void	lookupBestTree(DataSet ds) {
		AccuracyCaculator acc = new AccuracyCaculator();
		double [] ys = ds.getIndependentVars();
		double[] pv = predict(ds);
		double min = acc.caculateAccuracy(ys, pv);
		System.out.println("The best acc is: "+min);
		bestTree = tree;
		for (int i = 0; i < treeList.size(); i++) {
			DecisionTree dt = treeList.get(i);
			pv = predict(dt.getRoot(), ds);
			double curr = acc.caculateAccuracy(ys, pv);
//			System.out.println("The current acc is: "+curr);
			if (curr > min) {
				System.out.println("The best acc is: "+min);
				min = curr;
				bestTree = dt;				
			}
		}
	}
	
	public void pruning() {
		caculateAlpha(tree.getRoot());
		DecisionTree dt = tree;
		treeList.add(dt);
		PruningAlpha [] alphas = sortAlpha();
		for (int i = 0; i < alphas.length; i++) {
			PruningAlpha alpha = alphas[i];
			dt = replicate(dt);
			pruning(alpha, dt.getRoot());
			treeList.add(dt);
			if (dt.getRoot().getLeftNode() == null && dt.getRoot().getRightNode() == null) {
				break;
			}
		}
	}
	
	
	public PruningAlpha [] sortAlpha() {
		PruningAlpha [] alphas = new PruningAlpha[alphaList.size()];
		for (int i = 0; i < alphas.length; i++) {
			alphas[i] = alphaList.get(i);			
		}
		PruningAlpha tmp = null;
		for (int i = 0; i < alphas.length; i++) {
			for (int j = i+1; j < alphas.length; j++) {
				if (alphas[i].getAlpha() > alphas[j].getAlpha()) {
					tmp = alphas[i];
					alphas[i] = alphas[j];
					alphas[j] = tmp;
				}
			}
		}
		return alphas;
	}
	
	public boolean pruning(PruningAlpha alpha, DecisionNode dn) {
		if (dn == null) {
			return false;
		}
		if (dn.getLeftNode() == null && dn.getRightNode() == null) {
			return false;
		}
		if (dn.getAlpha() == alpha.getAlpha()) {
			dn.setLeftNode(null);
			dn.setRightNode(null);
			return true;
		} else if (pruning(alpha, dn.getLeftNode())) {
			return true;
		} else if (pruning(alpha, dn.getRightNode())) {
			return true;
		}
		return false;
	}
	
	public void caculateAlpha(DecisionNode node) {
		if (node == null) {
			return;
		}
		DataSet ds = node.getTrainingDataSet();
		if (ds == null || (node.getLeftNode() == null && node.getRightNode() == null)) {
			return;
		}
		double [] target = ds.getIndependentVars();
		double [] ysOfTree = predict(node, ds);
		double [] ysOfNode = new double[ysOfTree.length];
		for (int i = 0; i < ysOfNode.length; i++) {
			ysOfNode[i] = node.getCurrentY();
		}
		double leafCnt = getLeafCnt(node);
		PruningAlpha alpha = new PruningAlpha();
		double treeCost = caculateCost(target, ysOfTree);
		double nodeCost = caculateCost(target, ysOfNode);
		alpha.setAlpha((nodeCost-treeCost)/(leafCnt -1.0));
		alphaList.add(alpha);
		node.setAlpha(alpha.getAlpha());
		caculateAlpha(node.getLeftNode());
		caculateAlpha(node.getRightNode());
	}
	
	public double caculateCost(double [] target, double [] pdV) {
		double sum = 0;
		for (int i = 0; i < pdV.length; i++) {
			double tmp = target[i] - pdV[i];
			sum = sum + tmp * tmp/2.0;
		}
		return sum;
	}
	
	public int getLeafCnt(DecisionNode nd) {
		int cnt = 0;
		if (nd == null) {
			return cnt;
		}
		if (nd.getLeftNode() == null && nd.getRightNode() == null) {
			return 1;
		} 
		cnt = cnt + getLeafCnt(nd.getLeftNode());
		cnt = cnt + getLeafCnt(nd.getRightNode());
		return cnt;
	}
	
	class PruningAlpha implements Serializable {
		double alpha;
		public double getAlpha() {
			return alpha;
		}
		public void setAlpha(double alpha) {
			this.alpha = alpha;
		} 		
	}
	List<PruningAlpha> alphaList = new ArrayList<PruningAlpha>();
	List<DecisionTree> treeList = new ArrayList<DecisionTree>();
	public DecisionTree replicate(DecisionTree dt) {
		DecisionTree dt1 = new DecisionTree();
		dt1.setRoot(replicateNode(null, dt.getRoot()));
		return dt1;
	}
	
	public DecisionNode replicateNode(DecisionNode parent, DecisionNode dn) {
		DecisionNode dn1 = new DecisionNode();
		dn1.setCurrentY(dn.getCurrentY());
		dn1.setDecisionCondition(dn.getDecisionCondition());
		dn1.setIndexOfVars(dn.getIndexOfVars());
		dn1.setParent(parent);
		dn1.setTrainingDataSet(dn.getTrainingDataSet());
		dn1.setAlpha(dn.getAlpha());
		if (dn.getLeftNode() != null) {
			dn1.setLeftNode(replicateNode(dn1, dn.getLeftNode()) );
		}
		if (dn.getRightNode() != null) {
			dn1.setRightNode(replicateNode(dn1, dn.getRightNode()) );
		}
		return dn1;
	}
	
	List<DecisionNode> leafList = new ArrayList<DecisionNode>();
	public void refreshLeafNodes() {
		if (leafList.size() != 0) {
			return ;
		}
		refreshLeafNodes(bestTree.getRoot());
	}
	
	public double [][] generateFeatures(DataSet ds) {
		refreshLeafNodes();
		double [] [] vars = ds.getDependentVars();
		double [] [] features = new double[vars.length][vars[0].length + leafList.size()];
		for (int i = 0; i < vars.length; i++) {
			features[i] = generateFeatures(vars[i]);
		}
		return features;
	}
	
	public double [] generateFeatures(double [] var) {
		refreshLeafNodes();
		DecisionNode dn = getLeaf(bestTree.getRoot() ,var);
		double [] features = new double[var.length + leafList.size()];
		for (int i = 0; i < var.length; i++) {
			features[i] = var[i];
		}
		for (int i = 0; i < leafList.size(); i++) {
			if (dn == leafList.get(i)) {
				features[var.length + i] = 1;
				break;
			}
		}
		return features;
	}
	
	public  DecisionNode getLeaf(DecisionNode node ,double [] var) {
		if (var[node.indexOfVars] < node.decisionCondition ) {
			if (node.getLeftNode()  != null) {
				return getLeaf(node.getLeftNode() , var) ;
			}
		} else {
			if (node.getRightNode() != null) {
				return getLeaf(node.getRightNode() , var) ;
			}
		}
		return node;
	}
	
	public void refreshLeafNodes(DecisionNode node) {
		if (node == null) {
			return;
		}
		if (node.getLeftNode() == null && node.getRightNode() == null) {
			leafList.add(node);
		} else {
			refreshLeafNodes(node.getLeftNode());
			refreshLeafNodes(node.getRightNode());
		}		
	}
	

}
