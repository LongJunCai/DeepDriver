package deepDriver.dl.aml.cart;

import java.io.Serializable;

public class DecisionNode implements Serializable {
	int indexOfVars;
	double decisionCondition;
	double currentY;
//	double leftY;
//	double rightY;
	
	DecisionNode parent;
	
	DecisionNode leftNode;
	DecisionNode rightNode;		
	
	DataSet trainingDataSet;
	
	double alpha = -1;	
		
	public double getAlpha() {
		return alpha;
	}
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	public DataSet getTrainingDataSet() {
		return trainingDataSet;
	}
	public void setTrainingDataSet(DataSet trainingDataSet) {
		this.trainingDataSet = trainingDataSet;
	}
	public double getCurrentY() {
		return currentY;
	}
	public void setCurrentY(double currentY) {
		this.currentY = currentY;
	}
	public DecisionNode getParent() {
		return parent;
	}
	public void setParent(DecisionNode parent) {
		this.parent = parent;
	}
//	public double getLeftY() {
//		return leftY;
//	}
//	public void setLeftY(double leftY) {
//		this.leftY = leftY;
//	}
//	public double getRightY() {
//		return rightY;
//	}
//	public void setRightY(double rightY) {
//		this.rightY = rightY;
//	}
	public int getIndexOfVars() {
		return indexOfVars;
	}
	public void setIndexOfVars(int indexOfVars) {
		this.indexOfVars = indexOfVars;
	}
	public double getDecisionCondition() {
		return decisionCondition;
	}
	public void setDecisionCondition(double decisionCondition) {
		this.decisionCondition = decisionCondition;
	}
	public DecisionNode getLeftNode() {
		return leftNode;
	}
	public void setLeftNode(DecisionNode leftNode) {
		this.leftNode = leftNode;
	}
	public DecisionNode getRightNode() {
		return rightNode;
	}
	public void setRightNode(DecisionNode rightNode) {
		this.rightNode = rightNode;
	}	
	
}
