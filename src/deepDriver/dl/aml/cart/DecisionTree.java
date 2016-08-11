package deepDriver.dl.aml.cart;

import java.io.Serializable;

public class DecisionTree implements Serializable {
	DecisionNode root;

	public DecisionNode getRoot() {
		return root;
	}

	public void setRoot(DecisionNode root) {
		this.root = root;
	}

	
}
