package deepDriver.dl.aml.lstm.beamSearch;

import java.util.ArrayList;
import java.util.List;

public class BeamNode {
	BeamNode parent;
//	List<BeamNode> children = new ArrayList<BeamNode>();
	BeamLayer bl;
	
	double prob;
	double pos;
	
	public BeamNode(BeamNode parent, BeamLayer bl, double prob, double pos) {
		super();
		this.parent = parent;
		this.bl = bl;
		this.prob = prob;
		this.pos = pos;
	}
	public BeamLayer getBl() {
		return bl;
	}
	public void setBl(BeamLayer bl) {
		this.bl = bl;
	}
	public BeamNode getParent() {
		return parent;
	}
	public void setParent(BeamNode parent) {
		this.parent = parent;
	}
//	public List<BeamNode> getChildren() {
//		return children;
//	}
//	public void setChildren(List<BeamNode> children) {
//		this.children = children;
//	}
	public double getProb() {
		return prob;
	}
	public void setProb(double prob) {
		this.prob = prob;
	}	
	
	public double getPos() {
		return pos;
	}
	public void setPos(double pos) {
		this.pos = pos;
	}
	
}
