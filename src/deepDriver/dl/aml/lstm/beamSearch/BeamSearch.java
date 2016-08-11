package deepDriver.dl.aml.lstm.beamSearch;

import java.util.ArrayList;
import java.util.List;

import deepDriver.dl.aml.lstm.PosValue;

public class BeamSearch {
	
	int size = 5;	
	
	List<BeamLayer> layers = new ArrayList<BeamLayer>();	

	public int getSize() {
		return size;
	}

	public void setSize(int size) {
		this.size = size;
	}

	public List<BeamLayer> getLayers() {
		return layers;
	}

	public void setLayers(List<BeamLayer> layers) {
		this.layers = layers;
	}
	
	public void createBn(PosValue [] pvs, BeamNode parent) {
		if (layers.size() == 0 || parent == null) {
			layers.add(new BeamLayer());
		} else if (layers.size() > 0 && layers.get(layers.size() - 1) == parent.getBl()) {
			layers.add(new BeamLayer());
		}
		BeamLayer bl = layers.get(layers.size() - 1);
		double ppb = 10.0;
		if (parent != null) {
			ppb = parent.getProb() * 10.0;
		}
		for (int i = 0; i < pvs.length; i++) {
			BeamNode nbn = new BeamNode(parent, bl, ppb * pvs[i].getValue(), pvs[i].getPos()); 
//			parent.getChildren().add(nbn);
			bl.getBns().add(nbn);
		}
	}
	
	public int getLLayerSize() {
		return layers.size();
	}
	
	public double [][] getBnById(int index, BeamLayer bl) {
		int offset = 1;
		if (layers.get(layers.size() - 1) == bl) {
			offset = 0;
		}
		List<BeamNode> bns = bl.getBns();
		//assume use thin data always
		double [][] v = new double[layers.size() - offset][1];
		BeamNode bn = bns.get(index);
		for (int i = v.length - 1; i >= 0; i-- ) {
			v[i] = new double[1];
			v[i][0] = bn.getPos();
			bn = bn.getParent();
		}
		return v;
	}
	
	public void sort(int [] si, List<BeamNode> bns) {
		for (int i = 0; i < si.length; i++) {
			for (int j = i + 1; j < si.length; j++) {
				BeamNode bi = bns.get(si[i]);
				BeamNode bj = bns.get(si[j]);
				if (bj.getProb() < bi.getProb()) {
					int t = si[i];
					si[i] = si[j];
					si[j] = t;
				}
			}
		}
	}
	
	public void sortAndPrunchByLastLayer() {
		if (layers.size() == 0) {
			return;
		}
		BeamLayer bl = layers.get(layers.size() - 1);
		//sort
		int [] si = new int[size];
		for (int i = 0; i < si.length; i++) {
			si[i] = i;
		}
		List<BeamNode> bns = bl.getBns();
		
		sort(si, bns);		
		for (int i = si.length; i < bns.size(); i++) {
			BeamNode bn = bns.get(i);
			double prob = bns.get(si[0]).getProb();
			if (bn.getProb() > prob) {
				si[0] = i;
				sort(si, bns);
			}
		}		
		//prunch
		BeamNode [] ba = new BeamNode[si.length];
		for (int i = 0; i < ba.length; i++) {
			ba[i] = bns.get(si[i]);
		}
		bns.clear();
		for (int i = 0; i < ba.length; i++) {
			bns.add(ba[i]);
		}
	}
	
	

}
