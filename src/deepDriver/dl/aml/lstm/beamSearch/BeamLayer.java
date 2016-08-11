package deepDriver.dl.aml.lstm.beamSearch;

import java.util.ArrayList;
import java.util.List;

public class BeamLayer {
	
	List<BeamNode> bns = new ArrayList<BeamNode>();

	public List<BeamNode> getBns() {
		return bns;
	}

	public void setBns(List<BeamNode> bns) {
		this.bns = bns;
	}
	

}
