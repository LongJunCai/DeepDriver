package deepDriver.dl.aml.lrate;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class BoldDriverLearningRateManager implements LearningRateManager, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double increaseRate = 1.05;
	double decreaseRate = 2.0/3.0;
	
	double errSize = 9;
	double flatThreshold = 0.004;
	
	List<Double> errs = new ArrayList<Double>();
	@Override
	public double adjustML(double err, double lrate) {
		if (errs.size() >= errSize) {
			errs.remove(0);
		}
		errs.add(err);
		if (errs.size() < errSize) {
			return lrate;
		}
		int pos = (int)(errSize / 2);
		double h0 = 0;
		double h1 = 0;
		for (int i = 0; i <= pos; i++) {
			h0 = h0 + errs.get(i);
		}
		h0 = h0/(double)(pos + 1);
		for (int i = pos; i < errs.size(); i++) {
			h1 = h1 + errs.get(i);
		}
		h1 = h1/(double)(pos + 1);
		if ((h0-h1)/h0 < flatThreshold) {
			lrate = lrate * decreaseRate;
			errs.clear();//no need to consider to decrease the rate for now
		}
		return lrate;
	}
	
	
	
	public double getDecreaseRate() {
		return decreaseRate;
	}



	public void setDecreaseRate(double decreaseRate) {
		this.decreaseRate = decreaseRate;
	}



	public double getErrSize() {
		return errSize;
	}



	public void setErrSize(double errSize) {
		this.errSize = errSize;
	}



	public double getFlatThreshold() {
		return flatThreshold;
	}



	public void setFlatThreshold(double flatThreshold) {
		this.flatThreshold = flatThreshold;
	}



	public static void main(String[] args) {
		double [] errs = {
				51.64363905,46.01572003,44.93469057,44.33275724,44.09911764,43.72607568,43.64642804,43.36474343,43.19501014,42.73952276,42.62139013,42.39641003,39.98051315,41.71429449,41.55983094,41.21811982,40.76850352,40.28876929,39.82513055,39.42032076,38.95726533,38.59601852,37.88821761,37.52106364,37.03269067,36.45697954,35.97959889,35.70694063,35.25369239,34.76584846,34.40229415,33.91516742,33.50239313,32.51684786,32.66210924,32.302534,31.76048097,31.38168072,30.82803434,30.58258911,30.3721781,29.96230986,29.63930382,29.28193522,28.08825694,28.36060964,28.04507809,27.7223476,27.47852037,27.10792243,26.82234044,26.55088067,26.43098343,26.05403139,25.58557526,24.97982444,25.06905505,24.63437206,23.83069806,23.53183778,23.36825327,23.34183498,23.0187586,22.70740289,22.51840864,21.80028669,22.17996576,21.78255676,21.38505491,21.08405526,20.96731652,20.81542762,20.79821318,20.63377551,20.44698437,20.29426847,20.24229765,20.08982449,19.98289853,19.75672275,19.69756311,19.64920158,19.59056593,19.5395936,19.44029642,19.30600539,19.12342935,19.15020973,18.86529246,18.27840178,18.10790878,18.02367857,17.83527891,17.75116549,17.71782082,17.62750831,17.63899528,17.47346993,16.57811945,16.9915132,16.79174879,16.64198195,16.69763386,16.60756479,16.60387264,16.54037465,16.43754244,16.45988984,16.35909125,16.33814695,16.26017503,16.1948574,16.26848186,16.17662073,16.13002164,16.1426888,16.15118885,16.08394581,16.06222355,16.01937959,16.02622608,16.01354869,16.0159257,15.99834812,15.95332576,15.43647214,15.97383149,16.05390141,16.05155036,15.96346746,15.9448814,15.93932388,15.99796993,15.95727521,16.01464223,15.99182952,15.96904702,16.01890457,16.00219646,16.09268443,15.9708895,15.94214141,15.99445352,15.97411912,15.992376,15.91711284
		};
		double lrate = 0.001;
		BoldDriverLearningRateManager lrManager = new BoldDriverLearningRateManager();
		for (int i = 0; i < errs.length; i++) {
			lrate = lrManager.adjustML(errs[i], lrate);
			System.out.println(errs[i] + ","+lrate);
		}
		
	}

}
