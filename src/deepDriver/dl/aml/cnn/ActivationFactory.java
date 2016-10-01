package deepDriver.dl.aml.cnn;


import deepDriver.dl.aml.af.SimpleSigmod;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.imp.LogicsticsActivationFunction;
import deepDriver.dl.aml.lstm.imp.TanhAf;

public class ActivationFactory {
	
	IActivationFunction acf = new LogicsticsActivationFunction();
	
	IActivationFunction ssigMod = new SimpleSigmod();

	IActivationFunction flatAcf = new FlatAcf();
	IActivationFunction reLU = new ReLU();
	IActivationFunction tanh = new TanhAf();
	
	static ActivationFactory af = new ActivationFactory();
	public static ActivationFactory getAf() {
		return af;
	}
	public IActivationFunction getAcf() {
		return acf;
	}
	public void setAcf(IActivationFunction acf) {
		this.acf = acf;
	}
	public IActivationFunction getFlatAcf() {
		return flatAcf;
	}
	public void setFlatAcf(IActivationFunction flatAcf) {
		this.flatAcf = flatAcf;
	}
	public IActivationFunction getReLU() {
		return reLU;
	}
	public void setReLU(IActivationFunction reLU) {
		this.reLU = reLU;
	}
	public IActivationFunction getTanh() {
		return tanh;
	}
	public void setTanh(IActivationFunction tanh) {
		this.tanh = tanh;
	}
	public IActivationFunction getSsigMod() {
		return ssigMod;
	}
	public void setSsigMod(IActivationFunction ssigMod) {
		this.ssigMod = ssigMod;
	}		
}
