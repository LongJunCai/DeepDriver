package deepDriver.dl.aml.math;

import java.util.Random;

import deepDriver.dl.aml.random.RandomFactory;

public class LinearExp implements IExp4Function {
	
	double [] parameters;
	double [] deltaPara;
	double [] deltaPara2;
	double [] dv;
	 
	
	protected double min = -1.0;
	protected double max = 1.0;
	protected double length = max - min;
	public static Random random = RandomFactory.getRandom();

	public LinearExp(int length) {
		super();
		parameters = new double[length];
		deltaPara = new double[parameters.length];
		deltaPara2 = new double[parameters.length];
		dv = new double[parameters.length];
		
		/***do we need to comment the xvaier
		double b = Math.pow(6.0/(double)(length + 
				length), 0.5);
		min = -b;
		max = b;
		this.length = max - min;**/
		init();
	}
	
	public void init() {
		for (int i = 0; i < parameters.length; i++) {
			parameters[i] = length * random.nextDouble()
				+ min;
		} 
	}
	
	double [] x;
	double r;
	
	public void compute(double [] x) {
		this.x = x;
		r = MathUtil.multiple(parameters, x); 
	}
	
	public double getR() {
		return r;
	}

	public void setR(double r) {
		this.r = r;
	}

	public void difCompute(double dy, double [] x) {
		difParams(dy, x);
		difValues(dy);
	}
	
	public double [] difParams(double dr) {
		double [] d = new double[deltaPara.length];
		MathUtil.difMultiple(dr, d, x);
		MathUtil.plus2V(d, deltaPara);
		return d;
	}
	
	public double [] difParams(double dr, double [] x) {
		double [] d = new double[deltaPara.length];
		MathUtil.difMultiple(dr, d, x);
		MathUtil.plus2V(d, deltaPara);
		return d;
	}
	
	public double [] difValues(double dr) {		
//		double [] d = new double[dv.length];
		MathUtil.difMultiple(dr, dv, parameters);
//		MathUtil.plus2V(d, dv);
		return dv;
	}

	public double[] getParameters() {
		return parameters;
	}

	public void setParameters(double[] parameters) {
		this.parameters = parameters;
	}

	public double[] getDeltaPara() {
		return deltaPara;
	}

	public void setDeltaPara(double[] deltaPara) {
		this.deltaPara = deltaPara;
	}

	public double[] getDv() {
		return dv;
	}

	public void setDv(double[] dv) {
		this.dv = dv;
	}

	@Override
	public void update(double l, double m) {
		for (int i = 0; i < parameters.length; i++) {
			deltaPara[i] = - l * deltaPara[i] + m * deltaPara2[i];
			parameters[i] = parameters[i] + deltaPara[i];
		}		
		for (int i = 0; i < deltaPara2.length; i++) {
			deltaPara2[i] = deltaPara[i];
		}
		MathUtil.reset2zero(deltaPara);
		MathUtil.reset2zero(dv);
	}

	@Override
	public void resetDv() { 
		MathUtil.reset2zero(dv);
	}
	
}
