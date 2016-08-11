package deepDriver.dl.aml.ann.test;

import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.InputParameters;

public class ANNAreaSplittingTest1 {
	
	public static int test(double x, double y) {
		if (y - x > 0) {
			return 1;
		}
		return 0;
	}
	
	public static int test2(double x, double y) {
		if ((1.8 * 1.8 - x* x - y * y) <0) {
			return 1;
		}
		return 0;
	}
	public static void main2(String[] args) {
		System.out.println(test(-1.4, 0));
	}
	public static void main(String[] args) {
		ArtifactNeuroNetwork  ann = new ArtifactNeuroNetwork();
		double [][] inputs = new double[2000][2];
		double [] y = new double[inputs.length];
		double xStep = 0.1;
		double r = 0.3;
		double rStep = 0.2;
		int cnt = 0;
		System.out.println("Preparing for training data: ");
		while (cnt < y.length - 1) {
			double x = -r;
			for (int i = 0; i < y.length; i++) {
				double yi = Math.sqrt(r * r - x * x);				
				inputs[cnt][0] = x;
				inputs[cnt][1] = yi;
				y[cnt++] = test(x, yi);
//				System.out.println(x +","+yi+","+test(x, yi));
				if (cnt >= y.length - 1) {
					break;
				}
				inputs[cnt][0] = x;
				inputs[cnt][1] = -yi;
				y[cnt++] = test(x, -yi);
//				System.out.println(x +","+(-yi)+","+test(x, -yi));
				if (cnt >= y.length - 1) {
					break;
				}
				x = x + xStep;
				if (x >= r) {
					break;
				}
			}
			r = r + rStep;
		}
		InputParameters parameters = new InputParameters();
		parameters.setInput(inputs);
		parameters.setResult(y);
		parameters.setLayerNum(3);
		ann.trainModel(parameters);
		System.out.println("test model: ");
//		parameters.setInput(new double [][]{
//				{-1.4, 0}, {0.1, 0.8},{0,7},{9,7}
//		});
		double [] ys = ann.testModel(parameters);
		int invalidCnt = 0;
		double [][] ts = parameters.getInput();
		for (int i = 0; i < ys.length; i++) {
			double pi = (ys[i]>0.5? 1: 0);
			double yi = test(ts[i][0], ts[i][1]);
			if (yi != pi) {
				invalidCnt ++;
			}
//			System.out.println(ts[i][0]+","+ts[i][1]+","+yi
//					+","+pi+","+ys[i]);
		}
		System.out.println("The number of failture is: "+invalidCnt
				+" and the accurency is: "+(1.0 - (double)invalidCnt/(double)ys.length));
	}

}
