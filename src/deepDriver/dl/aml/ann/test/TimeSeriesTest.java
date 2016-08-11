package deepDriver.dl.aml.ann.test;

import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.ann.Normalizer;

public class TimeSeriesTest {
	public static void main(String[] args) {
		ArtifactNeuroNetwork  ann = new ArtifactNeuroNetwork();
		Normalizer normalizer = new Normalizer();
		double [] ts = new double[] {
//				40,43.545,47.214,50.518,52.959,54.102,53.639,51.449,47.621,42.47,36.506,30.392,24.864,20.646,18.356
				40,43.545,47.214,50.518,52.959,54.102,53.639,51.449,47.621,42.47,36.506,30.392,24.864,20.646,18.356,18.424,21.022,26.027,33.015,41.295,49.977,58.067,64.579,68.656,69.681,67.358,61.772,53.394,43.044,31.816,20.959,11.741,5.299,2.5,3.833,9.327,18.537,30.57,44.172,57.857,70.071,79.368,84.577,84.954,80.276,70.889,57.687,42.03,25.605,10.242	
		};
		double [] [] inputs = new double[ts.length][1];
//		double max = 0;
		for (int i = 0; i < inputs.length; i++) {
			inputs[i][0] = i + 1;
		}
		double [] tts = normalizer.
				transformParameters(ts);
		InputParameters parameters = new InputParameters();
		parameters.setInput(inputs);
		parameters.setResult(tts);
		parameters.setAlpha(0.1);//0.1 best
		parameters.setIterationNum(1000000);
		parameters.setNeuros(new int [] {
//				1,5, 1//best
				1,15, 1
		});
		ann.trainModel(parameters);
		System.out.println("test model: ");
		double [][] testData = new double[20][1];
		for (int i = 0; i < testData.length; i++) {
			testData[i][0] = 40+i ;
		}
//		parameters.setInput(testData);
		double [] ys = ann.testModel(parameters);
		int invalidCnt = 0;
//		double [][] ts = parameters.getInput();
		double [] tys = normalizer.transformBackParameters(
				 ys);
		for (int i = 0; i < tys.length; i++) {
			System.out.println(tys[i]);
		}
		System.out.println("The number of failture is: "+invalidCnt
				+" and the accurency is: "+(1.0 - (double)invalidCnt/(double)ys.length));
	}

}
