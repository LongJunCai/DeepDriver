package deepDriver.dl.aml.cart;

import java.io.Serializable;

public class Gbdt implements Serializable {
	int treeCnt = 10;
	GbdtParameter [] cartParams = new GbdtParameter[treeCnt];
	
	public double [] test(DataSet ds) {
		double [] ys = new double[ds.dependentVars.length];
		for (int i = 0; i < cartParams.length; i++) {
			double [] ys1 = cartParams[i].getCart().predict(ds);
			for (int j = 0; j < ys.length; j++) {
				ys[j] = ys[j] + ys1[j] * cartParams[i].getR();
			}
		}
		return ys;
	}
	
	public double [][] generateFeatures(DataSet trainingDs, DataSet testDs, boolean doesTraining) {
		if (doesTraining) {
			train(trainingDs, testDs);
		}	
		return generateFeatures(trainingDs);
	}
	
	public double [][] generateFeatures(DataSet ds0) {
		DataSet ds = new DataSet();
		ds.setDependentVars(ds0.getDependentVars());
		for (int i = 0; i < cartParams.length; i++) {
			double [][] features = cartParams[i].getCart().generateFeatures(ds);
			ds.setDependentVars(features);
		}
		return ds.getDependentVars();
	}
	
	public double [][] generateFeatures(DataSet trainingDs, DataSet testDs) {
		return generateFeatures(trainingDs, testDs, true);
	}
	
	public void train(DataSet trainingDs, DataSet testDs) {
		double [] camTrainingYs = new double[trainingDs.independentVars.length];
		double [] camTestYs = new double[testDs.independentVars.length];
		
		for (int i = 0; i < cartParams.length; i++) {
			System.out.println("Training the "+(i + 1)+" tree ");
			Cart cart = new Cart();	
			DataSet ds1 = new DataSet();
			ds1.dependentVars = trainingDs.dependentVars;
			ds1.independentVars = new double[trainingDs.independentVars.length];
			ds1.labels = trainingDs.labels;
			for (int j = 0; j < camTrainingYs.length; j++) {
				ds1.independentVars[j] = trainingDs.independentVars[j] - camTrainingYs[j];
			}
			
			cart.trainTree(ds1);
			//
			DataSet ds2 = new DataSet();
			ds2.dependentVars = testDs.dependentVars;
			ds2.independentVars = new double[testDs.independentVars.length];
			ds2.labels = testDs.labels;
			for (int j = 0; j < camTestYs.length; j++) {
				ds2.independentVars[j] = testDs.independentVars[j] - camTestYs[j];
			}
			System.out.println("Optimize the sub-tree.");
			cart.lookupBestTree(ds2);
			//			
			double [] yis = cart.predict(ds1);
			double s1 = 0;
			double s2 = 0;
			double r = 1.0;
			if (i != 0) {
				for (int j = 0; j < yis.length; j++) {
					s1 = s1 + (trainingDs.independentVars[j] - camTrainingYs[j])*yis[j];
					s2 = s2+ yis[j] * yis[j];
				}
				if (s2 == 0) {
					r = 1;
				} else {
					r = s1/s2;
				}
				
			}
			for (int j = 0; j < yis.length; j++) {
				camTrainingYs[j] = camTrainingYs[j] + r * yis[j];
			}
			double [] yis2 = cart.predict(ds2);
			for (int j = 0; j < yis2.length; j++) {
				camTestYs[j] = camTestYs[j] + r * yis2[j];				 
			}
			cartParams[i] = new GbdtParameter();
			cartParams[i].setCart(cart);
			cartParams[i].setR(r);
		}
	}

}
