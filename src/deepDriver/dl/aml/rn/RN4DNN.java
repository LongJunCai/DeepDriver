package deepDriver.dl.aml.rn;

import java.io.Serializable;
import java.util.List;

import deepDriver.dl.aml.ann.ArtifactNeuroNetworkV2;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.ann.imp.NeuroUnitImpV3;
import deepDriver.dl.aml.costFunction.ICostFunction;
import deepDriver.dl.aml.dnn.DNN;
import deepDriver.dl.aml.math.MathUtil;

public class RN4DNN implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	ArtifactNeuroNetworkV2 gf;
	DNN ff;
	
	ICostFunction cf;
	public ICostFunction getCf() {
		return cf;
	}

	public void setCf(ICostFunction cf) {
		this.cf = cf;
	}
	
	public void trainModel(InputParameters gfPs, InputParameters ffPs, RelationObjectSet [] ross) {
		double [][] inputs = new double[ross.length][];
//		double [][] targets = new double[ross.length][];
		
		double [] i0 = ross[0].gettObj().getInput();
		double [] i1 = ross[0].getrObjs()[0].getInput();
		double [] i2 = MathUtil.mergeVector(i0, i1);
		buildUpGf(gfPs, new double[][]{i2});
		
		System.out.println("gfPs.getAlpha():"+gfPs.getAlpha());
		System.out.println("gfPs.getLamda():"+gfPs.getLamda());
		System.out.println("gfPs.getM():"+gfPs.getM());
		gfPs.setM(0.3);
		
		System.out.println("ffPs.getAlpha():"+ffPs.getAlpha());
		System.out.println("ffPs.getLamda():"+ffPs.getLamda());
		System.out.println("ffPs.getM():"+ffPs.getM());
		
		for (int i = 0; i < ross.length; i++) {
//			RelationObject tObj = ross[i].gettObj();
//			double [] target = tObj.getTarget();
			double [] input = forwardGf(ross[i]);
			inputs[i] = input;
//			targets[i] = target;
		}
		ffPs.setInput(inputs);
//		if (ffPs.getResult2() == null) {
//			ffPs.setResult2(targets);
//		}		
		
		ff = new DNN();
		ff.setCf(cf);
		ff.setNormalize(false);
		ff.learnSelf(ffPs);
		
		double error = 0;
		int errorCnt = 0;
		int loop = ffPs.getIterationNum();
		for (int i = 0; i < loop; i++) {
			error = 0;
			for (int j = 0; j < ross.length; j++) {
				error = error + fineTuneSingle(ross[j], j, ffPs, gfPs);
			}
			//2. optimize the DNN over all
			errorCnt++;
			if (errorCnt % 1 == 0) {
				info(errorCnt+" Fine tuning RNDNNMTL error ="+error);
			}			
			if (error < acc) {
				System.out.println("Training RNDNNMTL is stopped early.");
				break;				
			}
		}
		
	}
	
	public double getAcc() {
		return acc;
	}

	public void setAcc(double acc) {
		this.acc = acc;
	}
	double acc = 0.1;
	
	
	private void info(String string) {
		System.out.println(string);		
	}

	public void name() {
		
	}



	public double fineTuneSingle(RelationObjectSet ros, int i, InputParameters ffps, InputParameters gfPs) {
//		RelationObject tObj = ros.gettObj();
		double [] input = forwardGf(ros);
		ffps.setBpFirstLayer(true);
		
//		double error = ff.fdBp(new double[][]{input}, i, new double[][]{ffps.getResult2()[i]}, ffps);
		double error = ff.runEpoch(input, i,  ffps.getResult2()[i] , ffps);
//		double error = ff.runEpoch(ffps.getInput()[i], i,  ffps.getResult2()[i] , ffps);

		/****/
		int dzZIndex = 0;
		ILayer first = ff.getFirstLayer();
		List<INeuroUnit> ffnus = first.getNeuros();
		double [] dzZs = new double[ffnus.size()];
		for (int j = 0; j < dzZs.length; j++) {
			NeuroUnitImpV3 v3 = (NeuroUnitImpV3) ffnus.get(j);
			dzZs[j] = v3.getDeltaZ()[dzZIndex];
		}
		
		ILayer lastLayer = getLastLayer(gf);
		List<INeuroUnit> gfNu = lastLayer.getNeuros();
		for (int j = 0; j < dzZs.length; j++) {
			NeuroUnitImpV3 v3 = (NeuroUnitImpV3) gfNu.get(j);
			v3.setParameters(gfPs);
			double [] dZ = v3.getDeltaZ();
			double l = dZ.length;//because it is used during deliver the aAs.
			for (int k = 0; k < dZ.length; k++) {
				dZ[k] = dzZs[j]/l * v3.getActivationFunction().deActivate(v3.getZzs()[k]);
			}
		}
		
		while (lastLayer.getPreviousLayer() != null) {
			lastLayer = lastLayer.getPreviousLayer();
			lastLayer.backPropagation(new double[][]{new double[gfNu.size()]}, gfPs);
		}
		
		ILayer layer = gf.getFirstLayer();
		while (layer.getNextLayer() != null) {
			layer.getNextLayer().updateNeuros();
			layer = layer.getNextLayer();
		}
//		ff.updateNk();
		return error;
	}
	
	public ILayer getLastLayer(ArtifactNeuroNetworkV2 v2) {
		ILayer layer = v2.getFirstLayer();
		while (layer.getNextLayer() != null) {
			layer = layer.getNextLayer();
		}
		return layer;
	}
	
	public void buildUpGf(InputParameters gfPs, double [][] input) {
		gf = new ArtifactNeuroNetworkV2();
		IActivationFunction acf = gf.createAcf();
		gf.setFirstLayer(gf.createLayer());
		
		gf.getFirstLayer().buildup(null, input, acf, false, input[0].length);
		ILayer tlayer = gf.getFirstLayer();
		for (int i = 0; i < gfPs.getLayerNum(); i++) {
			ILayer newLayer = gf.createLayer();
			int neuroCnt = input[0].length;
			if (gfPs.getNeuros() != null) {
				neuroCnt = gfPs.getNeuros()[i];
			}
			newLayer.setPos(i+1);
			newLayer.buildup(tlayer, input, acf, 
					i == gfPs.getLayerNum() - 1, neuroCnt);
			tlayer = newLayer;
		}
		
	}
	

	public double [] forwardGf(RelationObjectSet ros) {
		RelationObject [] objs = ros.getrObjs();
		RelationObject tObj = ros.gettObj(); 
		double [] aAs = null;
		double [][] input = new double[objs.length][];
		for (int i = 0; i < objs.length; i++) {
			input[i] = MathUtil.mergeVector(tObj.getInput(), objs[i].getInput());
		}
		
		ILayer layer = gf.getFirstLayer();
		ILayer lastLayer = layer;
		while (layer != null) {
			layer.forwardPropagation(input);
			lastLayer = layer;
			layer = layer.getNextLayer();
		}
		
		List<INeuroUnit> list = lastLayer.getNeuros();
		if (aAs == null) {
			aAs = new double[list.size()];
		}
		double l = input.length;
		for (int j = 0; j < aAs.length; j++) {
			for (int i = 0; i < input.length; i++) {
				aAs[j] = aAs[j] + list.get(j).getAaz(i)/l;
			}			
		}
		return aAs;
	}
	 
	
	
	public double [][] testModel2(RelationObjectSet [] ross) {
		InputParameters tmp = new InputParameters();
		double [][] rt = new double[ross.length][];
		for (int i = 0; i < ross.length; i++) {
			double [] input = forwardGf(ross[i]);
			tmp.setInput(new double[][]{input});
			rt[i] = ff.testModel2(tmp)[0];
		}
		return rt;
	}

}
