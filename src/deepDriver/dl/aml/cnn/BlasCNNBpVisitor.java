package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.math.MathUtil;

public class BlasCNNBpVisitor implements ICNNLayerVisitor {

	protected CNNBP bp;
		
	public BlasCNNBpVisitor(CNNBP bp) {
		super();
		this.bp = bp;
	}

	@Override
	public void visitCNNLayer(final CNNLayer layer) { 
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();
		for (int i = 0; i < fmsInLastLayer.length; i++) { 
			MathUtil.reset2zero(fmsInLastLayer[i].getDeltaZzs());
		}
		for (int i = 0; i < fms.length; i++) {
			if (CNNUtils.useBN(bp.cfg, fms[i])) {
				CNNUtils.batchNorm(fms[i]);
			} else {
				CNNUtils.deActiveDZzs(fms[i]);
			}
			CNNUtils.deActivateGlobal(bp, fms[i]);
		}
		
		// copy back2 fms;
		int[][][] oIds = layer.getOutIds();
		final float[][] dom = layer.getDoutM();
		for (int i = 0; i < dom.length; i++) {
			for (int j = 0; j < dom[i].length; j++) {
				double[][] pfs = fms[j].getDeltaZzs();
				dom[i][j] = (float) pfs[oIds[i][j][0]][oIds[i][j][1]];
			}
		}
		final float[][] ckm = layer.getCkM();
		float[][] preM = layer.getPreM();
		
		long t = System.currentTimeMillis();
		//X * Y = Z//		
//		Thread th = new Thread() { 
//			public void run() { 
//				MathUtil.difMultipleX(dom, ckm, layer.getDpreM());
//			}			
//		};
//		th.start();
		MathUtil.difMultipleX(dom, ckm, layer.getDpreM());
		float [][] dckm = MathUtil.difMultipleY(dom, preM, layer.getDckM4Tmp());
//		try {
//			th.join();
//		} catch (InterruptedException e) { 
//			e.printStackTrace();
//		}
		float [][] dpreM = layer.getDpreM();
//		System.out.println("bpMo.." +(System.currentTimeMillis() - t));
		
		int [][][] preMIds = layer.getPreIds();
		for (int i = 0; i < dpreM.length; i++) {
			for (int j = 0; j < dpreM[i].length; j++) {
				int [] pt = preMIds[i][j];	 
				int r = pt[0];
				int c = pt[1];
				int pos = pt[2];
				if (r < 0 || c < 0) {
//					preM[i][j] = 0;
				} else {					 
					fmsInLastLayer[pos].getDeltaZzs()[r][c] = fmsInLastLayer[pos].getDeltaZzs()[r][c] + dpreM[i][j];	
				}						
			}
		}
		//we did not handle the non-shared global b cases.
		MathUtil.plus(layer.getDckM(), (float)bp.cfg.getM(), dckm, (float)-bp.cfg.getL(), layer.getDckM());
		
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {		 
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();
		for (int i = 0; i < fmsInLastLayer.length; i++) { 
			MathUtil.reset2zero(fmsInLastLayer[i].getDeltaZzs());
		}
		for (int i = 0; i < fms.length; i++) {
			CNNUtils.deActiveDZzs(fms[i]);
			CNNUtils.deActivateGlobal(bp, fms[i]);
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				SubSamplingKernal ssk = (SubSamplingKernal)cks[j];
				ssk.initwW = false;
				ssk.initB = false;
			}
		}
		
		// copy back2 fms;
		int[][][] oIds = layer.getOutIds();
		float[][] dom = layer.getDoutM();
		for (int i = 0; i < dom.length; i++) {
			for (int j = 0; j < dom[i].length; j++) {
				double[][] pfs = fms[j].getDeltaZzs();
				dom[i][j] = (float) pfs[oIds[i][j][0]][oIds[i][j][1]];
			}
		}
		float [][] ckm = layer.getCkM();
		float [][] preM = layer.getPreM();
		//X * Y = Z
		float[][] dpreM = MathUtil.difMultipleX(dom, ckm, layer.getDpreM());
		float[][] dckm = MathUtil.difMultipleY(dom, preM, layer.getDckM4Tmp());
		
		int [][][] preMIds = layer.getPreIds();
		for (int i = 0; i < dpreM.length; i++) {
			for (int j = 0; j < dpreM[i].length; j++) {
				int [] pt = preMIds[i][j];	 
				int r = pt[0];
				int c = pt[1];
				int pos = pt[2];
				if (r < 0 || c < 0) {
//					preM[i][j] = 0;
				} else {					 
					fmsInLastLayer[pos].getDeltaZzs()[r][c] = fmsInLastLayer[pos].getDeltaZzs()[r][c] + dpreM[i][j];	
				}						
			}
		}
		// copy ckm
		int[][][] ckIds = layer.getCkIds();
		for (int i = 0; i < dckm.length; i++) {
			for (int j = 0; j < dckm[i].length; j++) {
				int pt = ckIds[i][j][0];
				if (pt < 0) {
					continue;
				}
				SubSamplingKernal ck = (SubSamplingKernal) fms[j].getKernals()[pt];
				if (!ck.initwW) {
					ck.initwW = true;
					ck.deltawW = bp.cfg.getM() * ck.deltawW 
							- bp.cfg.getL() * dckm[i][j];
				} else {
					ck.deltawW = ck.deltawW 
							- bp.cfg.getL() * dckm[i][j];
				}
//				ck.deltawW = //dckm[i][j];
			}
		}
		//we did not handle the non-shared global b cases.
//		MathUtil.plus(layer.getDckM(), bp.cfg.getM(), dckm, -bp.cfg.getL(), layer.getDckM());
		
	
	}

	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) { 

	}

}
