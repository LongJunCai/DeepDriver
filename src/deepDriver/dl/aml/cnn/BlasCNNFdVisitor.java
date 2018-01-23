package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.math.MathUtil;

public class BlasCNNFdVisitor implements ICNNLayerVisitor {
	
	protected CNNBP bp;	
	
	public BlasCNNFdVisitor(CNNBP bp) {
		super();
		this.bp = bp;
	}

	//assume the ck is fixed size
	@Override
	public void visitCNNLayer(CNNLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();	
		double [][] fll = fmsInLastLayer[0].getFeatures();
		ConvolutionKernal cks = (ConvolutionKernal) fms[0].getKernals()[0];
		int padding = 2 * layer.getPreviousLayer().getLc().getPadding();
		//asume step = 1, and no need padding.
//		int r = padding + featureOfPrevious.length - ckRows + 1;
//		int c = padding + featureOfPrevious[0].length - ckColumns + 1;
		int ckw = cks.wWs.length;
		int ckh = cks.wWs[0].length;
		int psize = 3;
		if (layer.getPreM() == null) {					
			int rows = (padding + fll.length - ckw + 1) * (padding + fll[0].length - ckh + 1);
			int cols = ckw * ckh * fmsInLastLayer.length;
			//assume all the previous fms are the same size
			layer.setPreM(MathUtil.allocate(rows, cols));
			layer.setDpreM(MathUtil.allocate(rows, cols));
			layer.setPreIds(MathUtil.allocateInt(rows, cols, psize));
			//assume all the ck are the same size.
			layer.setCkM(MathUtil.allocate(ckw * ckh * fmsInLastLayer.length, fms.length));
			layer.setDckM(MathUtil.allocate(ckw * ckh * fmsInLastLayer.length, fms.length));
			//assume all the fms are the same size.
			double [][] ff = fms[0].getFeatures();
//			layer.setOutM(MathUtil.allocate(ff.length * ff[0].length, fms.length));
			layer.setDoutM(MathUtil.allocate(ff.length * ff[0].length, fms.length));
			layer.setOutIds(MathUtil.allocateInt(ff.length * ff[0].length, fms.length, psize));
			//copy CKs
			double [][] ckm = layer.getCkM();
			int ick = 0;
			for (int i = 0; i < fms.length; i++) {
				IConvolutionKernal [] cks1 = fms[i].getKernals();
				ick = 0;
				for (int j = 0; j < cks1.length; j++) {
					ConvolutionKernal ck = (ConvolutionKernal)cks1[j];
					double [][] wWs = ck.wWs;
					for (int k = 0; k < wWs.length; k++) {
						for (int k2 = 0; k2 < wWs.length; k2++) {
							ckm[ick++][i] = wWs[k][k2];//assuming the fully ck is sorted.
						}
					}
				}
			}
			
			
			//copy preM indexes.
			int [][][] preMIds = layer.getPreIds();
			for (int i = 0; i < fmsInLastLayer.length; i++) {
				double [][] pfs = fmsInLastLayer[i].getFeatures();
				int pd = layer.getPreviousLayer().getLc().getPadding();
				int r = -1;
				for (int j = -pd; j < pfs.length - ckw + 1 + pd; j++) {					
					for (int j2 = -pd; j2 < pfs[j].length - ckh + 1 + pd; j2++) {	
						r++;
						int c = 0;
						for (int k = 0; k < ckw; k++) {
							for (int k2 = 0; k2 < ckh; k2++) {
								int nr = j + k;
								int nc = j2 + k2;
								if (nr < 0 || nr > pfs.length - 1) {
									nr = -1; 
									nc = -1;
								} else if (nc < 0 || nc > pfs[j].length - 1) {
									nr = -1; 
									nc = -1;
								} 
								preMIds[r][c][0] = nr;		
								preMIds[r][c][1] = nc;
								preMIds[r][c][2] = i;
								c++;
							}
						}
					}
				}
			}
			
			//copy outM indexes.
			int [][][] oIds = layer.getOutIds();
			for (int i = 0; i < fms.length; i++) {
				double [][] pfs = fms[i].getFeatures();
				int r = 0;
				for (int j = 0; j < pfs.length; j++) {
					for (int j2 = 0; j2 < pfs[j].length; j2++) {
						oIds[r][i][0] = j;
						oIds[r][i][1] = j2;
						r++;
					}
				}
			}
		}
		//copy preM
		double [][] preM = layer.getPreM();
		int [][][] preMIds = layer.getPreIds();
		for (int i = 0; i < preM.length; i++) {
			for (int j = 0; j < preM[i].length; j++) {
				int [] pt = preMIds[i][j];	 
				int r = pt[0];
				int c = pt[1];
				if (r < 0 || c < 0) {
					preM[i][j] = 0;
				} else {
					preM[i][j] = fmsInLastLayer[pt[2]].getFeatures()[r][c];		
				}						
			}
		}
				
		//caculate outM 
		double [][] ckm = layer.getCkM();
		layer.setOutM(MathUtil.multiple(preM, ckm));
		
		//copy back2 fms;
		int [][][] oIds = layer.getOutIds();
		double [][] outM = layer.getOutM();
		for (int i = 0; i < outM.length; i++) {
			for (int j = 0; j < outM[i].length; j++) {
				double [][] pfs = fms[j].getzZs();
				pfs[oIds[i][j][0]][oIds[i][j][1]] = outM[i][j];
			}
		}
		
		for (int i = 0; i < fms.length; i++) {
			CNNUtils.activateConvZzs(bp, bp.cfg, fms[i]);
		}
		
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();	
		double [][] fll = fmsInLastLayer[0].getFeatures();
		SubSamplingKernal cks = (SubSamplingKernal) fms[0].getKernals()[0];
		int padding = 2 * layer.getPreviousLayer().getLc().getPadding(); 
		int ckw = cks.ckRows;
		int ckh = cks.ckColumns;
		int psize = 3;
		if (layer.getPreM() == null) {					
			//assume it is well designed, the length is same size as chw and ckh
			int rows = ((padding + fll.length)/ckw) * ((padding + fll[0].length)/ckh);
			int cols = ckw * ckh * fmsInLastLayer.length;
			//assume all the previous fms are the same size
			layer.setPreM(MathUtil.allocate(rows, cols));
			layer.setDpreM(MathUtil.allocate(rows, cols));
			layer.setPreIds(MathUtil.allocateInt(rows, cols, psize));
			//assume all the ck are the same size.
			layer.setCkM(MathUtil.allocate(ckw * ckh * fmsInLastLayer.length, fms.length));
			layer.setDckM(MathUtil.allocate(ckw * ckh * fmsInLastLayer.length, fms.length));
			layer.setCkIds(MathUtil.allocateInt(ckw * ckh * fmsInLastLayer.length, fms.length, psize));
			//assume all the fms are the same size.
			double [][] ff = fms[0].getFeatures();
//			layer.setOutM(MathUtil.allocate(ff.length * ff[0].length, fms.length));
			layer.setDoutM(MathUtil.allocate(ff.length * ff[0].length, fms.length));
			layer.setOutIds(MathUtil.allocateInt(ff.length * ff[0].length, fms.length, psize));
			//copy CKs			
			int [][][] ckIds = layer.getCkIds();
			int ick = 0;
			for (int i = 0; i < fms.length; i++) {
				IConvolutionKernal [] cks1 = fms[i].getKernals();
				ick = 0;
				for (int j = 0; j < cks1.length; j++) {
					SubSamplingKernal ck = (SubSamplingKernal)cks1[j];
					double wW = ck.wW;
					for (int k = 0; k < ckw; k++) {
						for (int k2 = 0; k2 < ckh; k2++) {
							ckIds[ick++][i][0] = j;//assuming the fully ck is sorted.
						}
					}
				}
			}
			
			
			//copy preM indexes.
			int [][][] preMIds = layer.getPreIds();
			for (int i = 0; i < fmsInLastLayer.length; i++) {
				double [][] pfs = fmsInLastLayer[i].getFeatures();
				int pd = 2 * layer.getPreviousLayer().getLc().getPadding();
				int r = -1;
				for (int j = 0; j < (pfs.length + pd)/ckw; j++) {					
					for (int j2 = 0; j2 < (pfs[j].length+ pd)/ckh; j2++) {	
						r++;
						int c = 0;
						for (int k = 0; k < ckw; k++) {
							for (int k2 = 0; k2 < ckh; k2++) {
								int nr = j * ckw + k;
								int nc = j2 * ckh + k2;
								if (nr < 0 || nr > pfs.length - 1) {
									nr = -1; 
									nc = -1;
								} else if (nc < 0 || nc > pfs[j].length - 1) {
									nr = -1; 
									nc = -1;
								} 
								preMIds[r][c][0] = nr;		
								preMIds[r][c][1] = nc;
								preMIds[r][c][2] = i;
								c++;
							}
						}
					}
				}
			}
			
			//copy outM indexes.
			int [][][] oIds = layer.getOutIds();
			for (int i = 0; i < fms.length; i++) {
				double [][] pfs = fms[i].getFeatures();
				int r = 0;
				for (int j = 0; j < pfs.length; j++) {
					for (int j2 = 0; j2 < pfs[j].length; j2++) {
						oIds[r][i][0] = j;
						oIds[r][i][1] = j2;
						r++;
					}
				}
			}
		}
		//copy preM
		double [][] preM = layer.getPreM();
		int [][][] preMIds = layer.getPreIds();
		for (int i = 0; i < preM.length; i++) {
			for (int j = 0; j < preM[i].length; j++) {
				int [] pt = preMIds[i][j];	 
				int r = pt[0];
				int c = pt[1];
				if (r < 0 || c < 0) {
					preM[i][j] = 0;
				} else {
					preM[i][j] = fmsInLastLayer[pt[2]].getFeatures()[r][c];		
				}						
			}
		}
		//copy ckm
		double [][] ckm = layer.getCkM();
		int [][][] ckIds = layer.getCkIds();
		for (int i = 0; i < ckm.length; i++) {
			for (int j = 0; j < ckm[i].length; j++) {
				SubSamplingKernal ck = (SubSamplingKernal)fms[j].getKernals()[ckIds[i][j][0]];
				ckm[i][j] = ck.wW;
			}
		}
				
		//caculate outM 
//		double [][] ckm = layer.getCkM();
		layer.setOutM(MathUtil.multiple(preM, ckm));
		
		//copy back2 fms;
		int [][][] oIds = layer.getOutIds();
		double [][] outM = layer.getOutM();
		for (int i = 0; i < outM.length; i++) {
			for (int j = 0; j < outM[i].length; j++) {
				double [][] pfs = fms[j].getzZs();
				pfs[oIds[i][j][0]][oIds[i][j][1]] = outM[i][j];
			}
		}
		
		for (int i = 0; i < fms.length; i++) {
			CNNUtils.activateZzs(bp, fms[i]);
		}	
	
	}

	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) {

	}

}
