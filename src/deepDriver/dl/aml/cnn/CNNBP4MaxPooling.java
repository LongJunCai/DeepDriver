package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.ann.IActivationFunction;

public class CNNBP4MaxPooling extends CNNBP {

	public CNNBP4MaxPooling(CNNConfigurator cfg) {
		super(cfg);
		this.fv = new CNNFV4MaxPooling();
		fv.bp = this;
	}
	
	
	public void bpSampling(SubSamplingKernal ck, IFeatureMap ffm, IFeatureMap t2fm, IActivationFunction acf, 
			boolean begin) {
		SamplingFeatureMap sfm = (SamplingFeatureMap) t2fm;
		FeatureMapTag [][] fmts = sfm.getFmts();
		for (int i = 0; i < t2fm.getDeltaZzs().length; i++) {
			for (int j = 0; j < t2fm.getDeltaZzs()[i].length; j++) {
				if (begin) {
					t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j] 
						* acf.deActivate(t2fm.getzZs()[i][j]);
				}				
				if (!ck.initB) {
					ck.initB = true;
					ck.deltab = cfg.getM() * ck.deltab - cfg.getL() * t2fm.getDeltaZzs()[i][j];
				} else {
					ck.deltab = ck.deltab - cfg.getL() * t2fm.getDeltaZzs()[i][j];
				}
				
				FeatureMapTag fmt = fmts[i][j];
				int fr = i * ck.ckRows + fmt.getR();
				int fc = j * ck.ckColumns + fmt.getC();
				/*auto padding
				 * **/
				if (fr >= ffm.getFeatures().length || fc >= ffm.getFeatures()[0].length) {
					continue;
				}/*auto padding
				 * **/
				if (!ffm.getInitDeltaZzs()[fr][fc]) {
					ffm.getInitDeltaZzs()[fr][fc] = true;							
					ffm.getDeltaZzs()[fr][fc] = t2fm.getDeltaZzs()[i][j] * ck.wW;
				} else {
					ffm.getDeltaZzs()[fr][fc] = ffm.getDeltaZzs()[fr][fc] + 
							t2fm.getDeltaZzs()[i][j] * ck.wW;
				}
				
				if (!ck.initwW) {
					ck.initwW = true;
					ck.deltawW = cfg.getM() * ck.deltawW 
							- cfg.getL() * t2fm.getDeltaZzs()[i][j] * ffm.getFeatures()[fr][fc];
				} else {
					ck.deltawW = ck.deltawW 
							- cfg.getL() * t2fm.getDeltaZzs()[i][j] * ffm.getFeatures()[fr][fc];
				}
				
//				double cs = 0;
//				for (int j2 = 0; j2 < ck.ckRows; j2++) {
//					for (int k = 0; k < ck.ckColumns; k++) {
//						
//					}
//				}				
			}
		}
	}

}
