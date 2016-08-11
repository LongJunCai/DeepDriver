package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.ann.IActivationFunction;

public class CNNFV4MaxPooling extends CNNForwardVisitor {
	
	public void sampling(SubSamplingKernal ck, double [][] ffms, IFeatureMap t2fm, boolean begin, IActivationFunction acf) {
		SamplingFeatureMap sfm = (SamplingFeatureMap) t2fm;
		FeatureMapTag [][] fmts = sfm.getFmts();
		for (int i = 0; i < t2fm.getFeatures().length; i++) {
			for (int j = 0; j < t2fm.getFeatures()[i].length; j++) {
				boolean init = false;
				double cs = 0;
				for (int j2 = 0; j2 < ck.ckRows; j2++) {
					for (int k = 0; k < ck.ckColumns; k++) {
						int fr = i * ck.ckRows + j2;
						int fc = j * ck.ckColumns + k;
						/*auto padding
						 * **/
						if (fr >= ffms.length || fc >= ffms[0].length) {
							continue;
						}/*auto padding
						 * **/
						double t = ffms[fr][fc];
						if (!init) {
							cs =  t;
							init = true;
							fmts[i][j].r = j2;
							fmts[i][j].c = k;
						} else {
							if (cs < t) {
								cs = t;
								fmts[i][j].r = j2;
								fmts[i][j].c = k;
							}
						}						
					}
				}
				cs = cs * ck.wW;
				if (!bp.useGlobalWeight) {
					cs = cs + ck.b;
				}				
//				double acs = acf.activate(cs);
				if (begin) {
//					t2fm.getFeatures()[i][j] = acs;
					t2fm.getzZs()[i][j] = cs;
				} else {
//					t2fm.getFeatures()[i][j] = t2fm.getFeatures()[i][j] + acs;
					t2fm.getzZs()[i][j] = t2fm.getzZs()[i][j] + cs;
				}
			}
		}
	}

}
