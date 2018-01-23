package deepDriver.dl.aml.cnn;

public class CNNUtils {
	
	public void batchNorm(CNNConfigurator cfg, IFeatureMap t2fm) {
		double [][] y = t2fm.getzZs();
		double [][] xi = t2fm.getoZzs();
		double [][] dy = t2fm.getDeltaZzs();
		
		double [][] dxm = new double[y.length][y[0].length];
		double [][] dxi = new double[xi.length][xi[0].length]; 
		double dvar = 0;
		double du = 0;
		double dgamma = 0;
		double dbeta = 0;
		
		double du1 = 0;
		double du2 = 0;
		double pq = (double)(y.length * y[0].length);
		for (int i = 0; i < dy.length; i++) {
			for (int j = 0; j < dy[i].length; j++) {
				t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j]
						* t2fm.getAcf().deActivate(t2fm.getzZs()[i][j]);
				
				dxm[i][j] = dy[i][j] * t2fm.getGema();
				dvar = dvar + dxm[i][j] * (xi[i][j] - t2fm.getU()) 
						* (-0.5) * Math.pow((t2fm.getVar2() + t2fm.getE()), -1.5);
				du1 = du1 - dxm[i][j]/Math.sqrt(t2fm.getVar2() + t2fm.getE());
				du2 = du2 - 2 * (xi[i][j] - t2fm.getU());
				
				double xm = (xi[i][j] - t2fm.getU())/Math.sqrt(t2fm.getVar2() + t2fm.getE());
				if (i == 0 && j == 0) {
					dgamma = cfg.getM() * t2fm.getDgamma() - cfg.getL() * dy[i][j] * xm;
					dbeta = cfg.getM() * t2fm.getDbeta() - cfg.getL() * dy[i][j];
				} else {
					dgamma = dgamma - cfg.getL() * dy[i][j] *  xm;
					dbeta = dbeta - cfg.getL() * dy[i][j];
				}				
			}
		}
		du = du1 + dvar * du2/pq;
		t2fm.setDgamma(dgamma);
		t2fm.setDbeta(dbeta);
		
		for (int i = 0; i < dxi.length; i++) {
			for (int j = 0; j < dxi[0].length; j++) {
				dxi[i][j] = dxm[i][j]/Math.sqrt(t2fm.getVar2() + t2fm.getE())
						+ dvar * 2 * (xi[i][j] - t2fm.getU())/pq + du/pq;
				dy[i][j] = dxi[i][j];
			}
		}
	}
	
	public static void deActiveDZzs(IFeatureMap t2fm) {
		double [][] dy = t2fm.getDeltaZzs();
		for (int i = 0; i < dy.length; i++) {
			for (int j = 0; j < dy[i].length; j++) {
				t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j]
						* t2fm.getAcf().deActivate(t2fm.getzZs()[i][j]);
			}
		}
	}
	
	public static void deActivateGlobal(CNNBP bp, IFeatureMap fm) {
		if (CNNUtils.useBN(bp.cfg, fm) || !bp.useGlobalWeight) {
			return;
		} 
		double [][] dzZ = fm.getDeltaZzs();
		fm.setInitBb(false);
		for (int i = 0; i < dzZ.length; i++) {
			for (int j = 0; j < dzZ[i].length; j++) {
				if (!fm.isInitBb()) {
					fm.setInitBb(true);
					fm.setDeltaBb(fm.getDeltaBb() * bp.cfg.getM()
							- bp.cfg.getL() * dzZ[i][j]);
				} else {
					fm.setDeltaBb(fm.getDeltaBb()
							- bp.cfg.getL() * dzZ[i][j]);
				}				
			}
		}
	}
	
	public static boolean useBN(CNNConfigurator cfg, IFeatureMap t2fm) {
		if (cfg.isUseBN()) {
			if (t2fm.getFeatures().length * t2fm.getFeatures()[0].length > 1) {
				return true;
			}
		}
		return false;
	}
	
	public static void activateConvZzs(CNNBP bp, CNNConfigurator cfg, IFeatureMap t2fm) {
		if (useBN(cfg, t2fm)) {
			batchNorm(t2fm);
		}		
		for (int i = 0; i < t2fm.getFeatures().length; i++) {
			for (int j = 0; j < t2fm.getFeatures()[i].length; j++) {
				//use global 
				if (!useBN(cfg, t2fm) && bp.useGlobalWeight) {
					t2fm.getzZs()[i][j] = t2fm.getzZs()[i][j] + t2fm.getbB();
				}				
				t2fm.getFeatures()[i][j] = t2fm.getAcf().activate(
						t2fm.getzZs()[i][j]);
			}
		}
	}	
	
	public static void batchNorm(IFeatureMap t2fm) {
		double sum = 0;
		double [][] z = t2fm.getzZs();
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < z[i].length; j++) {
				sum = sum + z[i][j];
			}
		}
		double pq = (double)(z.length * z[0].length);
		t2fm.setU(sum/pq);
		sum = 0;
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < z[i].length; j++) {
				double a = (z[i][j] - t2fm.getU());
				sum = sum + a * a;
			}
		}
		t2fm.setVar2(sum/pq);
		
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < z[i].length; j++) { 
				t2fm.getoZzs()[i][j] = z[i][j];
				double y = t2fm.getBeta() + t2fm.getGema() * 
						(z[i][j] - t2fm.getU())/Math.sqrt(t2fm.getVar2() + t2fm.getE()) ;						
				z[i][j] = y;
			}
		}
		
		t2fm.setSumU(t2fm.getSumU() + t2fm.getU());
		t2fm.setSumVar2(t2fm.getSumVar2() + t2fm.getVar2());
		t2fm.setSamplesCnt(t2fm.getSamplesCnt() + 1);
	}

}
