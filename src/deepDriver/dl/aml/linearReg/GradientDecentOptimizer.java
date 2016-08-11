package deepDriver.dl.aml.linearReg;

public class GradientDecentOptimizer {
	
	double precision = 0.00000000000001;
	double alpha = 0.1;
	int maxLoop = 50000;
	
	boolean doesScaleOrNot = true;
	public double [] optimizeFunction(ISubject2Optimized subject, 
			double[][] xVector, double[] y, boolean doesScaleOrNot) {
		if (!doesScaleOrNot) {
			subject.initSubjectFunction(xVector, y);
			return optimizeFunction(subject);
		} else {
			ParameterScaler ps = new ParameterScaler();
			subject.initSubjectFunction(ps.scaleParameters(xVector), y);
			return ps.scaleCoefficients(optimizeFunction(subject));
		}		
	}
	public double [] optimizeFunction(ISubject2Optimized subject, 
			double[][] xVector, double[] y) {
		return optimizeFunction(subject, 
				xVector, y, doesScaleOrNot);
	}
	public double [] optimizeFunction(ISubject2Optimized subject) {		
		double [] thetas = new double[subject.getThetasNum()];
		double [] initThetas = subject.getInitTheta();
		if (initThetas != null) {
			for (int i = 0; i < thetas.length; i++) {
				thetas[i] = initThetas[i];
			}
		}
		double [] oldThetas = new double[subject.getThetasNum()];
		double old = subject.cacluateSubject(thetas);
		double newValue = -old;
		double residual = old;	
		while (!Double.isInfinite(residual) && residual > precision ) {			
			subject.updateThetas(thetas);
			copyTheta(thetas, oldThetas) ;
			for (int i = 0; i < thetas.length; i++) {
				thetas[i] = thetas[i] - alpha * subject.getThetaDecent(i);
			}			
			newValue = subject.cacluateSubject(thetas);
			double tmpResidual = Math.abs(old - newValue);
//			System.out.println("("+newValue+"):("+old+"):("+residual+")="+thetas[0]+"="+thetas[1]);
			if (tmpResidual > residual) {
				alpha = alpha/3.0;
//				copyTheta(oldThetas, thetas) ;
//				continue;
			}
			residual = tmpResidual;			
			old = newValue;
		}		
		return thetas;
	}
	
	public void copyTheta(double [] from, double [] to) {
		for (int i = 0; i < to.length; i++) {
			to[i] = from[i];
		}
	}
	
}
