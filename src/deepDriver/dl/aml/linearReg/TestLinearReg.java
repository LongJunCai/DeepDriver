package deepDriver.dl.aml.linearReg;

import deepDriver.dl.aml.cart.DataSet;

public class TestLinearReg {
	
	public static DataSet  creatDs(double coef) {
		int cnt = 5;
		int columns = 6;
		double [] [] vars = new double[cnt][columns];
		double [] inVars = new double[cnt];
		String [] lables = new String[cnt];
		for (int i = 0; i < cnt; i++) {
			vars[i] = new double[columns];
			for (int j = 0; j < columns; j++) {
				if (j == 0) {
					vars[i][j] = i ;
				} else {
					vars[i][j] = i + j;
				}
				
			}
			inVars[i] =coef *  i+1 ;
			lables[i] = "la"+i;
		}
		DataSet ds = new DataSet();
		ds.setDependentVars(vars);
		ds.setIndependentVars(inVars);
		ds.setLabels(lables);
		return ds;
	}
	
	public static void main(String[] args) {
		LinearRegression reg = new LinearRegression();
		LinearExpression le = reg.fit(creatDs(1));
		DataSet ds = creatDs(0.75);
		double [] ys = le.predict(ds);
		for (int i = 0; i < ys.length; i++) {
			System.out.println(ds.getLabels()[i]+","+ds.getIndependentVars()[i]+","+ys[i]);
		}
	}

}
