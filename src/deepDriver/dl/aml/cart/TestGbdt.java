package deepDriver.dl.aml.cart;

public class TestGbdt {
	
	public static void main(String[] args) {
		DataSet ds = creatDs(0.9);
		DataSet ds1 = creatDs(0.65);
		DataSet ds2 = creatDs(0.8);
		Gbdt gbdt = new Gbdt();
		gbdt.train(ds, ds1);
		double [] ys = gbdt.test(ds2);
		for (int i = 0; i < ys.length; i++) {
			System.out.println(ds2.getLabels()[i]+","+ds2.getIndependentVars()[i]+","+ys[i]);
		}
		
//		cart.lookupBestTree(ds1);
	}
	
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

}
