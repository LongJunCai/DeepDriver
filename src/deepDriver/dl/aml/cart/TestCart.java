package deepDriver.dl.aml.cart;

public class TestCart {
		
	public static void main(String[] args) {
		DataSet ds = creatDs(1);
		Cart cart = new Cart();		
		cart.trainTree(ds);
		double [] ys = cart.predict(ds);
		for (int i = 0; i < ys.length; i++) {
			System.out.println(ds.getLabels()[i]+","+ds.getIndependentVars()[i]+","+ys[i]);
		}
		DataSet ds1 = creatDs(0.75);
		cart.lookupBestTree(ds1);
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
