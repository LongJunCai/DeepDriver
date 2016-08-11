package deepDriver.dl.aml.linearReg.test;


import deepDriver.dl.aml.cart.DataSet;
import deepDriver.dl.aml.linearReg.LinearExpression;
import deepDriver.dl.aml.linearReg.LinearRegression;
import deepDriver.dl.aml.utils.AccuracyCaculator;

public class TestSelfProduced {
	public static DataSet  creatDs(double coef) {
		int cnt = 7;
		int columns = 2;
		double [] [] vars = new double[cnt][columns];
		double [] inVars = new double[cnt];
		double [] pv = {79.32 ,87.28 ,455.35 ,113.91 ,86.15 ,179.35 ,248.71,127.5,46.00 ,88.15,145.62 ,61.30 ,28.40 ,41.04};
		double [] vv = {145.603105,146.592232,769.615582,214.942849,170.157229,368.989821,525.367893,461.566178,200.951821,405.199137,911.204181,460.704428,235.400016,367.576954};
		String [] lables = new String[cnt];
		for (int i = 0; i < cnt; i++) {
			vars[i] = new double[columns];
			vars[i][0] = pv[i];
			vars[i][1] = 1;
			inVars[i] = vv[i];
			lables[i] = "epi"+i;
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
		double [] ts = le.getThetas();
		for (int i = 0; i < ts.length; i++) {
			System.out.println("t"+i+":"+ts[i]);
		}
		
		AccuracyCaculator acc = new AccuracyCaculator();
		double ac = acc.caculateAccuracy(ds.getIndependentVars(), ys);
		System.out.println("ac:"+ac);
	}

}
